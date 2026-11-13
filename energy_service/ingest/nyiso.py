"""Incremental, idempotent NYISO ingestion with provenance.

Daily files exist only for the current month; completed months are consolidated into monthly zips
(see docs/operational-timing.md). The existing, DST-tested parsers in src/data_loader.py are reused
unchanged; a daily CSV is wrapped in an in-memory zip to feed them.
"""
from __future__ import annotations

import hashlib
import io
import sqlite3
import zipfile
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from energy_service.config import Settings
from energy_service.db import JobRun, iso, log_event
from energy_service.http import FetchError, NotYetPublished
from energy_service.ingest.validate import (
    ConflictError,
    Flag,
    SchemaError,
    validate_frame,
)
from src import data_loader as dl

ET = ZoneInfo("America/New_York")
BASE = "https://mis.nyiso.com/public/csv"

DATASETS = {
    "dam_price": dict(daily="damlbmp/{d}damlbmp_zone.csv", monthly_ds="damlbmp_zone", table="prices_dam",
                      col="lbmp", parse_csv=lambda b: dl.parse_dam_lbmp(_as_zip(b)),
                      parse_zip=lambda b: dl.parse_dam_lbmp(io.BytesIO(b)), bounds=("price_min", "price_max")),
    "actual_load": dict(daily="palIntegrated/{d}palIntegrated.csv", monthly_ds="palIntegrated", table="load_actual",
                        col="load_mw", parse_csv=lambda b: dl.parse_integrated_load(_as_zip(b)),
                        parse_zip=lambda b: dl.parse_integrated_load(io.BytesIO(b)), bounds=("load_min", "load_max")),
}


def _as_zip(csv_bytes: bytes) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("file.csv", csv_bytes)
    buf.seek(0)
    return buf


def daily_url(dataset: str, d: date) -> str:
    return f"{BASE}/{DATASETS[dataset]['daily'].format(d=d.strftime('%Y%m%d'))}"


def monthly_url(dataset: str, year: int, month: int) -> str:
    return dl.nyiso_url(DATASETS[dataset]["monthly_ds"], year, month)


def expected_available(dataset: str, delivery: date, now: datetime, s: Settings) -> bool:
    """Has the file for `delivery` normally been published by `now`? (empirical times, docs/operational-timing.md)"""
    if dataset == "dam_price":
        h, m = map(int, s.dam_publish_et.split(":"))
        due = datetime.combine(delivery - timedelta(days=1), time(h, m), tzinfo=ET)
    else:
        h, m = map(int, s.load_publish_et.split(":"))
        due = datetime.combine(delivery + timedelta(days=1), time(h, m), tzinfo=ET)
    return now >= due


def _add_flags(conn, now, dataset, flags, raw_id, delivery_date=None):
    for f in flags:
        conn.execute("INSERT INTO quality_flags (created_at, dataset, delivery_date, code, severity, detail, raw_file_id) "
                     "VALUES (?,?,?,?,?,?,?)", (iso(now), dataset, f.delivery_date or delivery_date, f.code, f.severity,
                                                f.detail, raw_id))
    conn.commit()


def _upsert(conn, table, col, zone, df, raw_id) -> tuple[int, int]:
    keys = [iso(t) for t in df["timestamp_utc"]]
    existing = {r[0]: r[1] for r in conn.execute(
        f"SELECT ts_utc, {col} FROM {table} WHERE zone=? AND ts_utc BETWEEN ? AND ?", (zone, min(keys), max(keys)))}
    vals = df[col].astype(float).tolist()
    changed = sum(1 for k, v in zip(keys, vals) if k in existing and abs(existing[k] - v) > 1e-9)
    new = sum(1 for k in keys if k not in existing)
    conn.executemany(
        f"INSERT INTO {table} (zone, ts_utc, {col}, raw_file_id) VALUES (?,?,?,?) "
        f"ON CONFLICT(zone, ts_utc) DO UPDATE SET {col}=excluded.{col}, raw_file_id=excluded.raw_file_id",
        [(zone, k, v, raw_id) for k, v in zip(keys, vals)])
    return new, changed


def store_file(conn: sqlite3.Connection, s: Settings, dataset: str, fetched, *, granularity: str,
               delivery: date | None, now: datetime) -> str:
    """Parse, validate and upsert one fetched file. Idempotent: identical content for a URL is skipped.
    Returns ingested | unchanged | rejected."""
    spec = DATASETS[dataset]
    sha = hashlib.sha256(fetched.body).hexdigest()
    if conn.execute("SELECT 1 FROM raw_files WHERE url=? AND sha256=?", (fetched.url, sha)).fetchone():
        return "unchanged"
    lm = iso(fetched.last_modified) if fetched.last_modified else None
    cur = conn.execute(
        "INSERT INTO raw_files (dataset, url, granularity, delivery_date, last_modified, sha256, bytes, fetched_at, status) "
        "VALUES (?,?,?,?,?,?,?,?, 'pending')",
        (dataset, fetched.url, granularity, str(delivery) if delivery else None, lm, sha, len(fetched.body), iso(now)))
    raw_id = cur.lastrowid
    try:
        parsed = (spec["parse_csv"] if granularity == "daily" else spec["parse_zip"])(fetched.body)
        lo, hi = (getattr(s, b) for b in spec["bounds"])
        v = validate_frame(parsed, zone=s.zone, value_col=spec["col"], lo=lo, hi=hi)
    except (SchemaError, ConflictError, dl.DataError, KeyError, ValueError, zipfile.BadZipFile, pd.errors.ParserError) as e:
        conn.execute("UPDATE raw_files SET status='rejected', detail=? WHERE id=?", (f"{type(e).__name__}: {e}", raw_id))
        conn.commit()
        _add_flags(conn, now, dataset, [Flag("file_rejected", "error", str(e))], raw_id, str(delivery) if delivery else None)
        log_event(conn, "error", "file_rejected", f"{dataset} file rejected: {e}", {"url": fetched.url}, now)
        return "rejected"
    new, changed = _upsert(conn, spec["table"], spec["col"], s.zone, v.df, raw_id)
    detail = f"rows={len(v.df)} new={new} revised={changed}"
    conn.execute("UPDATE raw_files SET status='ingested', detail=? WHERE id=?", (detail, raw_id))
    _add_flags(conn, now, dataset, v.flags, raw_id, str(delivery) if delivery else None)
    if changed:
        _add_flags(conn, now, dataset, [Flag("revised_values", "info", f"{changed} existing values changed")],
                   raw_id, str(delivery) if delivery else None)
    conn.commit()
    return "ingested"


def _fetch_and_store(conn, s, dataset, url, *, granularity, delivery, fetch, now, flag_missing=True) -> str:
    try:
        f = fetch(url)
    except NotYetPublished:
        if delivery is not None and expected_available(dataset, delivery, now, s):
            if flag_missing:
                _add_flags(conn, now, dataset, [Flag("expected_file_missing", "warning", url, str(delivery))], None, str(delivery))
            return "expected_missing"
        return "not_published"
    except FetchError as e:
        log_event(conn, "warning", "fetch_failed", str(e), {"dataset": dataset}, now)
        return "error"
    return store_file(conn, s, dataset, f, granularity=granularity, delivery=delivery, now=now)


def day_present(conn, dataset: str, zone: str, d: date) -> bool:
    """Any stored row for the local delivery date `d`?"""
    spec = DATASETS[dataset]
    lo = pd.Timestamp(d).tz_localize(ET).tz_convert("UTC")
    hi = pd.Timestamp(d + timedelta(days=1)).tz_localize(ET).tz_convert("UTC")
    return conn.execute(f"SELECT 1 FROM {spec['table']} WHERE zone=? AND ts_utc>=? AND ts_utc<? LIMIT 1",
                        (zone, iso(lo), iso(hi))).fetchone() is not None


def run_ingest(conn, s: Settings, *, now: datetime, fetch, lookback_days: int = 3) -> dict:
    """Fetch the recent window. Daily files are the fast path (a rolling window of about 12 days); an earlier
    month in the window comes from its monthly zip. If an expected daily file has already rolled off, the
    current month's zip (which holds the whole month and is updated daily) is the fallback; a day is flagged
    missing only if that does not have it either. Day-ahead prices are fetched up to tomorrow (published
    ~09:32 ET today); actual load up to today."""
    today = now.astimezone(ET).date()
    first_of_month = today.replace(day=1)
    counts: dict[str, int] = {}

    def bump(k):
        counts[k] = counts.get(k, 0) + 1

    with JobRun(conn, "ingest", now) as job:
        for dataset in DATASETS:
            last = today + timedelta(days=1) if dataset == "dam_price" else today
            done_months: set[tuple[int, int]] = set()
            missing: list[date] = []
            d = today - timedelta(days=lookback_days)
            while d <= last:
                if d >= first_of_month:
                    out = _fetch_and_store(conn, s, dataset, daily_url(dataset, d), granularity="daily", delivery=d,
                                           fetch=fetch, now=now, flag_missing=False)
                    if out == "expected_missing":
                        missing.append(d)
                        d += timedelta(days=1)
                        continue
                elif (d.year, d.month) not in done_months:
                    done_months.add((d.year, d.month))
                    out = _fetch_and_store(conn, s, dataset, monthly_url(dataset, d.year, d.month), granularity="monthly",
                                           delivery=None, fetch=fetch, now=now)
                else:
                    d += timedelta(days=1)
                    continue
                bump(f"{dataset}:{out}")
                d += timedelta(days=1)
            if missing:
                out = _fetch_and_store(conn, s, dataset, monthly_url(dataset, today.year, today.month), granularity="monthly",
                                       delivery=None, fetch=fetch, now=now)
                bump(f"{dataset}:monthly_fallback_{out}")
                for m in missing:
                    if day_present(conn, dataset, s.zone, m):
                        bump(f"{dataset}:recovered_from_monthly_zip")
                    else:
                        bump(f"{dataset}:expected_missing")
                        _add_flags(conn, now, dataset, [Flag("expected_file_missing", "warning", daily_url(dataset, m), str(m))],
                                   None, str(m))
        job.details = counts
        if any(k.endswith(":error") or k.endswith("_error") for k in counts):
            job.status = "partial"
        if counts and all(k.endswith(":error") for k in counts):
            job.status = "failed"
    return {"status": job.status, "counts": counts}


def run_backfill(conn, s: Settings, *, start: str, end: str, fetch, now: datetime) -> dict:
    """Load completed months start..end ('YYYY-MM') from monthly zips. Re-running is a no-op."""
    counts: dict[str, int] = {}
    cur_month = now.astimezone(ET).date().replace(day=1)
    with JobRun(conn, "backfill", now) as job:
        for p in pd.period_range(start, end, freq="M"):
            if date(p.year, p.month, 1) >= cur_month:
                continue  # the current month comes from daily files
            for dataset in DATASETS:
                out = _fetch_and_store(conn, s, dataset, monthly_url(dataset, p.year, p.month), granularity="monthly",
                                       delivery=None, fetch=fetch, now=now)
                counts[f"{dataset}:{out}"] = counts.get(f"{dataset}:{out}", 0) + 1
        job.details = counts
        if any(k.endswith(":error") for k in counts):
            job.status = "partial"
    return {"status": job.status, "counts": counts}
