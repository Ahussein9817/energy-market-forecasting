"""Synthetic NYISO-format files and a fake site for offline tests (no real NYISO data is committed)."""
from __future__ import annotations

import io
import zipfile
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from energy_service.http import Fetched, FetchError, NotYetPublished
from energy_service.ingest import nyiso

DAM_HEAD = "Time Stamp,Name,PTID,LBMP ($/MWHr),Marginal Cost Losses ($/MWHr),Marginal Cost Congestion ($/MWHr)\n"
LOAD_HEAD = "Time Stamp,Time Zone,Name,PTID,Integrated Load\n"


def local_hours(day: date) -> list[pd.Timestamp]:
    d = pd.Timestamp(day)
    return list(pd.date_range(d, d + pd.Timedelta(days=1), freq="h", tz="America/New_York", inclusive="left"))


def dam_csv(day: date, price=lambda h: 30.0 + h, zones=("N.Y.C.", "WEST"), drop_hours=(), extra_rows="") -> bytes:
    rows = []
    for z in zones:
        for h, t in enumerate(local_hours(day)):
            if h in drop_hours:
                continue
            rows.append(f"{t.strftime('%m/%d/%Y %H:%M')},{z},1,{price(h) if z == 'N.Y.C.' else 20.0},1.0,0.5\n")
    return (DAM_HEAD + "".join(rows) + extra_rows).encode()


def load_csv(day: date, load=lambda h: 5000.0 + 10 * h, zone="N.Y.C.") -> bytes:
    rows = []
    for h, t in enumerate(local_hours(day)):
        rows.append(f"{t.strftime('%m/%d/%Y %H:%M:%S')},{t.tzname()},{zone},1,{load(h)}\n")
    return (LOAD_HEAD + "".join(rows)).encode()


def zip_of(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for n, c in files.items():
            zf.writestr(n, c)
    return buf.getvalue()


class FakeSite:
    """Callable fetch(url) -> Fetched backed by a dict. Missing URL -> NotYetPublished; value Exception -> raised."""

    def __init__(self):
        self.files: dict[str, object] = {}
        self.calls: list[str] = []

    def put_daily(self, dataset, day, body, last_modified=None):
        self.files[nyiso.daily_url(dataset, day)] = (body, last_modified)

    def put_monthly(self, dataset, year, month, body, last_modified=None):
        self.files[nyiso.monthly_url(dataset, year, month)] = (body, last_modified)

    def fail(self, url, exc=None):
        self.files[url] = exc or FetchError(f"{url}: HTTP 503")

    def __call__(self, url):
        self.calls.append(url)
        v = self.files.get(url)
        if v is None:
            raise NotYetPublished(url)
        if isinstance(v, Exception):
            raise v
        body, lm = v
        return Fetched(body, lm or datetime(2026, 11, 12, 14, 33, tzinfo=UTC), url)


def populate_days(site: FakeSite, start: date, end: date, load_end: date | None = None):
    d = start
    while d <= end:
        site.put_daily("dam_price", d, dam_csv(d))
        if load_end is None or d <= load_end:
            site.put_daily("actual_load", d, load_csv(d))
        d += timedelta(days=1)


# ---- a deterministic synthetic world inside the database ---------------------------------------
def build_world(conn, end: date, days: int = 150, seed: int = 0, with_load_through: date | None = None):
    """Insert `days` of hourly prices (through `end`), load (through `with_load_through`, default end-1),
    and observed temperature (through `end`) for the N.Y.C. zone. Price depends on hour, weekday and temperature."""
    import numpy as np
    rng = np.random.default_rng(seed)
    start = end - timedelta(days=days - 1)
    idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1), freq="h", tz="America/New_York",
                        inclusive="left")
    utc = idx.tz_convert("UTC")
    doy = np.array([t.day_of_year for t in idx])
    temp = 10 + 12 * np.sin((doy - 100) / 365 * 2 * np.pi) + 3 * np.sin(np.array([t.hour for t in idx]) / 24 * 2 * np.pi) \
        + rng.normal(0, 1.5, len(idx))
    hour = np.array([t.hour for t in idx])
    price = 35 + 8 * np.sin(hour / 24 * 2 * np.pi - 1.5) + 2.0 * (np.array([t.day_of_week for t in idx]) < 5) \
        + 0.9 * np.abs(temp - 18) + rng.normal(0, 1.0, len(idx))
    load = 5500 + 120 * np.abs(temp - 18) + 600 * np.sin(hour / 24 * 2 * np.pi - 2.0) + rng.normal(0, 40, len(idx))
    rid = conn.execute("INSERT INTO raw_files (dataset, url, granularity, sha256, bytes, fetched_at, status) "
                       "VALUES ('dam_price','synthetic','daily','x',0,'2026-01-01T00:00:00Z','ingested')").lastrowid
    conn.executemany("INSERT INTO prices_dam (zone, ts_utc, lbmp, raw_file_id) VALUES ('N.Y.C.',?,?,?)",
                     [(t.strftime("%Y-%m-%dT%H:%M:%SZ"), float(p), rid) for t, p in zip(utc, price)])
    load_end = pd.Timestamp(with_load_through or (end - timedelta(days=1))) + pd.Timedelta(days=1)
    conn.executemany("INSERT INTO load_actual (zone, ts_utc, load_mw, raw_file_id) VALUES ('N.Y.C.',?,?,?)",
                     [(t.strftime("%Y-%m-%dT%H:%M:%SZ"), float(v), rid) for t, v, loc in zip(utc, load, idx)
                      if loc.tz_localize(None) < load_end])
    sid = conn.execute("INSERT INTO weather_snapshots (kind, retrieved_at, n_hours, sha256) VALUES ('archive_actual',"
                       "'2026-01-01T00:00:00Z',?, 'x')", (len(idx),)).lastrowid
    conn.executemany("INSERT INTO weather_hours (snapshot_id, ts_utc, temp_c) VALUES (?,?,?)",
                     [(sid, t.strftime("%Y-%m-%dT%H:%M:%SZ"), float(v)) for t, v in zip(utc, temp)])
    conn.commit()
    return {"temp": dict(zip(idx, temp)), "price": dict(zip(idx, price))}


def add_live_snapshot(conn, target: date, retrieved_at: datetime, temp=12.0, null_hours=(), base=None):
    """Live forecast snapshot covering the local day `target` (UTC hours)."""
    idx = pd.date_range(pd.Timestamp(target) - pd.Timedelta(days=1), pd.Timestamp(target) + pd.Timedelta(days=2),
                        freq="h", tz="America/New_York", inclusive="left").tz_convert("UTC")
    sid = conn.execute("INSERT INTO weather_snapshots (kind, model, retrieved_at, n_hours, sha256) VALUES "
                       "('live_forecast','gfs_global',?,?, 'x')", (retrieved_at.strftime("%Y-%m-%dT%H:%M:%SZ"), len(idx))).lastrowid
    local_hours = idx.tz_convert("America/New_York")
    rows = []
    for t, lt in zip(idx, local_hours):
        v = None if (lt.date() == target and lt.hour in null_hours) else float(temp)
        rows.append((sid, t.strftime("%Y-%m-%dT%H:%M:%SZ"), v))
    conn.executemany("INSERT INTO weather_hours (snapshot_id, ts_utc, temp_c) VALUES (?,?,?)", rows)
    conn.commit()
    return sid
