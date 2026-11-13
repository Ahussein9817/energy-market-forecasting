"""SYNTHETIC demo data. Never mix with real data: `seed_demo` refuses to run on a database that already holds prices.
Used by the `demo-seed` command (so a fresh container shows something) and by the tests."""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from energy_service.db import log_event


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



def seed_demo(conn, s, now: datetime) -> dict:
    """Synthetic prices, load and weather through today (America/New_York), a live weather snapshot and a baseline forecast."""
    from energy_service.forecast import generate_forecast
    if conn.execute("SELECT 1 FROM prices_dam LIMIT 1").fetchone():
        raise RuntimeError("refusing to seed: the database already contains price data")
    today = now.astimezone(__import__("zoneinfo").ZoneInfo("America/New_York")).date()
    build_world(conn, today, days=150)
    add_live_snapshot(conn, today + timedelta(days=1), now - timedelta(hours=1))
    log_event(conn, "warning", "demo_seed", "SYNTHETIC demo data loaded; not NYISO data", None, now)
    r = generate_forecast(conn, s, origin_date=today, now=now, source="manual")
    return {"origin_date": str(today), "target_date": r.target_date, "served": r.served_kind, "note": "synthetic data"}
