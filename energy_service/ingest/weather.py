"""Weather ingestion into SQLite: live forecast snapshots (for operation) and archived series (for training).

Weather data by Open-Meteo.com (CC BY 4.0; free API is non-commercial). A live snapshot is stored with its
retrieval time because live forecasts are revised: the stored snapshot is what the forecast actually used.
Missing hours stay NULL and are never filled.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from urllib.parse import urlencode

import pandas as pd

from energy_service.config import Settings
from energy_service.db import JobRun, iso, log_event
from energy_service.http import FetchError, NotYetPublished

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "gfs_global"  # not gfs_seamless (can blend HRRR)


class WeatherError(Exception):
    pass


def _parse(body: bytes, var: str) -> pd.DataFrame:
    try:
        h = json.loads(body)["hourly"]
        times, vals = h["time"], h[var]
    except (KeyError, ValueError, TypeError) as e:
        raise WeatherError(f"unexpected Open-Meteo response (no hourly.{var}): {e}") from e
    if len(times) != len(vals) or not times:
        raise WeatherError("Open-Meteo hourly arrays empty or of different length")
    return pd.DataFrame({"timestamp_utc": pd.to_datetime(times, utc=True),
                         "temp_c": pd.to_numeric(pd.Series(vals), errors="coerce")})


def _store(conn: sqlite3.Connection, kind: str, df: pd.DataFrame, body: bytes, *, now: datetime, model, lead, s: Settings) -> int:
    cur = conn.execute(
        "INSERT INTO weather_snapshots (kind, model, lead_days, retrieved_at, lat, lon, n_hours, sha256) VALUES (?,?,?,?,?,?,?,?)",
        (kind, model, lead, iso(now), s.lat, s.lon, len(df), hashlib.sha256(body).hexdigest()))
    sid = cur.lastrowid
    conn.executemany("INSERT INTO weather_hours (snapshot_id, ts_utc, temp_c) VALUES (?,?,?)",
                     [(sid, iso(t), None if pd.isna(v) else float(v)) for t, v in zip(df["timestamp_utc"], df["temp_c"])])
    conn.commit()
    return sid


def snapshot_live_forecast(conn, s: Settings, *, now: datetime, fetch) -> int | None:
    """Store the current GFS hourly 2 m temperature forecast. Returns the snapshot id, or None on failure
    (failure is logged; the forecast job then falls back to the baseline)."""
    q = urlencode({"latitude": s.lat, "longitude": s.lon, "hourly": "temperature_2m", "models": MODEL,
                   "forecast_days": 4, "timezone": "UTC"})
    with JobRun(conn, "weather_snapshot", now) as job:
        try:
            f = fetch(f"{FORECAST_URL}?{q}")
            df = _parse(f.body, "temperature_2m")
        except (FetchError, NotYetPublished, WeatherError) as e:
            log_event(conn, "warning", "weather_snapshot_failed", str(e), None, now)
            job.status = "failed"
            job.details = {"error": str(e)}
            return None
        sid = _store(conn, "live_forecast", df, f.body, now=now, model=MODEL, lead=None, s=s)
        job.details = {"snapshot_id": sid, "n_hours": len(df), "n_null": int(df["temp_c"].isna().sum())}
        return sid


def backfill_archive(conn, s: Settings, *, start: str, end: str, now: datetime, fetch) -> int:
    """Observed ('actual') hourly temperature for training (archive API; the most recent days may be null)."""
    q = urlencode({"latitude": s.lat, "longitude": s.lon, "start_date": start, "end_date": end,
                   "hourly": "temperature_2m", "timezone": "UTC"})
    f = fetch(f"{ARCHIVE_URL}?{q}")
    return _store(conn, "archive_actual", _parse(f.body, "temperature_2m"), f.body, now=now, model=None, lead=None, s=s)


def backfill_previous_runs(conn, s: Settings, *, start: str, end: str, lead_days: int, now: datetime, fetch) -> list[int]:
    """Archived forecasts issued `lead_days` before the valid time, one request per calendar year."""
    var = f"temperature_2m_previous_day{lead_days}"
    ids = []
    for yr in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        lo, hi = max(pd.Timestamp(start), pd.Timestamp(yr, 1, 1)), min(pd.Timestamp(end), pd.Timestamp(yr, 12, 31))
        q = urlencode({"latitude": s.lat, "longitude": s.lon, "start_date": str(lo.date()), "end_date": str(hi.date()),
                       "hourly": var, "models": MODEL, "timezone": "UTC"})
        f = fetch(f"{PREVIOUS_RUNS_URL}?{q}")
        ids.append(_store(conn, "previous_runs", _parse(f.body, var), f.body, now=now, model=MODEL, lead=lead_days, s=s))
    return ids
