"""Read model inputs back out of SQLite as pandas frames and local-date x hour panels."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import pandas as pd

from energy_service.db import iso
from src import panel as pn


class DataIncomplete(Exception):
    """Required input is missing or has gaps, so a forecast cannot be built from it."""


def _frame(conn, table, col, zone) -> pd.DataFrame:
    rows = conn.execute(f"SELECT ts_utc, {col} FROM {table} WHERE zone=? ORDER BY ts_utc", (zone,)).fetchall()
    return pd.DataFrame({"timestamp_utc": pd.to_datetime([r[0] for r in rows], utc=True),
                         col: [r[1] for r in rows]})


def prices_frame(conn, zone) -> pd.DataFrame:
    return _frame(conn, "prices_dam", "lbmp", zone)


def load_frame(conn, zone) -> pd.DataFrame:
    return _frame(conn, "load_actual", "load_mw", zone)


def panel_from(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Local-date x hour panel. Raises DataIncomplete on interior gaps (never interpolated)."""
    if df.empty:
        raise DataIncomplete(f"no {col} data")
    try:
        return pn.to_panel(df, col)
    except ValueError as e:
        raise DataIncomplete(str(e)) from e


def weather_frame(conn, kind: str, lead_days: int | None = None) -> pd.DataFrame:
    """Hourly temperature for one snapshot kind; where snapshots overlap the most recently retrieved wins."""
    sql = ("SELECT h.ts_utc, h.temp_c, s.retrieved_at FROM weather_hours h JOIN weather_snapshots s ON s.id=h.snapshot_id "
           "WHERE s.kind=?" + (" AND s.lead_days=?" if lead_days is not None else "") + " ORDER BY h.ts_utc, s.retrieved_at")
    rows = conn.execute(sql, (kind, lead_days) if lead_days is not None else (kind,)).fetchall()
    df = pd.DataFrame({"timestamp_utc": pd.to_datetime([r[0] for r in rows], utc=True),
                       "temp_fc_c": [r[1] for r in rows]})
    return df.drop_duplicates("timestamp_utc", keep="last").reset_index(drop=True)


def latest_live_snapshot(conn: sqlite3.Connection, now: datetime, max_age_hours: float):
    """Newest live forecast snapshot retrieved at or before `now`. Returns (row, frame, age_hours) or None."""
    row = conn.execute("SELECT * FROM weather_snapshots WHERE kind='live_forecast' AND retrieved_at<=? "
                       "ORDER BY retrieved_at DESC LIMIT 1", (iso(now),)).fetchone()
    if row is None:
        return None
    rows = conn.execute("SELECT ts_utc, temp_c FROM weather_hours WHERE snapshot_id=? ORDER BY ts_utc", (row["id"],)).fetchall()
    df = pd.DataFrame({"timestamp_utc": pd.to_datetime([r[0] for r in rows], utc=True), "temp_fc_c": [r[1] for r in rows]})
    age = (now - pd.Timestamp(row["retrieved_at"]).to_pydatetime()).total_seconds() / 3600
    return row, df, age


def day_after(d) -> str:
    return str(pd.Timestamp(d).date() + timedelta(days=1))
