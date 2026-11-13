"""Readiness and data-health summary shared by the CLI `status` command and the API `/ready`."""
from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from energy_service import db
from energy_service.config import Settings

ET = ZoneInfo("America/New_York")


def _latest_local_date(conn, table: str, zone: str):
    r = conn.execute(f"SELECT MAX(ts_utc) FROM {table} WHERE zone=?", (zone,)).fetchone()[0]
    if r is None:
        return None
    return pd.Timestamp(r).tz_convert(ET).date()


def readiness(conn, s: Settings, now: datetime) -> dict:
    """not_ready: the service cannot produce a trustworthy baseline (no or stale prices, no schema).
    degraded: it works but something reduces quality (model cannot serve, load stale, no forecast yet).
    Rules follow docs/operational-timing.md: on day D, prices through D are always public (D+1 from ~09:32 ET)."""
    now_et = now.astimezone(ET)
    today = now_et.date()
    reasons: list[str] = []
    degraded: list[str] = []
    version = db.schema_version(conn)
    if version is None:
        return {"status": "not_ready", "ready": False, "reasons": ["schema_missing"], "degraded": [], "checked_at": db.iso(now)}
    price_day = _latest_local_date(conn, "prices_dam", s.zone)
    load_day = _latest_local_date(conn, "load_actual", s.zone)
    if price_day is None:
        reasons.append("no_price_data")
    elif price_day < today:
        reasons.append(f"prices_stale: latest day {price_day}, need {today}")
    if load_day is None:
        degraded.append("no_load_data")
    elif now_et.time() >= time(1, 30) and load_day < today - timedelta(days=1):
        degraded.append(f"load_stale: latest day {load_day}, need {today - timedelta(days=1)}")
    active = conn.execute("SELECT model_id FROM models WHERE active=1 AND feature_set='OD'").fetchone()
    if active is None:
        degraded.append("no_active_model: baseline serves")
    tomorrow = str(today + timedelta(days=1))
    fc = conn.execute("SELECT id, served_kind, fallback_reason, generated_at FROM forecasts WHERE target_date=? "
                      "ORDER BY version DESC LIMIT 1", (tomorrow,)).fetchone()
    if fc is None and now_et.time() >= time(6, 0):
        degraded.append(f"no_forecast_for_{tomorrow}")
    status = "not_ready" if reasons else ("degraded" if degraded else "ready")
    return {"status": status, "ready": not reasons, "reasons": reasons, "degraded": degraded, "schema_version": version,
            "latest_price_date": str(price_day) if price_day else None, "latest_load_date": str(load_day) if load_day else None,
            "active_model": active[0] if active else None,
            "forecast_for_tomorrow": None if fc is None else {"target_date": tomorrow, "served_kind": fc["served_kind"],
                                                              "fallback_reason": fc["fallback_reason"],
                                                              "generated_at": fc["generated_at"]},
            "checked_at": db.iso(now)}


def recent_jobs(conn, limit: int = 8) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT job, started_at, finished_at, status, error FROM job_runs ORDER BY id DESC LIMIT ?", (limit,))]
