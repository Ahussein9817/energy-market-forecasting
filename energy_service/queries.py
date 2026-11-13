"""Read-only queries shared by the API and the dashboard (one definition of "latest forecast", "evaluated day", ...)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from energy_service.db import iso

LATEST_VERSION = "f.version = (SELECT MAX(version) FROM forecasts WHERE target_date=f.target_date)"


def latest_forecast_row(conn):
    return conn.execute("SELECT * FROM forecasts ORDER BY target_date DESC, version DESC LIMIT 1").fetchone()


def forecast_for(conn, target_date: str):
    return conn.execute("SELECT * FROM forecasts WHERE target_date=? ORDER BY version DESC LIMIT 1", (target_date,)).fetchone()


def forecast_hours(conn, forecast_id: int) -> pd.DataFrame:
    rows = conn.execute("SELECT hour, baseline, model, served FROM forecast_hours WHERE forecast_id=? ORDER BY hour",
                        (forecast_id,)).fetchall()
    return pd.DataFrame([dict(r) for r in rows], columns=["hour", "baseline", "model", "served"])


def evaluation_frame(conn, days: int = 90, include_late: bool = False, today: datetime | None = None) -> pd.DataFrame:
    """One row per evaluated target day (latest version). `late` forecasts (made after the prices were public) are
    excluded by default because they carry no operational information."""
    cutoff = str((pd.Timestamp(today) if today is not None else pd.Timestamp.now()).date() - timedelta(days=days))
    rows = conn.execute(
        f"SELECT f.id, f.target_date, f.version, f.served_kind, f.fallback_reason, f.late, f.source, f.served_model_id, "
        f"e.mae_served, e.mae_baseline, e.mae_model, e.bias_served FROM forecasts f JOIN evaluations e ON e.forecast_id=f.id "
        f"WHERE {LATEST_VERSION} AND f.target_date>=? ORDER BY f.target_date", (cutoff,)).fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if df.empty:
        return pd.DataFrame(columns=["id", "target_date", "version", "served_kind", "fallback_reason", "late", "source",
                                     "served_model_id", "mae_served", "mae_baseline", "mae_model", "bias_served"])
    df["target_date"] = pd.to_datetime(df["target_date"])
    if not include_late:
        df = df[df["late"] == 0]
    return df[df["source"] != "replay"].reset_index(drop=True)


def hourly_errors(conn, days: int = 90, include_late: bool = False, today: datetime | None = None) -> pd.DataFrame:
    """Hour-level actual, baseline, model and served values for evaluated forecast days."""
    ev = evaluation_frame(conn, days, include_late, today)
    if ev.empty:
        return pd.DataFrame(columns=["target_date", "hour", "actual", "baseline", "model", "served", "served_kind"])
    ids = ",".join(str(int(i)) for i in ev["id"])
    rows = conn.execute(
        f"SELECT h.forecast_id, h.hour, h.baseline, h.model, h.served, a.actual FROM forecast_hours h "
        f"JOIN evaluation_hours a ON a.forecast_id=h.forecast_id AND a.hour=h.hour WHERE h.forecast_id IN ({ids})").fetchall()
    df = pd.DataFrame([dict(r) for r in rows]).merge(ev[["id", "target_date", "served_kind"]], left_on="forecast_id", right_on="id")
    return df.drop(columns=["id", "forecast_id"]).sort_values(["target_date", "hour"]).reset_index(drop=True)


def performance_table(h: pd.DataFrame) -> pd.DataFrame:
    """MAE / RMSE / bias for served, baseline and (where present) model forecasts over the given hourly frame."""
    out = []
    for name, col in (("served", "served"), ("baseline", "baseline"), ("model (when produced)", "model")):
        d = h.dropna(subset=[col])
        if d.empty:
            continue
        e = d[col] - d["actual"]
        out.append({"forecast": name, "hours": len(d), "MAE": float(e.abs().mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
                    "bias": float(e.mean())})
    return pd.DataFrame(out)


def spike_analysis(h: pd.DataFrame, threshold: float) -> dict:
    """Errors on hours whose actual price was at or above `threshold` (e.g. the 95th percentile of the last year)."""
    s = h[h["actual"] >= threshold]
    if s.empty:
        return {"threshold": threshold, "hours": 0}
    return {"threshold": threshold, "hours": int(len(s)), "served_mae": float((s["served"] - s["actual"]).abs().mean()),
            "baseline_mae": float((s["baseline"] - s["actual"]).abs().mean()), "served_bias": float((s["served"] - s["actual"]).mean())}


def price_history(conn, zone: str, days: int = 14) -> pd.DataFrame:
    rows = conn.execute("SELECT ts_utc, lbmp FROM prices_dam WHERE zone=? ORDER BY ts_utc DESC LIMIT ?",
                        (zone, days * 25)).fetchall()
    df = pd.DataFrame({"ts": pd.to_datetime([r[0] for r in rows], utc=True).tz_convert("America/New_York"),
                       "lbmp": [r[1] for r in rows]}).sort_values("ts")
    if df.empty:
        return df
    return df[df["ts"] >= df["ts"].max() - pd.Timedelta(days=days)].reset_index(drop=True)


def price_quantile(conn, zone: str, q: float, days: int = 365) -> float | None:
    rows = conn.execute("SELECT lbmp FROM prices_dam WHERE zone=? ORDER BY ts_utc DESC LIMIT ?", (zone, days * 24)).fetchall()
    return float(np.quantile([r[0] for r in rows], q)) if rows else None


def recent_flags(conn, now: datetime, hours: int = 48) -> pd.DataFrame:
    rows = conn.execute("SELECT created_at, dataset, delivery_date, code, severity, detail FROM quality_flags "
                        "WHERE created_at>=? ORDER BY id DESC", (iso(now - timedelta(hours=hours)),)).fetchall()
    return pd.DataFrame([dict(r) for r in rows], columns=["created_at", "dataset", "delivery_date", "code", "severity", "detail"])


def recent_events(conn, limit: int = 20) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT created_at, level, code, message FROM events ORDER BY id DESC LIMIT ?", (limit,))]


def recent_jobs(conn, limit: int = 15) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT job, started_at, finished_at, status, error FROM job_runs ORDER BY id DESC LIMIT ?", (limit,))]


def day_detail(conn, target_date: str, zone: str) -> dict | None:
    f = forecast_for(conn, target_date)
    if f is None:
        return None
    hours = forecast_hours(conn, f["id"])
    actual = pd.DataFrame([dict(r) for r in conn.execute(
        "SELECT hour, actual FROM evaluation_hours WHERE forecast_id=? ORDER BY hour", (f["id"],))], columns=["hour", "actual"])
    flags = conn.execute("SELECT created_at, dataset, code, severity, detail FROM quality_flags WHERE delivery_date IN (?, ?)",
                         (f["origin_date"], target_date)).fetchall()
    ev = conn.execute("SELECT * FROM evaluations WHERE forecast_id=?", (f["id"],)).fetchone()
    return {"forecast": dict(f), "inputs": json.loads(f["input_json"]), "hours": hours.merge(actual, on="hour", how="left"),
            "flags": [dict(x) for x in flags], "evaluation": dict(ev) if ev else None}
