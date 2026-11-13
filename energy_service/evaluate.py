"""Evaluation job: score stored forecasts against the day-ahead prices once those are published."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from energy_service.config import Settings
from energy_service.db import JobRun, iso
from src import panel as pn


def actual_day(conn, s: Settings, target: pd.Timestamp) -> np.ndarray | None:
    """24 local-hour day-ahead prices for `target`, or None if that day is not (completely) available."""
    lo = pd.Timestamp(target - timedelta(days=1)).tz_localize("America/New_York").tz_convert("UTC")
    hi = pd.Timestamp(target + timedelta(days=2)).tz_localize("America/New_York").tz_convert("UTC")
    rows = conn.execute("SELECT ts_utc, lbmp FROM prices_dam WHERE zone=? AND ts_utc>=? AND ts_utc<? ORDER BY ts_utc",
                        (s.zone, iso(lo), iso(hi))).fetchall()
    if not rows:
        return None
    df = pd.DataFrame({"timestamp_utc": pd.to_datetime([r[0] for r in rows], utc=True), "lbmp": [r[1] for r in rows]})
    try:
        p = pn.to_panel(df, "lbmp")
    except ValueError:
        return None
    return p.loc[target].to_numpy() if target in p.index else None


def evaluate_pending(conn, s: Settings, *, now: datetime) -> dict:
    rows = conn.execute("SELECT f.id, f.target_date FROM forecasts f LEFT JOIN evaluations e ON e.forecast_id=f.id "
                        "WHERE e.forecast_id IS NULL ORDER BY f.target_date").fetchall()
    done = waiting = 0
    with JobRun(conn, "evaluate", now) as job:
        for r in rows:
            actual = actual_day(conn, s, pd.Timestamp(r["target_date"]))
            if actual is None:
                waiting += 1
                continue
            h = conn.execute("SELECT hour, baseline, model, served FROM forecast_hours WHERE forecast_id=? ORDER BY hour",
                             (r["id"],)).fetchall()
            base, served = np.array([x["baseline"] for x in h]), np.array([x["served"] for x in h])
            model = None if any(x["model"] is None for x in h) else np.array([x["model"] for x in h])
            conn.execute("INSERT INTO evaluations (forecast_id, evaluated_at, target_date, mae_served, mae_baseline, mae_model, "
                         "bias_served) VALUES (?,?,?,?,?,?,?)",
                         (r["id"], iso(now), r["target_date"], float(np.abs(served - actual).mean()),
                          float(np.abs(base - actual).mean()),
                          None if model is None else float(np.abs(model - actual).mean()), float((served - actual).mean())))
            conn.executemany("INSERT INTO evaluation_hours (forecast_id, hour, actual) VALUES (?,?,?)",
                             [(r["id"], i, float(v)) for i, v in enumerate(actual)])
            done += 1
        conn.commit()
        job.details = {"evaluated": done, "waiting_for_actuals": waiting}
    return {"evaluated": done, "waiting_for_actuals": waiting}
