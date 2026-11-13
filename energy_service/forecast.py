"""Forecast job: build the day-ahead forecast for D+1 at the origin of day D from information available then.

The baseline (price of day D, hour by hour) is always computed. The model forecast is served only if the model
is eligible (backtest evidence), the features are complete and a fresh weather forecast exists; otherwise the
baseline is served and the reason is recorded. Forecasts generated after the day-ahead prices are public are
flagged `late`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from energy_service import dataset, registry
from energy_service.config import Settings
from energy_service.db import JobRun, iso, log_event
from src import forecast_weather as fw
from src.features import make_features

ET = ZoneInfo("America/New_York")
HISTORY_DAYS = 60       # recent window used to build features: bounds the gap check and the work


class ForecastUnavailable(Exception):
    """Even the baseline cannot be produced (price of day D missing)."""


@dataclass
class ForecastResult:
    forecast_id: int
    target_date: str
    version: int
    served_kind: str
    fallback_reason: str | None
    late: bool
    existing: bool


def _recent(df: pd.DataFrame, origin: date, days: int = HISTORY_DAYS) -> pd.DataFrame:
    cutoff = pd.Timestamp(origin, tz="UTC") - pd.Timedelta(days=days)
    return df[df["timestamp_utc"] >= cutoff].reset_index(drop=True)


def _model_candidate(conn, feature_set: str):
    """Returns (row, reason). reason is a fallback code when the model cannot be used."""
    row = conn.execute("SELECT * FROM models WHERE active=1 AND feature_set=? ORDER BY trained_at DESC LIMIT 1",
                       (feature_set,)).fetchone()
    if row is None:
        return None, "no_active_model"
    if not row["eligible"]:
        return None, "model_not_eligible"
    if row["feature_version"] != registry.FEATURE_VERSION:
        return None, "feature_version_mismatch"
    return row, None


def _model_forecast(conn, s: Settings, row, price, origin: date, target: pd.Timestamp, now: datetime):
    """Returns (values|None, reason|None, info). Never raises for missing inputs; those become fallback reasons."""
    info: dict = {}
    try:
        model, meta = registry.load_model(row)
    except Exception as e:  # any load failure (missing file, bad hash, unpickling) => baseline
        return None, "model_load_failed", {"error": f"{type(e).__name__}: {e}"}
    try:
        load = dataset.panel_from(_recent(dataset.load_frame(conn, s.zone), origin), "load_mw")
    except dataset.DataIncomplete as e:
        return None, "load_missing", {"error": str(e)}
    if load.index.max() < pd.Timestamp(origin) - timedelta(days=1):
        return None, "load_missing", {"error": f"latest load day {load.index.max().date()}, need {origin - timedelta(days=1)}"}
    snap = dataset.latest_live_snapshot(conn, now, s.weather_max_age_hours * 24)  # age filtered below for a precise reason
    if snap is None:
        return None, "weather_missing", {}
    srow, wdf, age = snap
    info.update(weather_snapshot_id=srow["id"], weather_retrieved_at=srow["retrieved_at"], weather_age_hours=round(age, 2))
    if age > s.weather_max_age_hours:
        return None, "weather_stale", info
    ext = price.index.append(pd.DatetimeIndex([target]))
    temp_fc = fw.to_forecast_panel(wdf, ext)
    if temp_fc.loc[target].isna().any():
        return None, "weather_missing", {**info, "missing_hours": int(temp_fc.loc[target].isna().sum())}
    panels = {"price": price.reindex(ext), "load": load.reindex(ext), "temp": temp_fc, "temp_fc": temp_fc}
    X, _ = make_features(panels, meta["feature_set"])
    X = X[X.index.get_level_values("date") == target]
    if len(X) != 24:
        return None, "features_incomplete", {**info, "complete_hours": len(X)}
    if set(meta["columns"]) != set(X.columns):
        return None, "feature_version_mismatch", info
    X = X[meta["columns"]]
    info["feature_hash"] = hashlib.sha256(pd.util.hash_pandas_object(X, index=True).values.tobytes()).hexdigest()
    return [float(v) for v in model.predict(X)], None, info


def generate_forecast(conn, s: Settings, *, origin_date: date, now: datetime, force: bool = False,
                      source: str = "scheduled") -> ForecastResult:
    target_date = origin_date + timedelta(days=1)
    existing = conn.execute("SELECT * FROM forecasts WHERE target_date=? ORDER BY version DESC LIMIT 1",
                            (str(target_date),)).fetchone()
    if existing is not None and not force:
        return ForecastResult(existing["id"], str(target_date), existing["version"], existing["served_kind"],
                              existing["fallback_reason"], bool(existing["late"]), True)
    version = existing["version"] + 1 if existing is not None else 1
    target = pd.Timestamp(target_date)
    with JobRun(conn, "forecast", now) as job:
        try:
            price = dataset.panel_from(_recent(dataset.prices_frame(conn, s.zone), origin_date), "lbmp")
            price = price.loc[:pd.Timestamp(origin_date)]
            if len(price) == 0 or price.index.max() != pd.Timestamp(origin_date):
                raise dataset.DataIncomplete(f"day-ahead prices for {origin_date} are missing or incomplete")
        except dataset.DataIncomplete as e:
            log_event(conn, "error", "forecast_failed", str(e), {"target_date": str(target_date)}, now)
            raise ForecastUnavailable(str(e)) from e
        baseline = [float(v) for v in price.loc[pd.Timestamp(origin_date)].to_numpy()]
        model_vals, reason, info = None, None, {}
        row, reason = _model_candidate(conn, "OD")
        if row is not None:
            model_vals, reason, info = _model_forecast(conn, s, row, price, origin_date, target, now)
        served_kind = "model" if model_vals is not None else "baseline"
        served = model_vals if model_vals is not None else baseline
        h, m = map(int, s.late_after_et.split(":"))
        late = now.astimezone(ET) > datetime.combine(origin_date, time(h, m), tzinfo=ET)
        inputs = {"price_last_day": str(origin_date), "baseline": "price of day D, same local hour",
                  "feature_version": registry.FEATURE_VERSION, "model_id": row["model_id"] if row is not None else None,
                  "origin_nominal": "05:00 America/New_York", **info}
        cur = conn.execute(
            "INSERT INTO forecasts (origin_date, target_date, version, origin_ts, generated_at, source, served_kind, "
            "served_model_id, fallback_reason, late, input_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (str(origin_date), str(target_date), version, iso(datetime.combine(origin_date, time(5, 0), tzinfo=ET)),
             iso(now), source, served_kind, row["model_id"] if (row is not None and model_vals is not None) else None,
             reason, int(late), json.dumps(inputs)))
        fid = cur.lastrowid
        conn.executemany("INSERT INTO forecast_hours (forecast_id, hour, baseline, model, served) VALUES (?,?,?,?,?)",
                         [(fid, hh, baseline[hh], None if model_vals is None else model_vals[hh], served[hh]) for hh in range(24)])
        conn.commit()
        if reason:
            log_event(conn, "info", "forecast_fallback", f"baseline served for {target_date}: {reason}",
                      {"forecast_id": fid, "reason": reason}, now)
        job.details = {"forecast_id": fid, "target_date": str(target_date), "served": served_kind, "reason": reason, "late": late}
    return ForecastResult(fid, str(target_date), version, served_kind, reason, late, False)
