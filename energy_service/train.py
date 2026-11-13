"""Offline training job. Not called from any API request."""
from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta

import pandas as pd

from energy_service import dataset, registry
from energy_service.config import Settings
from energy_service.db import JobRun
from src import forecast_weather as fw
from src.features import make_features
from src.forecast_models import fit_xgb


def training_panels(conn, s: Settings, through: date) -> dict:
    """Panels as known at the origin of day `through` (D): prices through D, load through D-1, and the
    observed temperature as the weather column (variant (a): train on actuals, predict with the forecast)."""
    price = dataset.panel_from(dataset.prices_frame(conn, s.zone), "lbmp").loc[:pd.Timestamp(through)]
    load = dataset.panel_from(dataset.load_frame(conn, s.zone), "load_mw").loc[:pd.Timestamp(through) - timedelta(days=1)]
    wx = dataset.weather_frame(conn, "archive_actual")
    temp = fw.to_forecast_panel(wx, price.index) if len(wx) else pd.DataFrame(index=price.index, columns=range(24), dtype=float)
    return {"price": price, "load": load.reindex(price.index), "temp": temp, "temp_fc": temp}


def train_model(conn, s: Settings, *, feature_set: str, origin_date: date, now: datetime, params: dict | None = None,
                evidence: dict | None = None) -> str:
    with JobRun(conn, "train", now) as job:
        panels = training_panels(conn, s, origin_date)
        X, y = make_features(panels, feature_set)
        if len(X) < 24 * 60:
            raise dataset.DataIncomplete(f"only {len(X)} complete training rows for feature set {feature_set}")
        params = params or registry.chosen_params(s.results_dir, feature_set)
        model = fit_xgb(X, y, **params)
        data_hash = hashlib.sha256(pd.util.hash_pandas_object(X, index=True).values.tobytes() + y.to_numpy().tobytes()).hexdigest()
        evidence = evidence if evidence is not None else registry.evidence_from_results(s.results_dir, feature_set)
        dates = X.index.get_level_values("date")
        model_id = registry.save_model(
            conn, s, model, kind="xgb", feature_set=feature_set, columns=list(X.columns), params=params,
            train_start=str(dates.min().date()), train_end=str(dates.max().date()), n_rows=len(X), data_hash=data_hash,
            evidence=evidence, now=now)
        job.details = {"model_id": model_id, "n_rows": len(X), "eligible": bool(evidence.get("eligible"))}
    return model_id
