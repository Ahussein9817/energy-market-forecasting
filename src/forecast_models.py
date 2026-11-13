"""Forecasting models: seasonal-naive benchmarks, XGBoost, SARIMAX."""

import warnings

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

XGB_DEFAULT = dict(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8,
                   colsample_bytree=0.8, min_child_weight=5, tree_method="hist",
                   objective="reg:squarederror", random_state=42, n_jobs=4)


def seasonal_naive(price: pd.DataFrame, lag_days: int) -> pd.Series:
    """Prediction for target day T = price on day T - lag_days, same local hour.
    lag_days=1: 'same hour yesterday' (24 h); lag_days=7: 'same hour last week' (168 h)."""
    s = price.shift(lag_days).stack()
    s.index.names = ["date", "hour"]
    return s.dropna()


def fit_xgb(X: pd.DataFrame, y: pd.Series, **overrides) -> XGBRegressor:
    params = {**XGB_DEFAULT, **overrides}
    return XGBRegressor(**params).fit(X, y)


def sarimax_day_ahead(history: np.ndarray, exog_hist=None, exog_future=None,
                      order=(1, 0, 1), seasonal_order=(1, 0, 1, 24)) -> np.ndarray | None:
    """Fit on `history` (hourly, consecutive) and forecast the next 24 hours.
    Returns None when estimation fails (counted by the caller, never silently imputed)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    if not np.isfinite(np.asarray(history, dtype=float)).all():
        return None  # statsmodels would otherwise 'fit' NaN input and emit meaningless forecasts
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = SARIMAX(history, exog=exog_hist, order=order, seasonal_order=seasonal_order,
                          enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=50)
            fc = res.forecast(steps=24, exog=exog_future)
        out = np.asarray(fc, dtype=float)
        return out if np.isfinite(out).all() and len(out) == 24 else None
    except Exception:
        return None
