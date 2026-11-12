"""Metrics, Diebold-Mariano test and the walk-forward driver."""

import numpy as np
import pandas as pd
from scipy import stats

from src.features import make_features
from src.forecast_models import fit_xgb


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def bias(y, p):
    return float(np.mean(np.asarray(p) - np.asarray(y)))


def scaled_errors(y, p, y_ref_pred):
    """rMAE / rRMSE: error divided by that of a reference forecast (here seasonal naive 168 h).
    < 1 means better than the reference. MAPE is deliberately not provided (price series can be
    near zero or negative, which makes percentage errors unstable)."""
    d_mae, d_rmse = mae(y, y_ref_pred), rmse(y, y_ref_pred)
    return {"rMAE": mae(y, p) / d_mae if d_mae > 0 else float("nan"),
            "rRMSE": rmse(y, p) / d_rmse if d_rmse > 0 else float("nan")}


def diebold_mariano(loss1, loss2, lag=7):
    """DM test on the day-level loss differential d_t = loss1_t - loss2_t, one value per day
    (e.g. the daily mean absolute error of each model; see `daily_loss`). Negative statistic:
    model 1 has the lower loss. Bartlett-kernel HAC variance with bandwidth `lag`, normal
    approximation."""
    d = np.asarray(loss1, float) - np.asarray(loss2, float)
    n = len(d)
    dm = d - d.mean()
    var = np.dot(dm, dm) / n
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1)
        var += 2 * w * np.dot(dm[k:], dm[:-k]) / n
    if var <= 1e-18:
        return {"stat": float("nan"), "p_value": float("nan"), "mean_diff": float(d.mean()), "n": n}
    stat = d.mean() / np.sqrt(var / n)
    return {"stat": float(stat), "p_value": float(2 * (1 - stats.norm.cdf(abs(stat)))),
            "mean_diff": float(d.mean()), "n": n}


def daily_loss(frame: pd.DataFrame, col: str, loss: str = "abs") -> np.ndarray:
    """Per-day mean loss of column `col` against frame['y'] (frame indexed by date, hour)."""
    e = frame[col] - frame["y"]
    e = e.abs() if loss == "abs" else e ** 2
    return e.groupby(level="date").mean().sort_index().to_numpy()


def quarter_starts(start: str, end: str) -> list[pd.Timestamp]:
    return list(pd.date_range(start, end, freq="QS"))


def walk_forward_xgb(panels, feature_set, test_start, test_end, params=None, refit="QS"):
    """Expanding-window walk-forward. For each refit block, train on ALL target days strictly
    before the block's first target day, then predict every day of the block. Returns a
    Series indexed (date, hour). Model never sees any target of a day it predicts."""
    X, y = make_features(panels, feature_set)
    dates = X.index.get_level_values("date")
    test_days = pd.date_range(test_start, test_end, freq="D")
    starts = list(pd.date_range(test_start, test_end, freq=refit))
    if not starts or starts[0] != pd.Timestamp(test_start):
        starts = [pd.Timestamp(test_start)] + starts
    preds = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else test_days[-1] + pd.Timedelta(days=1)
        tr = dates < s
        te = (dates >= s) & (dates < e)
        if tr.sum() == 0 or te.sum() == 0:
            continue
        model = fit_xgb(X[tr], y[tr], **(params or {}))
        preds.append(pd.Series(model.predict(X[te]), index=X.index[te], name=f"xgb_{feature_set}"))
    return pd.concat(preds)


def summarize(preds: pd.DataFrame, models: list[str], ref: str, high_price_threshold: float | None = None) -> dict:
    """Metrics from a frame indexed (date, hour) with column `y` and one column per model.
    Only rows where every model has a prediction are used, so models are compared on identical points."""
    d = preds.dropna(subset=["y"] + models)
    out = {"n_points": int(len(d)), "n_days": int(d.index.get_level_values("date").nunique()),
           "overall": {}, "by_year": {}, "by_quarter": {}, "high_price_hours": {}, "dm_vs_ref": {}}
    dates = d.index.get_level_values("date")

    def block(sub):
        r = {}
        for m in models:
            r[m] = {"MAE": mae(sub["y"], sub[m]), "RMSE": rmse(sub["y"], sub[m]), "bias": bias(sub["y"], sub[m]),
                    **scaled_errors(sub["y"], sub[m], sub[ref])}
        return r

    out["overall"] = block(d)
    for yr in sorted(set(dates.year)):
        out["by_year"][str(yr)] = block(d[dates.year == yr])
    for q in sorted(set(zip(dates.year, dates.quarter))):
        out["by_quarter"][f"{q[0]}Q{q[1]}"] = block(d[(dates.year == q[0]) & (dates.quarter == q[1])])
    if high_price_threshold is not None:
        hi = d["y"] >= high_price_threshold
        out["high_price_hours"] = {"threshold": float(high_price_threshold), "n": int(hi.sum()),
                                   "metrics": block(d[hi]) if hi.any() else {}}
    ref_loss = daily_loss(d, ref)
    for m in models:
        if m != ref:
            out["dm_vs_ref"][m] = diebold_mariano(daily_loss(d, m), ref_loss)
    return out
