import numpy as np
import pandas as pd
import pytest

from src import evaluation as ev
from src import features as ft
from src import forecast_models as fm
from src import panel as pn


def _utc_df(start_local, days, fn):
    idx = pd.date_range(start_local, periods=days * 24, freq="h", tz="America/New_York",
                        ambiguous="infer", nonexistent="shift_forward").tz_convert("UTC")
    idx = idx.unique()
    return pd.DataFrame({"timestamp_utc": idx, "v": [fn(i, t) for i, t in enumerate(idx)]})


def test_panel_handles_both_dst_days():
    # 2024-03-10 spring forward (23 h), 2024-11-03 fall back (25 h): UTC-contiguous hourly
    utc = pd.date_range("2024-03-01 05:00", "2024-12-01 04:00", freq="h", tz="UTC")  # local midnight to local midnight
    df = pd.DataFrame({"timestamp_utc": utc, "v": np.arange(len(utc), dtype=float)})
    p = pn.to_panel(df, "v")
    assert p.shape[1] == 24 and not p.isna().any().any()
    assert p.index.is_monotonic_increasing and (p.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()


def test_panel_raises_on_real_gap():
    utc = list(pd.date_range("2024-05-01 04:00", periods=24 * 10, freq="h", tz="UTC"))
    del utc[100:110]  # 10-hour hole in the middle
    df = pd.DataFrame({"timestamp_utc": utc, "v": 1.0})
    with pytest.raises(ValueError, match="gaps remain"):
        pn.to_panel(df, "v")


def _panels(n=60, seed=0):
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    mk = lambda: pd.DataFrame(rng.random((n, 24)) * 50 + 20, index=idx, columns=range(24))  # noqa: E731
    return {"price": mk(), "load": mk(), "temp": mk()}


@pytest.mark.parametrize("fs", ["A", "B", "C"])
def test_features_never_use_target_day_or_future_prices_or_load(fs):
    """Leakage test: scramble every price/load value from day T onward; features for T must not change."""
    panels = _panels()
    X0, _ = ft.make_features(panels, fs)
    T = pd.Timestamp("2023-02-15")
    changed = {k: v.copy() for k, v in panels.items()}
    for k in ("price", "load"):
        changed[k].loc[changed[k].index >= T] += 1000.0
    X1, _ = ft.make_features(changed, fs)
    a = X0[X0.index.get_level_values("date") == T]
    b = X1[X1.index.get_level_values("date") == T]
    pd.testing.assert_frame_equal(a, b)
    # sanity: the target itself did change, so the test could detect a leak
    assert (ft.make_features(changed, fs)[1].loc[b.index] != ft.make_features(panels, fs)[1].loc[a.index]).all()


def test_only_set_C_uses_target_day_temperature():
    panels = _panels()
    for fs, expected in (("A", False), ("B", False), ("C", True)):
        assert ("temp_t" in ft.make_features(panels, fs)[0].columns) is expected


def test_lag_alignment_values():
    panels = _panels()
    X, y = ft.make_features(panels, "A")
    T, h = pd.Timestamp("2023-02-20"), 5
    assert X.loc[(T, h), "p_d1"] == panels["price"].loc[T - pd.Timedelta(days=1), h]
    assert X.loc[(T, h), "p_d7"] == panels["price"].loc[T - pd.Timedelta(days=7), h]
    assert X.loc[(T, h), "p_wk_mean"] == pytest.approx(
        panels["price"].loc[T - pd.Timedelta(days=7):T - pd.Timedelta(days=1), h].mean())
    assert y.loc[(T, h)] == panels["price"].loc[T, h]


def test_features_reject_gappy_panel_and_bad_set():
    panels = _panels()
    gappy = {k: v.drop(v.index[10]) for k, v in panels.items()}
    with pytest.raises(ValueError, match="contiguous"):
        ft.make_features(gappy, "A")
    with pytest.raises(ValueError):
        ft.make_features(panels, "Z")


def test_seasonal_naive_lags():
    panels = _panels()
    s1, s7 = fm.seasonal_naive(panels["price"], 1), fm.seasonal_naive(panels["price"], 7)
    T = pd.Timestamp("2023-02-10")
    assert s1.loc[(T, 3)] == panels["price"].loc[T - pd.Timedelta(days=1), 3]
    assert s7.loc[(T, 3)] == panels["price"].loc[T - pd.Timedelta(days=7), 3]
    assert (pd.Timestamp("2023-01-01"), 0) not in s1.index  # no prediction without history


def test_metrics_known_values():
    y, p = np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 5.0])
    assert ev.mae(y, p) == pytest.approx(1.0)
    assert ev.rmse(y, p) == pytest.approx(np.sqrt(5 / 3))
    assert ev.bias(y, p) == pytest.approx(1.0)
    assert ev.scaled_errors(y, p, np.array([1.0, 2.0, 3.0]) + 2)["rMAE"] == pytest.approx(0.5)


def test_diebold_mariano_detects_better_model_and_null_case():
    rng = np.random.default_rng(0)
    n = 400
    good = np.abs(rng.normal(0, 1, n))
    bad = np.abs(rng.normal(0, 2, n))
    r = ev.diebold_mariano(good, bad)
    assert r["stat"] < -3 and r["p_value"] < 0.01 and r["mean_diff"] < 0
    same = ev.diebold_mariano(good, good.copy())
    assert np.isnan(same["stat"])
    a, b = np.abs(rng.normal(0, 1, n)), np.abs(rng.normal(0, 1, n))
    assert ev.diebold_mariano(a, b)["p_value"] > 0.01


def test_daily_loss_and_summarize_known_values():
    idx = pd.MultiIndex.from_product([pd.date_range("2023-01-01", periods=3), range(2)], names=["date", "hour"])
    df = pd.DataFrame({"y": [10.0] * 6, "ref": [12.0] * 6, "m": [11.0, 9.0, 10.0, 10.0, 13.0, 7.0]}, index=idx)
    assert ev.daily_loss(df, "m").tolist() == [1.0, 0.0, 3.0]
    s = ev.summarize(df, ["ref", "m"], ref="ref", high_price_threshold=10.0)
    assert s["overall"]["m"]["MAE"] == pytest.approx(8 / 6)
    assert s["overall"]["ref"]["rMAE"] == 1.0
    assert s["overall"]["m"]["rMAE"] == pytest.approx((8 / 6) / 2)
    assert s["by_year"]["2023"]["m"]["MAE"] == pytest.approx(8 / 6)
    assert s["by_quarter"]["2023Q1"]["ref"]["MAE"] == 2.0
    assert s["high_price_hours"]["n"] == 6
    assert "m" in s["dm_vs_ref"] and "ref" not in s["dm_vs_ref"]


def test_summarize_uses_only_rows_where_all_models_predict():
    idx = pd.MultiIndex.from_product([pd.date_range("2023-01-01", periods=2), range(1)], names=["date", "hour"])
    df = pd.DataFrame({"y": [1.0, 1.0], "a": [1.0, 1.0], "b": [np.nan, 2.0]}, index=idx)
    s = ev.summarize(df, ["a", "b"], ref="a")
    assert s["n_points"] == 1


def test_walk_forward_never_trains_on_prediction_days():
    panels = _panels(n=200)
    captured = []
    real_fit = ev.fit_xgb

    def spy(Xtr, ytr, **kw):
        captured.append(Xtr.index.get_level_values("date").max())
        return real_fit(Xtr, ytr, n_estimators=5, **kw)

    ev.fit_xgb = spy
    try:
        preds = ev.walk_forward_xgb(panels, "A", "2023-04-01", "2023-06-30", refit="MS")
    finally:
        ev.fit_xgb = real_fit
    assert len(captured) == 3  # April, May, June blocks
    starts = [pd.Timestamp("2023-04-01"), pd.Timestamp("2023-05-01"), pd.Timestamp("2023-06-01")]
    for last_train, s in zip(captured, starts):
        assert last_train < s
    dates = preds.index.get_level_values("date")
    assert dates.min() == pd.Timestamp("2023-04-01") and dates.max() == pd.Timestamp("2023-06-30")
    assert len(preds) == 91 * 24 and not preds.index.duplicated().any()


def test_sarimax_returns_24_values_or_none():
    rng = np.random.default_rng(0)
    h = 30 + 10 * np.sin(np.arange(24 * 20) * 2 * np.pi / 24) + rng.normal(0, 1, 24 * 20)
    out = fm.sarimax_day_ahead(h, order=(1, 0, 0), seasonal_order=(1, 0, 0, 24))
    assert out is not None and out.shape == (24,)
    assert fm.sarimax_day_ahead(np.array([np.nan] * 100)) is None


def test_panel_trims_incomplete_edge_days_but_keeps_interior():
    utc = pd.date_range("2024-05-01 00:00", periods=24 * 5, freq="h", tz="UTC")  # starts 8pm local
    df = pd.DataFrame({"timestamp_utc": utc, "v": 1.0})
    p = pn.to_panel(df, "v")
    # 00:00Z May 1 = 8pm local Apr 30 (partial, trimmed); series ends 7pm local May 5 (partial, trimmed)
    assert p.index.min() == pd.Timestamp("2024-05-01") and p.index.max() == pd.Timestamp("2024-05-04")
    assert len(p) == 4


def test_scaled_errors_perfect_reference_gives_nan_not_crash():
    y = np.array([1.0, 2.0])
    r = ev.scaled_errors(y, y + 1, y.copy())
    assert np.isnan(r["rMAE"]) and np.isnan(r["rRMSE"])
