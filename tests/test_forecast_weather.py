"""Offline tests for the forecast-weather experiment (no network, no real models)."""
import numpy as np
import pandas as pd
import pytest

from src import evaluation as ev
from src import features as ft
from src import forecast_weather as fw
from src.data_loader import DataError


class FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self._p


class FakeSession:
    def __init__(self, payload):
        self.payload, self.calls = payload, []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return FakeResp(self.payload)


def test_fetch_keeps_nulls_as_nan_and_requests_the_right_variable():
    payload = {"hourly": {"time": ["2023-01-01T00:00", "2023-01-01T01:00", "2023-01-01T02:00"],
                          "temperature_2m_previous_day2": [1.5, None, 3.0]}}
    s = FakeSession(payload)
    df = fw.fetch_previous_runs("2023-01-01", "2023-01-01", lead_days=2, session=s, retries=1)
    assert s.calls[0][1]["hourly"] == "temperature_2m_previous_day2"
    assert s.calls[0][1]["models"] == "gfs_global"  # never the HRRR-blending gfs_seamless
    assert df["temp_fc_c"].isna().tolist() == [False, True, False]  # null stays missing, not filled
    assert str(df["timestamp_utc"].dt.tz) == "UTC"


def test_fetch_raises_when_variable_missing():
    with pytest.raises(DataError):
        fw.fetch_previous_runs("2023-01-01", "2023-01-01", session=FakeSession({"hourly": {"time": []}}),
                               retries=1, backoff=0)


def _hourly(start_utc, end_utc, value=10.0):
    ts = pd.date_range(start_utc, end_utc, freq="h", tz="UTC")
    return pd.DataFrame({"timestamp_utc": ts, "temp_fc_c": value})


def test_panel_never_fills_an_isolated_forecast_hole():
    df = _hourly("2023-06-01 04:00", "2023-06-04 03:00")  # local days 06-01..06-03, no DST
    df.loc[30, "temp_fc_c"] = np.nan  # one missing hour, mid-day 06-02
    idx = pd.date_range("2023-06-01", "2023-06-03", freq="D")
    p = fw.to_forecast_panel(df, idx)
    assert p.isna().sum().sum() == 1
    assert fw.missing_days(p).tolist() == [pd.Timestamp("2023-06-02")]


def test_panel_fills_only_the_nonexistent_spring_forward_hour():
    # 2024-03-10: local hour 2 does not exist; the other 23 hours are present
    df = _hourly("2024-03-10 05:00", "2024-03-11 03:59")
    df["temp_fc_c"] = np.arange(len(df), dtype=float)
    idx = pd.DatetimeIndex(["2024-03-10"])
    p = fw.to_forecast_panel(df, idx)
    assert not p.isna().any().any()
    assert p.loc["2024-03-10", 2] == pytest.approx((p.loc["2024-03-10", 1] + p.loc["2024-03-10", 3]) / 2)


def _panels(n=60, seed=0):
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    mk = lambda: pd.DataFrame(rng.random((n, 24)) * 50 + 20, index=idx, columns=range(24))  # noqa: E731
    p = {"price": mk(), "load": mk(), "temp": mk()}
    p["temp_fc"] = p["temp"] + 3.0
    return p


def test_set_D_requires_forecast_panel():
    p = _panels()
    del p["temp_fc"]
    with pytest.raises(ValueError, match="temp_fc"):
        ft.make_features(p, "D")


def test_set_D_uses_forecast_not_target_day_actual_temperature():
    p = _panels()
    X0, _ = ft.make_features(p, "D")
    T = pd.Timestamp("2023-02-15")
    changed = {k: v.copy() for k, v in p.items()}
    changed["temp"].loc[changed["temp"].index >= T] += 500.0   # actuals from day T onward
    changed["price"].loc[changed["price"].index >= T] += 1000.0
    X1, _ = ft.make_features(changed, "D")
    a, b = (X[X.index.get_level_values("date") == T] for X in (X0, X1))
    pd.testing.assert_frame_equal(a, b)  # target-day actual temperature and later prices cannot leak
    assert "temp_t" not in X0.columns and "temp_fc_t" in X0.columns
    # the forecast column carries the forecast, and temp_d1 is the observed temperature of day D
    row = X0.loc[(T, 7)]
    assert row["temp_fc_t"] == p["temp_fc"].loc[T, 7]
    assert row["temp_d1"] == p["temp"].loc[T - pd.Timedelta(days=1), 7]


def test_missing_forecast_rows_are_dropped_not_filled():
    p = _panels()
    p["temp_fc"].loc["2023-02-10", 5] = np.nan
    X, _ = ft.make_features(p, "D")
    dates = X.index.get_level_values("date")
    assert not X.isna().any().any()
    assert (pd.Timestamp("2023-02-10"), 5) not in X.index
    assert (dates == pd.Timestamp("2023-02-10")).sum() == 23


def test_walk_forward_train_panels_and_train_dates(monkeypatch):
    seen = []

    class Dummy:
        def predict(self, X):
            return np.zeros(len(X))

    def fake_fit(X, y, **kw):
        seen.append(X.copy())
        return Dummy()

    monkeypatch.setattr(ev, "fit_xgb", fake_fit)
    p = _panels(n=120)
    train_panels = {**p, "temp_fc": p["temp"]}          # variant (a): actuals in the forecast column
    allowed = pd.date_range("2023-01-20", "2023-04-30", freq="D")
    out = ev.walk_forward_xgb(p, "D", "2023-03-01", "2023-04-15", train_panels=train_panels,
                              train_dates=allowed, refit="MS")
    assert len(out) == 46 * 24
    assert seen, "model was never fitted"
    for X in seen:
        d = X.index.get_level_values("date")
        assert d.isin(allowed).all()                      # only the permitted window
        # training rows carry ACTUAL temperature in temp_fc_t, not the forecast (+3)
        r = X.index[0]
        assert X.loc[r, "temp_fc_t"] == p["temp"].loc[r[0], r[1]]
    first = seen[0].index.get_level_values("date")
    assert first.max() < pd.Timestamp("2023-03-01")       # strictly before the first refit block


def test_default_walk_forward_is_unchanged_without_new_arguments(monkeypatch):
    seen = []

    class Dummy:
        def predict(self, X):
            return np.zeros(len(X))

    monkeypatch.setattr(ev, "fit_xgb", lambda X, y, **kw: seen.append(X) or Dummy())
    p = _panels(n=100)
    ev.walk_forward_xgb(p, "C", "2023-03-01", "2023-04-05", refit="MS")
    X_all, _ = ft.make_features(p, "C")
    d0 = seen[0].index.get_level_values("date")
    assert len(seen[0]) == int((X_all.index.get_level_values("date") < pd.Timestamp("2023-03-01")).sum())
    assert d0.max() < pd.Timestamp("2023-03-01")
