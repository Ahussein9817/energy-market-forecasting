"""Integrity checks on the committed forecast-weather results (no network, no model fits)."""
import json
from pathlib import Path

import pandas as pd
import pytest

RES = Path(__file__).resolve().parent.parent / "results"
fw_path = RES / "forecast_weather_predictions.parquet"
pytestmark = pytest.mark.skipif(not fw_path.exists(), reason="results not generated")

D_COLS = ["xgb_D1_train_actual", "xgb_D1_train_fc", "xgb_D2_train_actual", "xgb_D2_train_fc"]


def test_forecast_models_are_missing_only_on_the_excluded_days():
    df = pd.read_parquet(fw_path)
    meta = json.loads((RES / "forecast_weather_run_metadata.json").read_text())
    excluded = set(pd.DatetimeIndex(meta["excluded_days_missing_forecast"]))
    dates = df.index.get_level_values("date")
    union = set()
    for c in D_COLS:
        missing = set(pd.DatetimeIndex(dates[df[c].isna()].unique()))
        assert missing <= excluded, c                    # nothing else is dropped, nothing is filled
        assert df.loc[~dates.isin(list(excluded)), c].notna().all()
        union |= missing
    # lead-1 and lead-2 gaps differ by a day; the exclusion set is their union, applied to all models
    assert union == excluded
    assert len(excluded) == meta["n_excluded_days"]


def test_shared_columns_equal_the_committed_original_backtest():
    df = pd.read_parquet(fw_path)
    old = pd.read_parquet(RES / "predictions_test.parquet")
    for c in ["y", "naive_24h", "naive_168h", "xgb_B", "xgb_C_oracle_temp"]:
        pd.testing.assert_series_equal(df[c], old.loc[df.index, c], check_names=False)


def test_no_weather_control_has_predictions_on_every_test_day():
    df = pd.read_parquet(fw_path)
    assert df["xgb_B_matched_window"].notna().all()      # the control needs no forecast


def test_metadata_records_frozen_params_and_forecast_quality():
    meta = json.loads((RES / "forecast_weather_run_metadata.json").read_text())
    orig = json.loads((RES / "run_metadata.json").read_text())["chosen_params"]
    assert meta["xgb_params_frozen_from_run_backtest"] == orig      # not retuned
    assert meta["weather_model"] == "gfs_global"
    for k in ("previous_day1", "previous_day2"):
        assert meta["weather"][k]["forecast_minus_actual_temp_C"]["MAE"] < 5
