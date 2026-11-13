"""The serving-eligibility rule and the committed operational backtest results."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import evaluate_operational as eo

RES = Path(__file__).resolve().parent.parent / "results"


def synthetic(better=True, seed=0, days=400):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product([pd.date_range("2023-01-01", periods=days), range(24)], names=["date", "hour"])
    y = 40 + rng.normal(0, 8, len(idx))
    df = pd.DataFrame({"y": y}, index=idx)
    df["naive_24h"] = y + rng.normal(0, 6, len(idx))
    sd = 3 if better else 6
    for m in eo.MODELS[1:]:
        df[m] = y + rng.normal(0, sd, len(idx))
    return df


def test_rule_marks_clearly_better_models_eligible_and_names_a_candidate():
    out = eo.compute(synthetic(better=True), hi=80.0)
    assert out["decision"]["serving_candidate"] is not None
    assert set(out["decision"]["eligible_models"]) == set(eo.MODELS[1:])
    v = out["vs_baseline"]["xgb_OD_train_actual"]
    assert v["mae_diff_ci"]["hi"] < 0 and v["dm"]["p_value"] < 0.05 and all(x < 0 for x in v["mean_diff_by_year"].values())


def test_rule_rejects_models_no_better_than_the_baseline():
    out = eo.compute(synthetic(better=False), hi=80.0)
    assert out["decision"]["eligible_models"] == [] and out["decision"]["serving_candidate"] is None
    assert "baseline" in out["decision"]["if_none_eligible"]


def test_rule_requires_improvement_in_every_calendar_year():
    df = synthetic(better=True, days=730)
    dates = df.index.get_level_values("date")
    worse = dates.year == 2024                       # model is better in 2023 but much worse in 2024
    df.loc[worse, "xgb_OD_train_actual"] = df.loc[worse, "y"] + np.random.default_rng(1).normal(0, 12, int(worse.sum()))
    v = eo.compute(df, hi=80.0)["vs_baseline"]["xgb_OD_train_actual"]
    assert v["mean_diff_by_year"]["2023"] < 0 < v["mean_diff_by_year"]["2024"] and v["eligible_to_serve"] is False


@pytest.mark.skipif(not (RES / "operational_predictions.parquet").exists(), reason="results not generated")
def test_committed_operational_results_are_consistent():
    df = pd.read_parquet(RES / "operational_predictions.parquet")
    meta = json.loads((RES / "operational_run_metadata.json").read_text())
    decision = json.loads((RES / "operational_decision.json").read_text())
    metrics = json.loads((RES / "operational_metrics.json").read_text())
    dates = df.index.get_level_values("date")
    excl = set(pd.DatetimeIndex(meta["excluded_days_missing_forecast"]))
    for c in ("xgb_OD_train_actual", "xgb_OD_train_fc"):
        assert set(pd.DatetimeIndex(dates[df[c].isna()].unique())) == excl          # missing only on excluded days, never filled
    assert df[["y", "naive_24h", "xgb_OB", "xgb_OB_matched_window"]].notna().all().all()
    assert meta["origin"].startswith("05:00") and meta["weather_lead_days"] == 2    # never the unsafe 1-day lead
    # the stored decision is exactly what the rule yields from the stored metrics
    expected = sorted(m for m, v in metrics["vs_baseline"].items() if v["eligible_to_serve"])
    assert decision["eligible_models"] == expected
    for m, v in metrics["vs_baseline"].items():
        rule = v["mae_diff_ci"]["hi"] < 0 and v["dm"]["p_value"] < 0.05 and all(x < 0 for x in v["mean_diff_by_year"].values())
        assert v["eligible_to_serve"] == rule, m
