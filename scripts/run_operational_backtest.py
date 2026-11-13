"""Operational-origin backtest: forecast at 05:00 ET on day D the 24 day-ahead prices of D+1.

    python scripts/build_dataset.py              # once
    python scripts/run_operational_backtest.py   # needs data/processed and the cached previous_day2 files
    python scripts/evaluate_operational.py

PRE-REGISTERED before looking at any result (see docs/operational-timing.md):
  * Feature sets OB (prices through D, calendar, load through D-1) and OD (OB + forecast temperature
    of D+1). Day-D load and observed temperature are NOT used (not known at the origin).
  * Weather proxy: Open-Meteo `previous_day2` (48 h lead, issued before the origin). `previous_day1`
    is not used because it is not available at this origin.
  * Hyperparameters: the same 4-config XGBoost grid as run_backtest.py, scored on 2022 only
    (trained 2019-2021), separately for OB and OD (variant a); OD variant b reuses OD(a)'s choice.
    Then frozen. The earlier backtest tuned with the oracle feature set; this one does not.
  * Test: 2023-01-01..2024-12-30, XGBoost refit each quarter on targets strictly before the quarter.
  * Variants: OB (no weather); OB_matched (OB trained only on the dates variant b can train on, the
    fair control); OD(a) trained with ACTUAL temperature in the weather column, predicting with the
    forecast; OD(b) trained on archived forecasts. Baseline: same hour of day D.
  * Days whose forecast has a missing hour are excluded for ALL models. Nothing is imputed.
  * Serving eligibility rule (applied in evaluate_operational.py): a model may serve only if, versus
    the baseline on the common days, (1) the 95% block-bootstrap CI of the daily MAE difference is
    entirely below zero, (2) Diebold-Mariano p < 0.05, and (3) the mean daily MAE difference is
    negative in each calendar year of the test window.
Outputs: results/operational_predictions.parquet, results/operational_run_metadata.json.
Weather data by Open-Meteo.com (CC BY 4.0); raw responses stay in the gitignored data/external.
"""
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import forecast_weather as fw  # noqa: E402
from src import panel as pn  # noqa: E402
from src.evaluation import mae, walk_forward_xgb  # noqa: E402
from src.features import make_features  # noqa: E402
from src.forecast_models import seasonal_naive  # noqa: E402

TEST_START = "2023-01-01"
VAL_START, VAL_END = "2022-01-01", "2022-12-31"
FETCH_START = "2021-01-01"
LEAD_DAYS = 2
GRID = [{"max_depth": d, "n_estimators": n} for d in (4, 6) for n in (300, 600)]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tune(P, fs, val_y, **kw):
    rows = []
    for g in GRID:
        p = walk_forward_xgb(P, fs, VAL_START, VAL_END, params=g, **kw)
        rows.append({**g, "val_MAE_2022": mae(val_y.loc[p.index], p)})
        print("  tune", fs, g, round(rows[-1]["val_MAE_2022"], 3), flush=True)
    best = min(rows, key=lambda r: r["val_MAE_2022"])
    return {k: best[k] for k in ("max_depth", "n_estimators")}, rows


def main():
    t0 = time.time()
    data = ROOT / "data/processed"
    P = pn.build_panels(*[pd.read_parquet(data / f) for f in
                          ("nyiso_dam_lbmp.parquet", "nyiso_load.parquet", "nyc_temperature.parquet")])
    price = P["price"]
    last = price.index.max()
    test_days = pd.date_range(TEST_START, last, freq="D")
    raw = fw.fetch_range(FETCH_START, str((last + pd.Timedelta(days=1)).date()), LEAD_DAYS, ROOT / "data/external")
    fc = fw.to_forecast_panel(raw, price.index)
    excluded = fw.missing_days(fc.loc[TEST_START:])
    panels = {**P, "temp_fc": fc}
    a_panels = {**P, "temp_fc": P["temp"]}          # actual temperature in the forecast column (variant a)
    val_y = price.loc[VAL_START:VAL_END].stack()

    p_ob, grid_ob = tune(panels, "OB", val_y)
    p_od, grid_od = tune(panels, "OD", val_y, train_panels=a_panels)
    end = str(last.date())
    preds = {
        "xgb_OB": walk_forward_xgb(panels, "OB", TEST_START, end, params=p_ob),
        "xgb_OD_train_actual": walk_forward_xgb(panels, "OD", TEST_START, end, params=p_od, train_panels=a_panels),
        "xgb_OD_train_fc": walk_forward_xgb(panels, "OD", TEST_START, end, params=p_od),
    }
    Xd, _ = make_features(panels, "OD")
    matched = Xd.index.get_level_values("date").unique()
    preds["xgb_OB_matched_window"] = walk_forward_xgb(panels, "OB", TEST_START, end, params=p_ob, train_dates=matched)
    preds["naive_24h"] = seasonal_naive(price, 1)

    y = price.loc[TEST_START:].stack()
    y.index.names = ["date", "hour"]
    df = pd.DataFrame({"y": y})
    for k, v in preds.items():
        df[k] = v
    df = df.loc[df.index.get_level_values("date").isin(test_days)]
    out = ROOT / "results"
    df.to_parquet(out / "operational_predictions.parquet")
    files = sorted((ROOT / "data/external").glob(f"gfs_previous_day{LEAD_DAYS}_*.parquet"))
    meta = {
        "protocol": "see scripts/run_operational_backtest.py docstring", "zone": "N.Y.C.", "origin": "05:00 America/New_York on day D",
        "weather_model": fw.MODEL, "weather_lead_days": LEAD_DAYS, "test_start": TEST_START, "test_end": end,
        "n_test_days": len(test_days), "excluded_days_missing_forecast": [str(d.date()) for d in excluded],
        "n_excluded_days": len(excluded), "chosen_params": {"OB": p_ob, "OD": p_od},
        "tuning_grid_results_2022": {"OB": grid_ob, "OD_train_actual": grid_od},
        "matched_window_train_days": int(len(matched)),
        "weather_files_sha256": {f.name: sha256(f) for f in files},
        "data_sha256": {f.name: sha256(data / f.name) for f in sorted(data.glob("*.parquet"))},
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
        "python": platform.python_version(), "runtime_seconds": round(time.time() - t0)}
    (out / "operational_run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"done in {meta['runtime_seconds']}s; chosen {meta['chosen_params']}; excluded {len(excluded)} days")


if __name__ == "__main__":
    main()
