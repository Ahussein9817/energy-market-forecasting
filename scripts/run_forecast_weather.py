"""Forecast-weather experiment (E-1): does weather still help when only a day-ahead FORECAST is known?

    python scripts/build_dataset.py            # once (prices, load, actual temperature)
    python scripts/run_forecast_weather.py     # fetches GFS previous-runs temperature (free, no key)
    python scripts/evaluate_forecast_weather.py

Fixed before looking at results:
  * Same origin, target, test window and walk-forward refits as scripts/run_backtest.py; XGBoost
    hyperparameters are the ones frozen there (chosen on 2022 only). They are NOT retuned here.
  * Weather feature = GFS (`gfs_global`) 2 m temperature issued 1 day before the valid hour
    (`previous_day1`). `previous_day2` is run as a leakage-safe sensitivity, because the run
    initialisation time behind `previous_day1` is undocumented.
  * Variant (a): train with ACTUAL temperature in the weather column, predict with the forecast.
    Variant (b): train on archived FORECASTS (rows without a forecast are dropped).
  * Control: xgb_B (no weather) retrained on exactly the dates variant (b) can train on.
  * Days whose target-day forecast has any missing hour are excluded for ALL models in the
    comparison; nothing is imputed and actuals are never used to fill forecasts.
  * naive_*, xgb_B and xgb_C_oracle_temp come from the committed results/predictions_test.parquet.
Outputs: results/forecast_weather_predictions.parquet, results/forecast_weather_run_metadata.json.
Weather data by Open-Meteo.com (CC BY 4.0); raw API responses stay in the gitignored data/external.
"""
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import forecast_weather as fw  # noqa: E402
from src import panel as pn  # noqa: E402
from src.evaluation import walk_forward_xgb  # noqa: E402
from src.features import make_features  # noqa: E402

TEST_START = "2023-01-01"
FETCH_START = "2021-01-01"  # 2021 is partly missing upstream; rows without a forecast are dropped
FROM_EXISTING = ["naive_24h", "naive_168h", "xgb_B", "xgb_C_oracle_temp"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    t0 = time.time()
    data = ROOT / "data/processed"
    P = pn.build_panels(*[pd.read_parquet(data / f) for f in
                          ("nyiso_dam_lbmp.parquet", "nyiso_load.parquet", "nyc_temperature.parquet")])
    price = P["price"]
    last = price.index.max()
    test_days = pd.date_range(TEST_START, last, freq="D")
    params = json.loads((ROOT / "results/run_metadata.json").read_text())["chosen_params"]
    cache = ROOT / "data/external"

    fc, weather_meta = {}, {}
    for lead in (1, 2):
        raw = fw.fetch_range(FETCH_START, str((last + pd.Timedelta(days=1)).date()), lead, cache)  # +1 day: local evening hours fall on the next UTC date
        fc[lead] = fw.to_forecast_panel(raw, price.index)
        miss = fw.missing_days(fc[lead].loc[TEST_START:])
        err = (fc[lead].loc[TEST_START:] - P["temp"].loc[TEST_START:]).stack().dropna()
        weather_meta[f"previous_day{lead}"] = {
            "test_days_with_missing_forecast": len(miss),
            "missing_day_first": str(miss.min().date()) if len(miss) else None,
            "missing_day_last": str(miss.max().date()) if len(miss) else None,
            "forecast_minus_actual_temp_C": {"MAE": float(err.abs().mean()), "bias": float(err.mean()),
                                             "RMSE": float(np.sqrt((err ** 2).mean())), "n_hours": int(len(err))},
            "files_sha256": {f.name: sha256(f) for f in sorted(cache.glob(f"gfs_previous_day{lead}_*.parquet"))}}
        print(f"lead {lead}: {len(miss)} test days with missing forecast hours", flush=True)

    excluded = fw.missing_days(fc[1].loc[TEST_START:]).union(fw.missing_days(fc[2].loc[TEST_START:]))
    preds = {}
    for lead in (1, 2):
        panels = {**P, "temp_fc": fc[lead]}
        # (a) train on actual temperature in the forecast column, predict with the forecast
        a_panels = {**P, "temp_fc": P["temp"]}
        preds[f"xgb_D{lead}_train_actual"] = walk_forward_xgb(panels, "D", TEST_START, str(last.date()),
                                                              params=params, train_panels=a_panels)
        # (b) train on archived forecasts
        preds[f"xgb_D{lead}_train_fc"] = walk_forward_xgb(panels, "D", TEST_START, str(last.date()), params=params)
        print(f"done lead {lead}", flush=True)

    # control: no-weather model B trained only on the dates variant (b) (lead 1) can train on
    Xd, _ = make_features({**P, "temp_fc": fc[1]}, "D")
    matched = Xd.index.get_level_values("date").unique()
    preds["xgb_B_matched_window"] = walk_forward_xgb(P, "B", TEST_START, str(last.date()),
                                                     params=params, train_dates=matched)

    old = pd.read_parquet(ROOT / "results/predictions_test.parquet")
    df = old[["y"] + FROM_EXISTING].copy()
    for k, v in preds.items():
        df[k] = v
    dates = df.index.get_level_values("date")
    df = df.loc[dates.isin(test_days)]
    df.to_parquet(ROOT / "results/forecast_weather_predictions.parquet")

    meta = {
        "protocol": "see scripts/run_forecast_weather.py docstring", "zone": "N.Y.C.", "weather_model": fw.MODEL,
        "test_start": TEST_START, "test_end": str(last.date()), "n_test_days": len(test_days),
        "xgb_params_frozen_from_run_backtest": params,
        "excluded_days_missing_forecast": [str(d.date()) for d in excluded], "n_excluded_days": len(excluded),
        "matched_window_train_days": int(len(matched)),
        "weather": weather_meta,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
        "python": platform.python_version(),
        "committed_predictions_sha256": sha256(ROOT / "results/predictions_test.parquet"),
        "runtime_seconds": round(time.time() - t0),
    }
    (ROOT / "results/forecast_weather_run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"done in {meta['runtime_seconds']}s")


if __name__ == "__main__":
    main()
