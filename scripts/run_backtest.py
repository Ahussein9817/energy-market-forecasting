"""Walk-forward day-ahead backtest, N.Y.C. zone, NYISO day-ahead LBMP.

    python scripts/build_dataset.py          # once, downloads data
    python scripts/run_backtest.py

Protocol (fixed before looking at test results):
  * Origin = end of local day D, target = 24 hourly prices of D+1.
  * Hyperparameters: 4-config XGBoost grid scored ONLY on 2022 (trained on 2019-2021), then frozen.
  * Test: 2023-01-01 .. last available day (2024-12-30). XGBoost refit each quarter on all
    targets strictly before that quarter. SARIMAX refit every day on the trailing 56 days.
  * Feature set C (and SARIMAX-X) use the target day's ACTUAL temperature = perfect-foresight
    weather. They are optimistic upper bounds, reported separately from A/B.
  * Outputs: results/predictions_test.parquet, backtest_metrics.json, backtest_report.md,
    run_metadata.json. MAPE is not reported (see docs/data-sources.md).
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
from src import panel as pn  # noqa: E402
from src.evaluation import mae, summarize, walk_forward_xgb  # noqa: E402
from src.forecast_models import XGB_DEFAULT, sarimax_day_ahead, seasonal_naive  # noqa: E402

TEST_START = "2023-01-01"
VAL_START, VAL_END = "2022-01-01", "2022-12-31"
SARIMAX_WINDOW_DAYS = 56
GRID = [{"max_depth": d, "n_estimators": n} for d in (4, 6) for n in (300, 600)]
MODELS = ["naive_24h", "naive_168h", "sarimax", "sarimax_x_oracle_temp",
          "xgb_A", "xgb_B", "xgb_C_oracle_temp"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sarimax_series(price, temp, test_days, use_exog):
    out, fails = {}, 0
    for i, T in enumerate(test_days):
        D = T - pd.Timedelta(days=1)
        lo = D - pd.Timedelta(days=SARIMAX_WINDOW_DAYS - 1)
        hist = price.loc[lo:D].to_numpy().ravel()
        kw = {}
        if use_exog:
            kw = {"exog_hist": temp.loc[lo:D].to_numpy().ravel()[:, None],
                  "exog_future": temp.loc[T].to_numpy()[:, None]}
        fc = sarimax_day_ahead(hist, **kw)
        if fc is None:
            fails += 1
            fc = np.full(24, np.nan)
        for h in range(24):
            out[(T, h)] = fc[h]
        if (i + 1) % 100 == 0:
            print(f"  sarimax{'-x' if use_exog else ''}: {i + 1}/{len(test_days)} days, {fails} failures", flush=True)
    s = pd.Series(out)
    s.index = pd.MultiIndex.from_tuples(s.index, names=["date", "hour"])
    return s, fails


def main():
    t0 = time.time()
    data = ROOT / "data/processed"
    files = [data / "nyiso_dam_lbmp.parquet", data / "nyiso_load.parquet", data / "nyc_temperature.parquet"]
    P = pn.build_panels(*[pd.read_parquet(f) for f in files])
    price, temp = P["price"], P["temp"]
    last = price.index.max()
    test_days = pd.date_range(TEST_START, last, freq="D")
    print(f"panel {price.index.min().date()}..{last.date()}; test days {len(test_days)}", flush=True)

    # ---- tuning on 2022 only -------------------------------------------------
    tuning = []
    val_y = price.loc[VAL_START:VAL_END].stack()
    for g in GRID:
        p = walk_forward_xgb(P, "C", VAL_START, VAL_END, params=g)
        e = mae(val_y.loc[p.index], p)
        tuning.append({**g, "val_MAE_2022": e})
        print("  tune", g, round(e, 3), flush=True)
    best = min(tuning, key=lambda r: r["val_MAE_2022"])
    best_params = {k: best[k] for k in ("max_depth", "n_estimators")}
    print("chosen", best_params, flush=True)

    # ---- test predictions ----------------------------------------------------
    preds = {"naive_24h": seasonal_naive(price, 1), "naive_168h": seasonal_naive(price, 7)}
    for fs, name in (("A", "xgb_A"), ("B", "xgb_B"), ("C", "xgb_C_oracle_temp")):
        preds[name] = walk_forward_xgb(P, fs, TEST_START, str(last.date()), params=best_params)
        print("done", name, flush=True)
    fails = {}
    preds["sarimax"], fails["sarimax"] = sarimax_series(price, temp, test_days, False)
    preds["sarimax_x_oracle_temp"], fails["sarimax_x_oracle_temp"] = sarimax_series(price, temp, test_days, True)

    y = price.loc[TEST_START:].stack()
    y.index.names = ["date", "hour"]
    df = pd.DataFrame({"y": y})
    for k, v in preds.items():
        df[k] = v
    df = df.loc[df.index.get_level_values("date").isin(test_days)]
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    df.to_parquet(out / "predictions_test.parquet")

    hi = float(price.loc[:"2022-12-31"].stack().quantile(0.95))  # from pre-test data only
    summ = summarize(df, MODELS, ref="naive_168h", high_price_threshold=hi)
    (out / "backtest_metrics.json").write_text(json.dumps(summ, indent=2))
    meta = {
        "protocol": "see scripts/run_backtest.py docstring", "zone": "N.Y.C.", "test_start": TEST_START,
        "test_end": str(last.date()), "n_test_days": len(test_days), "sarimax_window_days": SARIMAX_WINDOW_DAYS,
        "sarimax_order": [[1, 0, 1], [1, 0, 1, 24]], "sarimax_failures": fails,
        "xgb_defaults": XGB_DEFAULT, "tuning_grid_results_2022": tuning, "chosen_params": best_params,
        "high_price_threshold_pre2023_p95": hi,
        "data_sha256": {f.name: sha256(f) for f in files},
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
        "python": platform.python_version(),
        "packages": {m: __import__(m).__version__ for m in ("pandas", "numpy", "xgboost", "statsmodels", "scipy", "sklearn")},
        "runtime_seconds": round(time.time() - t0),
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"done in {meta['runtime_seconds']}s")


if __name__ == "__main__":
    main()
