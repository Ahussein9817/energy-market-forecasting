# Claim-to-evidence table (energy-market-forecasting)

Evidence types: **regenerated** (metrics recomputed from committed predictions), **full rerun**, **test**, **CI**. State on branch `improve/portfolio-refinement` (head 16685eb, CI run 43162033391 green). All results are NYISO N.Y.C. zone, day-ahead price, 2023-01-01..2024-12-30; no gas, outage or generation-mix data.

| Claim | Status | Evidence type | Where | Caveat |
|---|---|---|---|---|
| NYISO day-ahead prices (15 zones) and load (11 zones), 2019-2024, no gaps or duplicates | VERIFIED | test, data-quality report | `docs/data-quality-report.md`, `tests/test_data_loader.py` | Needs network to rebuild; data licence terms not verified |
| DST-correct local-date x hour panels (25- and 23-hour days) | VERIFIED | test | `tests/test_modeling.py`, `src/panel.py` | |
| Walk-forward day-ahead backtest, tuned on 2022 only, 730 test days | VERIFIED (design and tests) | test | `scripts/run_backtest.py`, leakage test | |
| MAE/RMSE/bias/rMAE and Diebold-Mariano results | VERIFIED | regenerated, CI | `scripts/evaluate_predictions.py`, `results/backtest_report.md` | Regenerated from committed predictions (max diff 8.9e-16) |
| The predictions themselves reproduce from a rerun | PARTIAL | one 2,331 s run | `results/run_metadata.json` | A second full run to confirm identical predictions has not been done |
| No non-oracle model beats "same hour yesterday" on MAE (6.03) | VERIFIED | regenerated | `results/backtest_metrics_vs_naive24h.json` | One zone, two years |
| XGBoost with actual target-day temperature: MAE 5.02 (17% lower) | VERIFIED as an **oracle** (perfect-foresight weather) | regenerated | same | Not an operational forecast; a real forecast-weather result has not been measured |
| All models under-forecast price spikes | VERIFIED | regenerated | `results/backtest_report.md` | |
| Result with real forecast weather | PROPOSED | none | | Feasibility assessed; not run |
| LSTM, gas data, shock/market analysis, Docker, model registry | PROPOSED (not done) | none | | |
| 31 tests, ruff, CI including metric regeneration | VERIFIED | CI | run 43162033391 | |
