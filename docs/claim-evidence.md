# Claim-to-evidence table (energy-market-forecasting)

Evidence types: **regenerated** (metrics recomputed from committed predictions), **full rerun**, **test**, **CI**. State on branch `improve/portfolio-refinement` (see `git log`; CI green on `main`. Rows below were extended after the forecast-weather experiment). All results are NYISO N.Y.C. zone, day-ahead price, 2023-01-01..2024-12-30; no gas, outage or generation-mix data.

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
| Result with real forecast weather (GFS day-ahead, 709 days): XGBoost MAE 4.72 vs 5.78 for yesterday-naive; no-weather control 6.14 | VERIFIED (regenerated from committed predictions) | regenerated, CI | `scripts/evaluate_forecast_weather.py`, `results/forecast_weather_report.md` | One run of the model fits (not re-run to confirm identical predictions); GFS only; one frozen hyperparameter set tuned on the oracle feature set; 23 outage days excluded for all models |
| Forecast-weather result is robust to the undocumented forecast run time | PARTIAL | regenerated | same | `previous_day2` gives the same MAE (4.71); this shows no sensitivity but does not prove availability at the cutoff |
| Forecast temperature as good as perfect-foresight temperature | PARTIAL | regenerated | same | Difference not statistically significant on 709 days (DM p=0.44); one zone, one window |
| LSTM, gas data, shock/market analysis, Docker, model registry | PROPOSED (not done) | none | | |
| 44 tests, ruff, CI including regeneration of both metric sets | VERIFIED | CI | run 43371758549 on `main` | |
| The end-of-day backtest corresponds to a forecast someone could have made | **UNSUPPORTED** | probe, `results/publication_times.json` | `docs/operational-timing.md` | D+1 prices are public ~09:32 ET on D, about 14 h before the backtest origin. Valid as a predictability experiment only |
| Day-ahead price publication time (~09:32 ET) | VERIFIED (12 sampled days, Nov 2026) | probe | `results/publication_times.json` | `Last-Modified` is the latest revision time; official current rules not yet confirmed |
| At a 05:00 ET origin, XGBoost + 48-hour-lead forecast weather beats repeating yesterday's price: MAE 4.84 vs 5.78 (709 days) | VERIFIED (regenerated from committed predictions) | regenerated, CI | `scripts/evaluate_operational.py`, `results/operational_report.md` | Pre-registered rule met (CI of difference [-1.36, -0.56], DM p<0.001, better in 2023 and 2024). Weather input is an archived-forecast proxy; hyperparameters tuned on 2022 with these features; one zone; one run of the model fits |
| Models without weather beat the baseline at the operational origin | UNSUPPORTED | regenerated | same | OB: MAE 5.71, CI [-0.61, +0.61]; trained only on archived forecasts: CI crosses zero |
| Ingestion is idempotent, validates input and handles failures | VERIFIED (tests) and exercised once on real NYISO files | test, CI; one real run 2026-11-13 | `tests/test_service_ingest.py` | Synthetic NYISO-format fixtures; the real run found and fixed one issue (daily files roll off after ~12 days) |
| The forecast job serves a baseline whenever the model cannot be trusted | VERIFIED (tests, nine fallback scenarios) | test, CI | `tests/test_service_forecast.py` | |
| The API is read-only and returns consistent errors | VERIFIED (tests; exercised against a real database) | test, CI | `tests/test_service_api.py` | No authentication needed because nothing can be changed; an admin endpoint is not built |
| Live operation | PARTIAL | one live forecast stored 2026-11-13 00:58 ET (for 2026-11-14) | local database (not committed) | n=1, not yet evaluated; no performance claim; no scheduler configured; local machine only |
| Dashboard, monitoring, alerts, Docker, deployment | PROPOSED (not built) | none | | |

