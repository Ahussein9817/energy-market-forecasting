
# Energy Market Price Forecasting Under Exogenous Shocks

**Status:** Working retrospective backtest for one market. A reproducible NYISO data pipeline, a walk-forward day-ahead price backtest (naive, SARIMAX, XGBoost) and a forecast-weather experiment exist, with tests and CI. Shock/causal analysis, gas data and deep models are **not** done (see "Current state" and "Current Progress").  
**Timeline:** Repository created 2025-12-22 (git history); earlier dates in the notebook and a previous README version were not backed by commits.  
**Original goal (partly done):** Build time-series forecasting models to predict electricity price volatility and analyze the causal impact of exogenous shocks (weather extremes, geopolitical events, supply disruptions) on energy markets.

---


## Current state (verified 2026-11-11)

What exists in this repository:
- `notebooks/01_data_exploration.ipynb`: exploratory analysis on a **synthetic, `np.random`-generated price series**. No saved outputs. It tells us nothing about real electricity markets.
- `notebooks/02_feature_engineering.ipynb`: empty placeholder.
- **Real-data foundation (added 2026-11-11):** `src/data_loader.py` and `scripts/build_dataset.py` download NYISO day-ahead zonal prices and integrated load plus Open-Meteo temperature for 2019-2024 (no API keys), with DST-safe UTC timestamps. Measured result (`docs/data-quality-report.md`): 52,608 hourly rows per zone for each series, no missing hours, no duplicates, 15 price zones (789,120 price rows) and 11 load zones (578,688 load rows). Sources, caveats and the move from PJM to NYISO are in `docs/data-sources.md`. 12 unit tests cover the parsers (including both DST transitions), retry/caching and weather parsing.

- **Forecasting backtest (added 2026-11-11):** walk-forward day-ahead forecasts for the N.Y.C. zone over 2023-01-01 to 2024-12-30 (730 days, 17,520 hourly points) from two naive benchmarks, SARIMAX and XGBoost; results below and in `results/backtest_report.md`.

What does **not** exist: natural-gas data, LSTM or other deep models, any causal or shock analysis, any live or trading use. The data is NYISO (not PJM as originally planned). Sections of this README labelled as the original plan (Methodology, Data Sources, shock variables) describe intended work, not results.

## Backtest results (NYISO N.Y.C. zone, day-ahead price, 2023-2024)

Setup (fixed before looking at test results; code in `scripts/run_backtest.py`, tests in `tests/test_modeling.py`): at the end of day D forecast the 24 hourly day-ahead prices of D+1 using data up to D; XGBoost hyperparameters picked on 2022 only and then frozen; XGBoost refit each quarter on all earlier days; SARIMAX(1,0,1)(1,0,1,24) refit every day on the trailing 56 days. A leakage test asserts that features for a target day do not change when all prices and loads from that day onward are altered. Metrics are MAE/RMSE in $/MWh (no MAPE: low prices make it unstable), with Diebold-Mariano tests on daily losses.

| Model | MAE | RMSE | note |
|---|---|---|---|
| Same hour last week (168 h) | 13.26 | 31.73 | weak benchmark here |
| **Same hour yesterday (24 h)** | **6.03** | 13.55 | strongest simple benchmark |
| SARIMAX | 6.47 | 13.02 | slightly worse MAE, best RMSE among non-oracle models |
| XGBoost, prices + calendar | 6.13 | 13.51 | no better than yesterday-naive (DM p=0.78) |
| XGBoost, + load lags | 6.08 | 13.35 | no better than yesterday-naive (DM p=0.89) |
| XGBoost, + **actual** target-day temperature | 5.02 | 11.04 | 17% lower MAE than yesterday-naive (DM p<0.001), but uses a perfect-foresight weather forecast |

What this does and does not show:
- Without information about the target day, the models did **not** beat "same hour yesterday" on MAE. The only clear gain comes from feeding in the target day's actual temperature; whether a real weather forecast keeps that gain is measured in the next section.
- Every model badly under-forecasts price spikes (bias of -24 to -36 $/MWh in the top 5% of hours), and the first quarters of each year are the hardest (winter volatility).
- No natural-gas, generation-mix or outage data is used; those are the main known drivers and would be the next thing to add. One zone, one 2-year window, day-ahead prices only.
- The metrics are regenerable from the committed predictions with `python scripts/evaluate_predictions.py` (verified identical to the saved metrics). The full backtest was run once (about 39 minutes); re-running it from scratch to confirm identical predictions has not been done. Run metadata (package versions, data checksums, tuning grid, git commit) is in `results/run_metadata.json`.
- Not a trading result and no claim of trading experience: this is a retrospective forecast evaluation.

## Project Overview

> **Scope note.** The Project Overview, Methodology and Data Sources sections below are the original project plan (their EIA/NOAA/PJM sources were not used). What has actually been built and measured is described in "Current state", "Backtest results" and "Forecast-weather experiment". The shock/causal-analysis question (2), LSTM, Random Forest and directional accuracy are **not implemented**; MAPE is deliberately not used (see Backtest results).

This project investigates how electricity prices respond to various external factors and develops predictive models to forecast short-term price movements. The focus is on understanding both predictable patterns (seasonality, demand cycles) and unpredictable shocks (extreme weather, geopolitical tensions, infrastructure failures).

### Research Questions
1. How accurately can we forecast day-ahead electricity prices using historical market data and weather variables?
2. What is the causal impact of specific exogenous shocks (e.g., cold snaps, supply chain disruptions) on price volatility?
3. Which modeling approach (statistical vs. machine learning) performs best for different forecasting horizons?

---

## Methodology

### Modeling Approaches
- **Baseline Models:** Naive forecasts, moving averages, ARIMA
- **Statistical Time Series:** SARIMAX (capturing seasonality and exogenous variables)
- **Machine Learning:** XGBoost, Random Forest (with engineered features)
- **Deep Learning:** LSTM networks (for capturing long-term dependencies)

### Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Directional accuracy (predicting price increases vs. decreases)

### Causal Analysis
- Event study methodology to isolate impact of specific shocks
- Counterfactual analysis comparing actual prices to model predictions without shock variables
- Sensitivity analysis across different market conditions

---

## Data Sources

### Primary Data
- **Electricity Prices:** US Energy Information Administration (EIA) - day-ahead and real-time LMP data
- **Weather Data:** NOAA Climate Data Online (temperature, precipitation, wind speed)
- **Energy Demand:** EIA hourly electricity demand by region
- **Natural Gas Prices:** Henry Hub spot prices (strong correlation with electricity)

### Exogenous Shock Variables
- Extreme weather events (heat waves, cold snaps, hurricanes)
- Geopolitical events (sanctions, trade disputes, conflicts)
- Infrastructure outages (power plant failures, transmission issues)
- Policy changes (carbon pricing, renewable energy mandates)

### Data Coverage
- **Geographic Focus (as built):** NYISO, N.Y.C. zone (the original plan named PJM; see `docs/data-sources.md` for why it changed)
- **Time Period:** 2019-2024 data; the backtest test window is 2023-01-01 to 2024-12-30
- **Frequency:** Hourly day-ahead prices, forecast as 24-hour days

---

## Repository Structure

```
energy-market-forecasting/
├── src/
│   ├── data_loader.py          # NYISO prices/load and Open-Meteo actual temperature
│   ├── panel.py                # DST-correct local-date x hour panels
│   ├── features.py             # leakage-safe features (sets A, B, C oracle, D forecast weather)
│   ├── forecast_models.py      # naive benchmarks, XGBoost, SARIMAX
│   ├── forecast_weather.py     # archived day-ahead forecast temperature (Open-Meteo Previous Runs)
│   └── evaluation.py           # metrics, Diebold-Mariano, walk-forward driver
├── scripts/
│   ├── build_dataset.py            # download and build data/processed (not committed)
│   ├── run_backtest.py             # original backtest (about 39 minutes)
│   ├── evaluate_predictions.py     # regenerate backtest metrics/report from committed predictions
│   ├── run_forecast_weather.py     # forecast-weather experiment (about 15 seconds)
│   └── evaluate_forecast_weather.py
├── results/                    # committed predictions, metrics, reports, run metadata
├── docs/                       # data sources, data-quality report, claim-evidence table
├── tests/                      # offline tests (network is mocked)
└── notebooks/                  # 01 uses synthetic data (not real prices); 02 is an empty placeholder
```

---

## Current Progress

### Done (see `docs/claim-evidence.md` for the evidence behind each item)
- [x] Reproducible NYISO price/load pipeline with data-quality report and tests
- [x] Walk-forward day-ahead backtest: naive benchmarks, SARIMAX, XGBoost, leakage tests, Diebold-Mariano
- [x] Oracle vs forecast-weather comparison (GFS day-ahead temperature)
- [x] Offline tests, ruff and GitHub Actions CI, including metric regeneration from committed predictions

### Not done
- [ ] Natural-gas, generation-mix and outage data (gas needs a free EIA API key)
- [ ] Confirming identical predictions from a second full backtest run
- [ ] LSTM, Random Forest, event/shock analysis, any causal claim
- [ ] Docker packaging and model registry
- [ ] NYISO data-licence review

---

## Installation & Usage

Tested on Python 3.12.

```bash
git clone https://github.com/Ahussein9817/energy-market-forecasting.git
cd energy-market-forecasting
pip install -r requirements.txt -r requirements-dev.txt
pytest -q && ruff check .

# regenerate all reported metrics from the committed predictions (no download needed)
python scripts/evaluate_predictions.py
python scripts/evaluate_forecast_weather.py
```

To rebuild from scratch (network needed; data is not committed):
```bash
python scripts/build_dataset.py          # NYISO + Open-Meteo actual temperature
python scripts/run_backtest.py           # about 39 minutes
python scripts/run_forecast_weather.py   # fetches archived GFS forecasts, about 15 seconds
```

## Forecast-weather experiment (added 2026-11-12)

The 5.02 MAE above uses perfect-foresight weather. This experiment replaces it with an archived **day-ahead forecast**: GFS 2 m temperature from the Open-Meteo Previous Runs API issued 1 day before the valid hour (a 2-day lead is the leakage-safe sensitivity, because the run initialisation time behind the 1-day value is undocumented). Same walk-forward protocol and the same frozen XGBoost hyperparameters (chosen on 2022 using the oracle feature set; not retuned). Code: `scripts/run_forecast_weather.py`, `src/forecast_weather.py`; report: `results/forecast_weather_report.md`; regenerate metrics with `python scripts/evaluate_forecast_weather.py`.

23 test days (2023-12-29 to 2024-01-20) have missing forecast hours from one upstream outage; they are excluded for **every** model in the table (709 days remain, identical points for all models). Nothing is imputed.

| Model (709 days) | MAE | RMSE | note |
|---|---|---|---|
| Same hour yesterday | 5.78 | 13.14 | reference |
| XGBoost B, no weather (control, same training dates) | 6.14 | 13.00 | not better than yesterday-naive |
| XGBoost + forecast temp (1-day lead), trained on actuals | 4.72 | 10.35 | 18% lower MAE than yesterday-naive (DM p<0.001) |
| XGBoost + forecast temp (1-day lead), trained on forecasts | 5.35 | 10.76 | 8% lower MAE, not significant (DM p=0.12) |
| XGBoost + forecast temp (2-day lead), trained on actuals | 4.71 | 10.33 | same as 1-day lead |
| XGBoost + actual temp (oracle) | 4.77 | 10.43 | upper bound |

What this shows and does not show:
- With a real day-ahead forecast the weather-based model still beats "same hour yesterday" on this window, and the gap to the oracle is not statistically distinguishable here (DM p=0.44). The GFS forecast itself is close to the observed temperature (MAE 1.75 deg C, bias +0.75).
- The gain depends on how the model is trained: training on actual temperature and predicting with the forecast did better than training only on the one to three years of archived forecasts, whose history is short (about 1,359 training days at most).
- One zone, one weather point, GFS only, one 2-year window, one frozen hyperparameter set. No claim is made about why prices move, and no gas, outage or generation-mix data is used. Not a trading result.
- Weather data by Open-Meteo.com (CC BY 4.0); raw responses are cached in the gitignored `data/external/` and are not committed.

---

## Key Findings

None yet. Earlier versions of this README listed expected relationships (seasonality, gas-price correlation, weather effects) as preliminary findings; no code or data in this repository computed them, so they were removed. They are hypotheses to test once real data is loaded.

---

## References & Related Work

- Weron, R. (2014). Electricity price forecasting: A review of the state-of-the-art with a look into the future. *International Journal of Forecasting*, 30(4), 1030-1081.
- Lago, J., et al. (2021). Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark. *Applied Energy*, 293, 116983.
- Nowotarski, J., & Weron, R. (2018). Recent advances in electricity price forecasting: A review of probabilistic forecasting. *Renewable and Sustainable Energy Reviews*, 81, 1548-1568.

---

## Contact

**Amina Ali Hussein**  
aminaali9816@gmail.com | [LinkedIn](https://linkedin.com/in/amina-ali-hussein) | [GitHub](https://github.com/Ahussein9817)

---

## License

MIT License - feel free to use this code for your own research and projects.
