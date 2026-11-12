
# Energy Market Price Forecasting Under Exogenous Shocks

**Status:** Early stage. A real-data ingestion pipeline (NYISO + weather) exists; no forecasting models or results exist yet (see "Current state").  
**Timeline:** Repository created 2025-12-22 (git history); earlier dates in the notebook and a previous README version were not backed by commits.  
**Goal:** Build time-series forecasting models to predict electricity price volatility and analyze the causal impact of exogenous shocks (weather extremes, geopolitical events, supply disruptions) on energy markets.

---


## Current state (verified 2026-11-11)

What exists in this repository:
- This plan, a folder skeleton, and a dependency list.
- `notebooks/01_data_exploration.ipynb`: exploratory analysis on a **synthetic, `np.random`-generated price series**. No saved outputs. It tells us nothing about real electricity markets.
- `notebooks/02_feature_engineering.ipynb`: empty placeholder.
- **Real-data foundation (added 2026-11-11):** `src/data_loader.py` and `scripts/build_dataset.py` download NYISO day-ahead zonal prices and integrated load plus Open-Meteo temperature for 2019-2024 (no API keys), with DST-safe UTC timestamps. Measured result (`docs/data-quality-report.md`): 52,608 hourly rows per zone for each series, no missing hours, no duplicates, 15 price zones (789,120 price rows) and 11 load zones (578,688 load rows). Sources, caveats and the move from PJM to NYISO are in `docs/data-sources.md`. 12 unit tests cover the parsers (including both DST transitions), retry/caching and weather parsing.

What does **not** exist: natural-gas data, any model (baseline or otherwise), any backtest, any evaluation metric, any result. The data is NYISO (not PJM as originally planned) and the data pipeline exists; everything about modelling below describes intended work. The files listed under "Repository Structure" that are not in the repo (`src/data_loader.py`, `preprocessing.py`, `models.py`, `evaluation.py`, `visualization.py`, notebooks 03-05) have not been written.

## Project Overview

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
- **Geographic Focus:** Initially focusing on PJM Interconnection (Mid-Atlantic/Midwest US)
- **Time Period:** 2019-2024 (5+ years of hourly/daily data)
- **Frequency:** Daily and hourly price data

---

## Repository Structure

```
energy-market-forecasting/
│
├── data/
│   ├── raw/                    # Original downloaded data
│   ├── processed/              # Cleaned and transformed data
│   └── external/               # Exogenous variables (weather, events)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # Initial EDA
│   ├── 02_feature_engineering.ipynb    # Creating model features
│   ├── 03_baseline_models.ipynb        # Simple forecasting models
│   ├── 04_advanced_models.ipynb        # ML and deep learning models
│   └── 05_shock_analysis.ipynb         # Causal impact analysis
│
├── src/
│   ├── data_loader.py          # Data ingestion scripts
│   ├── preprocessing.py        # Data cleaning functions
│   ├── models.py               # Model implementations
│   ├── evaluation.py           # Metrics and evaluation
│   └── visualization.py        # Plotting functions
│
├── results/
│   ├── figures/                # Visualizations
│   └── model_outputs/          # Predictions and performance metrics
│
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── .gitignore
```

---

## Current Progress

### Completed
- [x] Literature review on energy price forecasting methods
- [x] Identified primary data sources (EIA, NOAA)
- [x] Set up repository structure
- [x] Exploratory-analysis scaffolding on synthetic data (not historical prices)

### In Progress
- [ ] Data collection and cleaning pipeline
- [ ] Feature engineering (lagged variables, rolling statistics, weather integration)
- [ ] Baseline model implementation (ARIMA, SARIMAX)

###  Upcoming
- [ ] Machine learning model development (XGBoost, Random Forest)
- [ ] LSTM implementation for sequence modeling
- [ ] Event study analysis for exogenous shocks
- [ ] Model comparison and evaluation
- [ ] Technical report writing

---

## Installation & Usage

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- See `requirements.txt` for full dependency list

### Setup
```bash
# Clone the repository
git clone https://github.com/Ahussein9817/energy-market-forecasting.git
cd energy-market-forecasting

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter to explore notebooks
jupyter notebook
```

### Getting the Data
Data files are not included in this repository due to size. To replicate:
1. Download EIA electricity price data from: https://www.eia.gov/electricity/
2. Download NOAA weather data from: https://www.ncdc.noaa.gov/cdo-web/
3. Place raw data in `data/raw/` directory
4. Run preprocessing notebooks in order

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
