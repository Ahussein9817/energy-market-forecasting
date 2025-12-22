
# Energy Market Price Forecasting Under Exogenous Shocks

**Status:** Work in Progress  
**Timeline:** June 2024 - Present  
**Goal:** Build time-series forecasting models to predict electricity price volatility and analyze the causal impact of exogenous shocks (weather extremes, geopolitical events, supply disruptions) on energy markets.

---

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
- [x] Initial exploratory data analysis on historical prices

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
git clone https://github.com/yourusername/energy-market-forecasting.git
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

## Key Findings (Preliminary)

*This section will be updated as the project progresses with key insights and model results.*

- Initial EDA shows strong seasonality in electricity prices (summer/winter peaks)
- High correlation between natural gas prices and electricity prices (r > 0.7)
- Weather variables (temperature extremes) show significant impact on price volatility

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

# energy-market-forecasting
Time-series forecasting of electricity prices with analysis of exogenous shocks
>>>>>>> 94301f29fade533abb4ffc74de6127e3c0c6e16c

## Recent Updates
- Added initial EDA notebook (Dec 2024)
- Exploring baseline forecasting approaches

