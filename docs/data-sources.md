# Data sources

All sources below were reachable without an API key on 2026-11-11. Coverage and missingness are measured in `docs/data-quality-report.md` (generated, not hand-written).

| Data | Source | Access | Granularity | Used for |
|---|---|---|---|---|
| Day-ahead zonal LBMP ($/MWh) | NYISO Market Information System, `mis.nyiso.com/public/csv/damlbmp/` | monthly zip of daily CSVs, no key | hourly, 11 load zones + external proxies | forecast target |
| Integrated actual load (MW) | NYISO, `.../public/csv/palIntegrated/` | same | hourly (as published), per zone | demand feature |
| 2 m temperature (deg C) | Open-Meteo historical weather API (`archive-api.open-meteo.com`), NYC point (40.7128, -74.0060) | HTTP JSON, no key | hourly | weather feature |
| Day-ahead **forecast** 2 m temperature (deg C) | Open-Meteo Previous Runs API (`previous-runs-api.open-meteo.com`), GFS (`gfs_global`), `temperature_2m_previous_day1` / `_day2`, same NYC point | HTTP JSON, no key | hourly | forecast-weather feature (2021-2024); archived forecast as it was N days before the valid hour |

**Why NYISO rather than PJM (the original README's focus):** PJM's Data Miner and the EIA API require registration/keys; NYISO publishes free CSVs, so the pipeline is reproducible from a fresh clone. This is a change of market from the original plan, not an oversight.

**Not yet included:** natural gas prices (Henry Hub is available through the EIA API, which needs a free key), generation mix/fuel, real-time prices, outage or event data. Nothing in this repo should be read as covering them.

**NYISO publication behaviour (verified 2026-11-13, see `docs/operational-timing.md`):** day-ahead prices for D+1 are posted ~09:32 ET on day D; actual load ~00:20 ET on D+1 and may be revised later; daily files are a rolling ~12-day window while the current month's zip holds the whole month and updates daily. Every fetched file is recorded with its URL, `Last-Modified` and SHA-256 in the operational database (not committed).

**Weather forecast data and attribution:** Weather data by Open-Meteo.com. Open-Meteo data are licensed CC BY 4.0 (attribution required); the free API is for non-commercial use (under 10,000 calls/day). This experiment makes about 8 requests. Raw API responses are cached in `data/external/` (gitignored) and are not redistributed. Upstream model: NOAA GFS. Verified 2026-11-12: `gfs_global` and `gfs_seamless` returned identical values on a June 2022 probe; `gfs_global` is used so that HRRR blending cannot enter. Known gaps: one contiguous outage of the archived GFS forecast, 2023-12-30 to 2024-01-19 UTC (23 local test days excluded for all models); 2021 is partly missing; the run initialisation time behind `previous_day1` is not documented, hence the `previous_day2` sensitivity. The stitched `temperature_2m` series and the Historical Forecast API are not day-ahead forecasts and are not used.

**Licences and terms:** not verified in this pass for NYISO. NYISO and Open-Meteo each publish their own terms of use; read them before redistributing any data. This repository does not commit raw or processed data (`data/raw`, `data/processed` are gitignored); `python scripts/build_dataset.py` re-downloads it.

**Known data caveats**
- Prices are day-ahead market-clearing prices, not real-time prices and not transaction prices; forecasting them is a different problem from forecasting real-time prices.
- NYISO price files use local Eastern timestamps with no timezone column. `src/data_loader.py` resolves DST from row order (fall-back day: 25 rows with a repeated 01:00; spring-forward: 23 rows) and stores UTC; covered by tests on both transition types.
- Negative prices occur in some zones (not N.Y.C.), and very low prices make MAPE unstable; MAE/RMSE and scaled errors are the planned primary metrics.
- Weather is one NYC point used as a coarse proxy for statewide conditions.
