# Data sources

All sources below were reachable without an API key on 2026-11-11. Coverage and missingness are measured in `docs/data-quality-report.md` (generated, not hand-written).

| Data | Source | Access | Granularity | Used for |
|---|---|---|---|---|
| Day-ahead zonal LBMP ($/MWh) | NYISO Market Information System, `mis.nyiso.com/public/csv/damlbmp/` | monthly zip of daily CSVs, no key | hourly, 11 load zones + external proxies | forecast target |
| Integrated actual load (MW) | NYISO, `.../public/csv/palIntegrated/` | same | hourly (as published), per zone | demand feature |
| 2 m temperature (deg C) | Open-Meteo historical weather API (`archive-api.open-meteo.com`), NYC point (40.7128, -74.0060) | HTTP JSON, no key | hourly | weather feature |

**Why NYISO rather than PJM (the original README's focus):** PJM's Data Miner and the EIA API require registration/keys; NYISO publishes free CSVs, so the pipeline is reproducible from a fresh clone. This is a change of market from the original plan, not an oversight.

**Not yet included:** natural gas prices (Henry Hub is available through the EIA API, which needs a free key), generation mix/fuel, real-time prices, outage or event data. Nothing in this repo should be read as covering them.

**Licences and terms:** not verified in this pass. NYISO and Open-Meteo each publish their own terms of use; read them before redistributing any data. This repository does not commit raw or processed data (`data/raw`, `data/processed` are gitignored); `python scripts/build_dataset.py` re-downloads it.

**Known data caveats**
- Prices are day-ahead market-clearing prices, not real-time prices and not transaction prices; forecasting them is a different problem from forecasting real-time prices.
- NYISO price files use local Eastern timestamps with no timezone column. `src/data_loader.py` resolves DST from row order (fall-back day: 25 rows with a repeated 01:00; spring-forward: 23 rows) and stores UTC; covered by tests on both transition types.
- Negative prices occur in some zones (not N.Y.C.), and very low prices make MAPE unstable; MAE/RMSE and scaled errors are the planned primary metrics.
- Weather is one NYC point used as a coarse proxy for statewide conditions.
