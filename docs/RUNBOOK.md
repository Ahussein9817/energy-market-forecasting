# Runbook: day-ahead forecast service (local)

Scope and claim level: this runs on one machine (Python or Docker Compose), serves NYISO N.Y.C. day-ahead price forecasts, and monitors itself. It is a **production-style application running locally**, not a hosted or operationally used system: no real users, no uptime history, no TLS, no authentication on read endpoints (they are bound to localhost). Do not describe it as production.

## 1. Run it

**Python**
```bash
make setup
python -m energy_service db-init
python -m energy_service backfill --from 2019-01 --to <last completed month>     # network: NYISO monthly zips (reuses zips in data/raw)
python -m energy_service weather-backfill --start 2019-01-01 --end <yesterday>   # Open-Meteo archive + 48 h-lead archived forecasts
python -m energy_service train --feature-set OD --activate                        # activates only if the backtest evidence qualifies it
python -m energy_service run-daily                                                # one pass of the daily flow
python -m energy_service serve        # API   http://127.0.0.1:8000  (docs at /docs)
streamlit run dashboard/app.py        # UI    http://127.0.0.1:8501
python -m energy_service schedule     # long-running scheduler (or use cron / `make run-daily`)
```

**Docker Compose** (services: `init`, `api`, `dashboard`, `scheduler`; state on the `energy-data` volume; ports bound to 127.0.0.1)
```bash
cp .env.example .env                                  # optional; set ENERGY_API_KEY to enable the admin endpoint
docker compose up -d --build --wait                   # waits for health checks
docker compose run --rm api python -m energy_service demo-seed        # SYNTHETIC data, for a look around
# real history instead of demo data: run backfill / weather-backfill / train as above via `docker compose run --rm api ...`
python scripts/verify_deployment.py --api-url http://localhost:8000 --dashboard-url http://localhost:8501
docker compose down            # keeps the volume;  `down -v` deletes the data
```
The Dockerfile and compose file were written on a machine without Docker. The CI `docker` job builds the image, starts the stack, waits for every health check and runs `verify_deployment.py` (first green run: 2026-11-13); that is the evidence they work. They have not been run on the author's own machine.

## 2. What runs when

The scheduler (`ENERGY_SCHEDULE_ET`, default `05:00,09:45` America/New_York, wall-clock so daylight saving is handled) runs the daily flow: **ingest -> live weather snapshot -> forecast for tomorrow -> evaluate -> monitor**.
- 05:00 ET: the nominal forecast origin (the day-ahead market is assumed to close then; see `docs/operational-timing.md`).
- 09:45 ET: NYISO has published tomorrow's prices (~09:32 ET); the previous forecast is scored. The second run does not create a new forecast (idempotent).
- On start the scheduler **catches up** a run missed while it was down (`ENERGY_SCHEDULER_CATCHUP=false` disables this).

## 3. Is it healthy?

| Where | What it tells you |
|---|---|
| `GET /health` | process is up (no dependencies) |
| `GET /ready` | 200 `ready`/`degraded`, **503 `not_ready`** with reasons when prices are missing or stale |
| `GET /monitoring/summary`, dashboard "Data health", `python -m energy_service monitor` | all checks below |
| `python scripts/verify_deployment.py ...` | 13 end-to-end checks; exits non-zero on any failure |
| `docker compose ps` | container health (API `/health`, dashboard `/_stcore/health`, scheduler heartbeat < 120 s old) |

### Monitoring checks: meaning and action

| Check | Fails/warns when | Do this |
|---|---|---|
| `prices_fresh` | latest price day is before today (ET) or none | Run `ingest`; check NYISO availability and `events` for `fetch_failed`/`file_rejected`. Until fixed, `/ready` is 503 and no forecast can be made |
| `load_fresh` | actual load older than yesterday after 01:30 ET | Usually NYISO lag; the model falls back to the baseline automatically |
| `ingest_recent` | no successful ingest in 26 h, or last run partial | Look at the `job_runs` table and the JSON logs for the failing source |
| `weather_snapshot_recent` | no live snapshot in 26 h | The model cannot serve (baseline is served); check Open-Meteo reachability |
| `data_quality_48h` | warning/error flags in 48 h (missing hours, rejected or out-of-range files); flags for days since recovered are ignored | Inspect the "Data-quality flags" table; a rejected file never changes stored data |
| `forecast_available` | no forecast for tomorrow after 06:00 ET | `python -m energy_service forecast` (or the admin endpoint); then find out why the scheduled run did not |
| `forecast_fallbacks` | more than half of recent forecasts used the baseline | The reasons are listed (e.g. `weather_missing`); fix the cause, do not just retrain |
| `model_integrity` | model file hash mismatch/unreadable, or no active model | Re-run `train --activate`; a tampered file is never loaded |
| `performance_vs_baseline` | served MAE > 1.0x (warn) / 1.10x (fail) the baseline's over the last 28 evaluated days; judged only with >= 10 days | Consider deactivating the model (baseline then serves); see "Model changes" |
| `error_spike` | latest day's MAE more than 3 sigma above the prior mean | Look at that day in "Investigate a day" |

Alerts: set `ENERGY_ALERT_WEBHOOK` to receive a message when a check enters or leaves `fail` (once per change, not repeated). Every status change is also stored as an event.

**Not monitored on purpose:** price-level drift. A PSI check was tried on the real data and rejected: a 28-day window sits in one price regime, so PSI against any multi-year reference was about 1.3, even season-matched. Performance against the baseline is the signal. Late forecasts (made after the prices were public) and replays are excluded from performance statistics.

## 4. Failure and restart behaviour

- Containers use `restart: unless-stopped`; the database and models are on a volume, so a restart loses nothing.
- A crashing scheduled run is logged and the loop continues; a missed run is caught up on the next start.
- Ingestion is idempotent; re-running it after a failure is always safe. A malformed file is rejected without touching stored data.
- If the model cannot be trusted for any reason the **baseline is served and the reason stored**. If even the baseline cannot be built (price of today missing) no forecast is written, the job fails, `run-daily` exits 2 and `/ready` goes 503.

## 5. Backup and restore (tested)

```bash
python -m energy_service backup --out backups/energy-$(date +%F).sqlite      # online-safe, verified, writes a .json checksum
python -m energy_service restore --from backups/energy-2026-11-13.sqlite      # refuses to overwrite an existing database
python -m energy_service restore --from <file> --force                        # replaces it; the old file is kept as <db>.before-restore
```
A corrupted backup or a checksum mismatch is refused. Models are on disk under `ENERGY_MODEL_DIR` (not in the database file); retrain or copy that directory too.

## 6. Secrets and exposure

- Nothing secret is committed; `.env` is git-ignored (`.env.example` is the template); CI runs `scripts/check_secrets.py` and `pip-audit`.
- The admin endpoint (`POST /admin/forecasts/generate`) is **disabled unless `ENERGY_API_KEY` is set**; it needs the `X-API-Key` header (constant-time comparison), is limited to 6 calls per hour, and only accepts today or yesterday as origin. Rotate the key by changing `.env` and restarting the API. Keys are redacted from logs.
- Ports are bound to 127.0.0.1. Putting this on a network needs TLS and authentication in front of it: not provided.
- Containers run as a non-root user with all capabilities dropped and `no-new-privileges`.

## 7. Model changes

Models are trained offline (`train`), registered with a SHA-256 and the backtest evidence, and activated only if eligible (`results/operational_decision.json`, produced by `scripts/evaluate_operational.py` under a rule fixed in advance). There is no automatic retraining. To retrain: `train --feature-set OD --activate`; to stop serving a model: set `active=0` in `models` (the baseline then serves). If the code's feature definitions change, bump `FEATURE_VERSION` so old models are not served.

## 8. Costs and data terms

Local: $0, no keys. Data comes from free public sources: NYISO (terms not reviewed) and Open-Meteo (CC BY 4.0, non-commercial free tier; attribution is in the README and data docs). A hosted deployment would add recurring cost and has not been evaluated or approved.

## 9. Known limits

One zone, one weather point, GFS only; serving evidence rests on a 48-hour-lead archived forecast standing in for the live one; the 05:00 ET market close is from a 2013 NYISO document; a single writer on SQLite; no hosted deployment, no TLS, no read authentication; Docker files are unverified until CI runs; very short operating history, so no live performance claim.
