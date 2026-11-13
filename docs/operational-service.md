# Operational service (Phase 1)

Scope: a local, tested pipeline from NYISO files to a retrievable, evaluated day-ahead forecast. Research code in `src/` (verified backtests) is reused unchanged; `energy_service/` adds the operational layer. Forecast definition: `docs/operational-timing.md`.

## Components

```
NYISO files ──► ingest (retry, idempotent, provenance) ──► validate ──► SQLite ◄── Open-Meteo live snapshot
                                                                         │
 train (offline) ──► registry (joblib + metadata.json + sha256) ─────────┤
                                                                         ▼
                                  forecast job (baseline always; model only if trusted) ──► forecasts
                                                                         │
                                  evaluate job (when day-ahead prices publish) ──► evaluations
                                                                         ▼
                                              read-only API: /health /ready /models /forecasts[/latest]
```

| Module | Responsibility |
|---|---|
| `ingest/nyiso.py` | Daily files (rolling ~12 days) and monthly zips; current-month zip as fallback; SHA-256 dedupe; upsert; revision tracking |
| `ingest/validate.py` | Schema, hourly timestamps, null/range, duplicates, completeness (23/24/25 local hours) |
| `ingest/weather.py` | Live forecast snapshots (stored with retrieval time), archive and archived-forecast backfill |
| `registry.py` | Persist/load models, hash check, evidence from the operational backtest, activation only if eligible |
| `train.py` | Offline training; refuses with too little history |
| `forecast.py` | Builds the forecast at the origin; fallback policy; provenance; idempotent per target day |
| `evaluate.py` | Scores stored forecasts against actual day-ahead prices |
| `status.py` | Readiness rules shared by the CLI and `/ready` |
| `api.py` | Read-only API, consistent error body, JSON request logs with request ids |
| `cli.py` | `python -m energy_service ...`; `run-daily` is the scheduled entry point |

## Data model (SQLite, `energy_service/migrations/001_init.sql`)

`raw_files` (provenance) -> `prices_dam`, `load_actual`; `weather_snapshots` -> `weather_hours`; `quality_flags`; `models`; `forecasts` -> `forecast_hours`; `evaluations` -> `evaluation_hours`; `job_runs`; `events`. Timestamps are UTC ISO text; hours of a delivery day are local (America/New_York), 0-23, as in the research panels.

SQLite was chosen over PostgreSQL because there is a single writer (the scheduled job), the data is a few hundred MB at most, and the API opens the file read-only. A server database would add an operational dependency with no concurrency requirement.

## Failure behaviour (each case has a test)

| Situation | Behaviour |
|---|---|
| NYISO file not published yet | Not an error; flagged `expected_file_missing` only if it is past its usual publication time and neither the daily file nor the current-month zip has the day |
| Source down / 5xx / timeout | Retried with exponential backoff; job `partial` or `failed`; stored data untouched; event logged |
| Malformed file or wrong columns | Rejected (`raw_files.status='rejected'`), flagged, nothing stored from it |
| Conflicting duplicate rows | File rejected; identical duplicates dropped with an info flag |
| Out-of-range or null values | Row dropped and flagged |
| Missing hour | Rows kept, day flagged `incomplete_day` (a forecast refuses to run if day D is incomplete) |
| File content revised upstream | New `raw_files` version, values updated, `revised_values` flag |
| No active/eligible model, weather missing/stale, load missing, feature mismatch, model file missing or tampered | Baseline served; reason stored on the forecast and logged as an event |
| Price of day D missing | `ForecastUnavailable`; no forecast row; job failed; `run-daily` exits non-zero |
| Prices stale at request time | `/ready` returns 503 with reasons |

## Real-data run (2026-11-13)

Backfill of 2019-01..2026-10 from monthly zips (94 months x 2 datasets, 26 s), current-month ingest, training (3.3 s), a live weather snapshot, one forecast for 2026-11-14 served by the model, and the API queried over HTTP. The run surfaced two issues that the synthetic tests had not: NYISO's daily files roll off after about 12 days (fixed with the current-month-zip fallback), and the server access log was silent without a logging handler (fixed). The generated database and model are local and not committed.

## What this is not

Not deployed, not monitored beyond readiness checks, no dashboard, no scheduler configured, no admin endpoint, and no operating history. Do not describe it as a production system.
