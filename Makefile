PY ?= python
export ENERGY_DB_PATH ?= data/energy.sqlite

.PHONY: setup test lint regenerate db-init backfill ingest forecast evaluate status run-daily api

setup:            ## install pinned dependencies and the package (editable)
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt
	$(PY) -m pip install -e . --no-deps

lint:
	ruff check .

test:             ## offline: no network, no API keys
	pytest -q

regenerate:       ## metrics must regenerate exactly from the committed predictions
	$(PY) scripts/evaluate_predictions.py
	$(PY) scripts/evaluate_forecast_weather.py
	$(PY) scripts/evaluate_operational.py

db-init:
	$(PY) -m energy_service db-init

backfill:         ## FROM/TO as YYYY-MM; reuses zips in data/raw when present (network for the rest)
	$(PY) -m energy_service backfill --from $(FROM) --to $(TO)

ingest:
	$(PY) -m energy_service ingest

forecast:
	$(PY) -m energy_service forecast

evaluate:
	$(PY) -m energy_service evaluate

status:
	$(PY) -m energy_service status

run-daily:        ## what a scheduler runs once per morning: ingest, weather snapshot, forecast, evaluate
	$(PY) -m energy_service run-daily

api:              ## read-only API on http://127.0.0.1:8000 (docs at /docs)
	$(PY) -m energy_service serve
