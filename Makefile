PY ?= python
export ENERGY_DB_PATH ?= data/energy.sqlite

.PHONY: setup test lint regenerate db-init backfill ingest forecast evaluate status monitor run-daily schedule api dashboard backup docker-up docker-down verify

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

monitor:          ## health checks; stores a snapshot and emits events/alerts for changes
	$(PY) -m energy_service monitor

run-daily:        ## what a scheduler runs once per morning: ingest, weather snapshot, forecast, evaluate, monitor
	$(PY) -m energy_service run-daily

api:              ## read-only API on http://127.0.0.1:8000 (docs at /docs)
	$(PY) -m energy_service serve

dashboard:        ## http://127.0.0.1:8501 (read-only view of the database)
	streamlit run dashboard/app.py

schedule:         ## long-running scheduler (05:00 and 09:45 ET by default; catches up after downtime)
	$(PY) -m energy_service schedule

backup:           ## BACKUP_FILE=backups/energy.sqlite make backup
	$(PY) -m energy_service backup --out $(BACKUP_FILE)

docker-up:        ## local containers (api, dashboard, scheduler); see docs/RUNBOOK.md
	docker compose up -d --build --wait

docker-down:
	docker compose down

verify:           ## smoke-test a running service
	$(PY) scripts/verify_deployment.py --api-url http://localhost:8000
