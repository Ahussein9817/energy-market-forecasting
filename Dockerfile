# Operational service image: API, scheduler and dashboard share it (different commands in compose.yaml).
# NOTE: written without Docker available on the authoring machine; it is built and smoke-tested by the `docker` CI job.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements-service.txt ./
RUN pip install -r requirements-service.txt

COPY pyproject.toml ./
COPY src ./src
COPY energy_service ./energy_service
COPY dashboard ./dashboard
COPY .streamlit ./.streamlit
COPY scripts/verify_deployment.py ./scripts/verify_deployment.py
# the registry reads the operational backtest evidence (serving eligibility) from here
COPY results/operational_decision.json results/operational_metrics.json results/operational_run_metadata.json ./results/
RUN pip install --no-deps . httpx==0.28.1

# non-root user; all state lives on the /data volume
RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown app:app /data
USER app
ENV ENERGY_DB_PATH=/data/energy.sqlite \
    ENERGY_MODEL_DIR=/data/models \
    ENERGY_CACHE_DIR=/data/cache \
    ENERGY_ZIP_CACHE_DIR=/data/zips \
    ENERGY_RESULTS_DIR=/app/results
VOLUME /data
EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["python", "-m", "energy_service", "serve", "--host", "0.0.0.0", "--port", "8000"]
