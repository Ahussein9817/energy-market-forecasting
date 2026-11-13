"""Environment-based configuration. No secrets are stored in the repository."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    db_path: str = str(ROOT / "data" / "energy.sqlite")
    model_dir: str = str(ROOT / "models")
    cache_dir: str = str(ROOT / "data" / "external")      # Open-Meteo previous-runs cache
    zip_cache_dir: str = str(ROOT / "data" / "raw")       # NYISO monthly zips (backfill reuse)
    results_dir: str = str(ROOT / "results")
    zone: str = "N.Y.C."
    lat: float = 40.7128
    lon: float = -74.0060
    user_agent: str = "energy-market-forecasting (research demo; github.com/Ahussein9817)"
    timeout: float = 30.0
    retries: int = 3
    backoff: float = 2.0
    price_min: float = -1000.0     # $/MWh sanity bounds; values outside are rejected and flagged
    price_max: float = 10000.0
    load_min: float = 0.0          # MW
    load_max: float = 50000.0
    dam_publish_et: str = "09:35"  # a day-ahead file for D+1 is expected on day D from this ET clock time
    load_publish_et: str = "01:30" # actual load of day D expected on D+1 from this ET clock time
    weather_max_age_hours: float = 12.0   # a live weather snapshot older than this is not used
    late_after_et: str = "09:30"   # forecasts generated after this ET time on day D are flagged late
    api_key: str | None = None     # admin operations; unset disables them
    # monitoring thresholds
    job_max_age_hours: float = 26.0        # a successful ingest must be newer than this
    perf_min_days: int = 10                # evaluated days needed before performance is judged
    perf_warn_ratio: float = 1.0           # served MAE / baseline MAE above this warns
    perf_fail_ratio: float = 1.10
    fallback_warn_rate: float = 0.5        # share of recent forecasts that fell back to the baseline
    spike_sigma: float = 3.0               # latest-day MAE this many sigmas above the prior mean warns
    alert_webhook: str | None = None       # optional; unset disables alert delivery
    schedule_et: str = "05:00,09:45"       # run-daily local times (America/New_York) for the scheduler
    log_level: str = "INFO"
    scheduler_catchup: bool = True         # on start, run a daily flow missed while the process was down
    admin_rate_limit_per_hour: int = 6

    @classmethod
    def from_env(cls, env: dict | None = None) -> Settings:
        e = os.environ if env is None else env
        d = cls()
        kw = {}
        for field, var, cast in (("db_path", "ENERGY_DB_PATH", str), ("model_dir", "ENERGY_MODEL_DIR", str),
                                 ("cache_dir", "ENERGY_CACHE_DIR", str), ("zip_cache_dir", "ENERGY_ZIP_CACHE_DIR", str),
                                 ("results_dir", "ENERGY_RESULTS_DIR", str), ("zone", "ENERGY_ZONE", str),
                                 ("user_agent", "ENERGY_USER_AGENT", str), ("timeout", "ENERGY_HTTP_TIMEOUT", float),
                                 ("retries", "ENERGY_HTTP_RETRIES", int), ("backoff", "ENERGY_HTTP_BACKOFF", float),
                                 ("weather_max_age_hours", "ENERGY_WEATHER_MAX_AGE_HOURS", float),
                                 ("api_key", "ENERGY_API_KEY", str), ("alert_webhook", "ENERGY_ALERT_WEBHOOK", str),
                                 ("schedule_et", "ENERGY_SCHEDULE_ET", str), ("log_level", "ENERGY_LOG_LEVEL", str),
                                 ("job_max_age_hours", "ENERGY_JOB_MAX_AGE_HOURS", float),
                                 ("perf_min_days", "ENERGY_PERF_MIN_DAYS", int),
                                 ("admin_rate_limit_per_hour", "ENERGY_ADMIN_RATE_LIMIT", int),
                                 ("scheduler_catchup", "ENERGY_SCHEDULER_CATCHUP",
                                  lambda v: v.strip().lower() in ("1", "true", "yes"))):
            if e.get(var):
                kw[field] = cast(e[var])
        return cls(**{**d.__dict__, **kw})
