-- Operational store. Timestamps are ISO-8601 UTC text ("YYYY-MM-DDTHH:MM:SSZ"); dates are "YYYY-MM-DD".
CREATE TABLE raw_files (               -- provenance of every fetched file (one row per distinct content)
    id INTEGER PRIMARY KEY,
    dataset TEXT NOT NULL,             -- dam_price | actual_load
    url TEXT NOT NULL,
    granularity TEXT NOT NULL,         -- daily | monthly
    delivery_date TEXT,                -- daily files only
    last_modified TEXT,                -- server Last-Modified (UTC), if provided
    sha256 TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    fetched_at TEXT NOT NULL,
    status TEXT NOT NULL,              -- ingested | rejected
    detail TEXT,
    UNIQUE (url, sha256)
);
CREATE TABLE prices_dam (
    zone TEXT NOT NULL, ts_utc TEXT NOT NULL, lbmp REAL NOT NULL,
    raw_file_id INTEGER NOT NULL REFERENCES raw_files(id),
    PRIMARY KEY (zone, ts_utc)
);
CREATE TABLE load_actual (
    zone TEXT NOT NULL, ts_utc TEXT NOT NULL, load_mw REAL NOT NULL,
    raw_file_id INTEGER NOT NULL REFERENCES raw_files(id),
    PRIMARY KEY (zone, ts_utc)
);
CREATE TABLE weather_snapshots (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,                -- live_forecast | archive_actual | previous_runs
    model TEXT, lead_days INTEGER,
    retrieved_at TEXT NOT NULL, lat REAL, lon REAL, n_hours INTEGER NOT NULL, sha256 TEXT NOT NULL
);
CREATE TABLE weather_hours (
    snapshot_id INTEGER NOT NULL REFERENCES weather_snapshots(id),
    ts_utc TEXT NOT NULL, temp_c REAL,  -- NULL = missing upstream (never filled)
    PRIMARY KEY (snapshot_id, ts_utc)
);
CREATE TABLE quality_flags (
    id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, dataset TEXT NOT NULL, delivery_date TEXT,
    code TEXT NOT NULL, severity TEXT NOT NULL, detail TEXT, raw_file_id INTEGER
);
CREATE TABLE models (
    model_id TEXT PRIMARY KEY, kind TEXT NOT NULL, feature_set TEXT NOT NULL, feature_version TEXT NOT NULL,
    path TEXT NOT NULL, sha256 TEXT NOT NULL, trained_at TEXT NOT NULL, train_start TEXT, train_end TEXT,
    n_rows INTEGER, params_json TEXT, evidence_json TEXT,
    eligible INTEGER NOT NULL DEFAULT 0,   -- backtest evidence met the serving rule
    active INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE forecasts (
    id INTEGER PRIMARY KEY,
    origin_date TEXT NOT NULL, target_date TEXT NOT NULL, version INTEGER NOT NULL,
    origin_ts TEXT NOT NULL, generated_at TEXT NOT NULL,
    source TEXT NOT NULL,              -- scheduled | manual | replay
    served_kind TEXT NOT NULL,         -- model | baseline
    served_model_id TEXT, fallback_reason TEXT,
    late INTEGER NOT NULL,             -- generated after the day-ahead prices were public
    input_json TEXT NOT NULL,
    UNIQUE (target_date, version)
);
CREATE TABLE forecast_hours (
    forecast_id INTEGER NOT NULL REFERENCES forecasts(id), hour INTEGER NOT NULL,
    baseline REAL NOT NULL, model REAL, served REAL NOT NULL,
    PRIMARY KEY (forecast_id, hour)
);
CREATE TABLE evaluations (
    forecast_id INTEGER PRIMARY KEY REFERENCES forecasts(id), evaluated_at TEXT NOT NULL, target_date TEXT NOT NULL,
    mae_served REAL NOT NULL, mae_baseline REAL NOT NULL, mae_model REAL, bias_served REAL NOT NULL
);
CREATE TABLE evaluation_hours (
    forecast_id INTEGER NOT NULL REFERENCES forecasts(id), hour INTEGER NOT NULL, actual REAL NOT NULL,
    PRIMARY KEY (forecast_id, hour)
);
CREATE TABLE job_runs (
    id INTEGER PRIMARY KEY, job TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT,
    status TEXT NOT NULL,              -- running | success | partial | failed
    details_json TEXT, error TEXT
);
CREATE TABLE events (
    id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, level TEXT NOT NULL, code TEXT NOT NULL,
    message TEXT NOT NULL, detail_json TEXT
);
CREATE INDEX idx_flags_created ON quality_flags(created_at);
CREATE INDEX idx_forecasts_target ON forecasts(target_date);
