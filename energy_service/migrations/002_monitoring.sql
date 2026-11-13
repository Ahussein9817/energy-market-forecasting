CREATE TABLE monitor_snapshots (
    id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL, checks_json TEXT NOT NULL
);
CREATE INDEX idx_monitor_created ON monitor_snapshots(created_at);
