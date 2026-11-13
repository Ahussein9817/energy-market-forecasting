"""SQLite access and versioned migrations (plain SQL files, applied once, in order)."""
from __future__ import annotations

import logging
import sqlite3
import time as _time
from datetime import UTC, datetime
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def iso(ts) -> str:
    """UTC ISO-8601 text with trailing Z (sortable). Accepts datetime or pandas Timestamp."""
    t = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
    if t.tzinfo is None:
        raise ValueError("naive timestamp: refusing to store without a timezone")
    return t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def utcnow() -> datetime:
    return datetime.now(UTC)


def connect(path: str, read_only: bool = False) -> sqlite3.Connection:
    if not read_only and path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    uri = f"file:{path}?mode=ro" if read_only else f"file:{path}"
    conn = sqlite3.connect(uri, uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    if not read_only:
        conn.execute("PRAGMA journal_mode=WAL")
    return conn


def migrate(conn: sqlite3.Connection) -> list[int]:
    """Apply pending migrations; returns the versions applied now."""
    conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
    done = {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}
    applied = []
    for f in sorted(MIGRATIONS_DIR.glob("*.sql")):
        v = int(f.name.split("_", 1)[0])
        if v in done:
            continue
        conn.executescript(f.read_text())
        conn.execute("INSERT INTO schema_migrations VALUES (?, ?)", (v, iso(utcnow())))
        conn.commit()
        applied.append(v)
    return applied


def schema_version(conn: sqlite3.Connection) -> int | None:
    try:
        return conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
    except sqlite3.Error:
        return None


def log_event(conn, level: str, code: str, message: str, detail: dict | None = None, now: datetime | None = None):
    import json
    conn.execute("INSERT INTO events (created_at, level, code, message, detail_json) VALUES (?,?,?,?,?)",
                 (iso(now or utcnow()), level, code, message, json.dumps(detail) if detail else None))
    conn.commit()


class JobRun:
    """Context manager recording a job in job_runs (success / failed + error text)."""

    def __init__(self, conn, job: str, now: datetime | None = None):
        self.conn, self.job = conn, job
        self.now = now or utcnow()
        self.details: dict = {}
        self.status = "success"

    def __enter__(self):
        self._t0 = _time.perf_counter()
        cur = self.conn.execute("INSERT INTO job_runs (job, started_at, status) VALUES (?,?, 'running')",
                                (self.job, iso(self.now)))
        self.conn.commit()
        self.id = cur.lastrowid
        return self

    def __exit__(self, exc_type, exc, tb):
        import json
        status = "failed" if exc_type else self.status
        self.conn.execute("UPDATE job_runs SET finished_at=?, status=?, details_json=?, error=? WHERE id=?",
                          (iso(utcnow()), status, json.dumps(self.details, default=str),
                           f"{exc_type.__name__}: {exc}" if exc_type else None, self.id))
        self.conn.commit()
        logging.getLogger("energy.job").info("job finished", extra={"fields": {
            "job": self.job, "status": status, "duration_s": round(_time.perf_counter() - self._t0, 2),
            "error": f"{exc_type.__name__}: {exc}" if exc_type else None}})
        return False
