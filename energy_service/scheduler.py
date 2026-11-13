"""A small scheduler loop: run `run_daily` at fixed America/New_York times. Chosen over an orchestrator because there
are two small daily steps. On start it catches up a run missed while the process was down; every failure is isolated
and logged so the loop keeps running; a heartbeat file proves liveness to the container health check."""
from __future__ import annotations

import logging
import time as _time
from collections.abc import Callable
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
log = logging.getLogger("energy.scheduler")


def parse_times(spec: str) -> list[time]:
    out = sorted({time(int(p.split(":")[0]), int(p.split(":")[1])) for p in spec.split(",") if p.strip()})
    if not out:
        raise ValueError("no schedule times given")
    return out


def _at(d, t: time) -> datetime:
    return datetime.combine(d, t, tzinfo=ET)


def next_run(now: datetime, times: list[time]) -> datetime:
    """The first scheduled local time strictly after `now` (DST-safe: wall-clock times in America/New_York)."""
    local = now.astimezone(ET)
    for d in (local.date(), local.date() + timedelta(days=1)):
        for t in times:
            c = _at(d, t)
            if c > local:
                return c
    raise AssertionError("unreachable")


def last_scheduled(now: datetime, times: list[time]) -> datetime:
    local = now.astimezone(ET)
    for d in (local.date(), local.date() - timedelta(days=1)):
        for t in sorted(times, reverse=True):
            c = _at(d, t)
            if c <= local:
                return c
    raise AssertionError("unreachable")


def missed_run(now: datetime, times: list[time], last_success: datetime | None) -> bool:
    """True if the most recent scheduled time has passed without a successful run since it."""
    return last_success is None or last_success < last_scheduled(now, times)


def heartbeat(path: str | None, now: datetime) -> None:
    if path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(now.isoformat())


def run_forever(job: Callable[[datetime], bool], times: list[time], *, clock: Callable[[], datetime],
                last_success: datetime | None, sleep: Callable[[float], None] = _time.sleep, heartbeat_path: str | None = None,
                max_runs: int | None = None, tick: float = 30.0) -> int:
    """`job(now) -> bool` returns True on success. Returns the number of runs made (for tests)."""
    runs = 0
    if missed_run(clock(), times, last_success):
        log.info("catching up a missed run", extra={"fields": {"last_success": str(last_success)}})
        ok = _safe(job, clock())
        last_success = clock() if ok else last_success
        runs += 1
    while max_runs is None or runs < max_runs:
        due = next_run(clock(), times)
        while clock() < due:
            heartbeat(heartbeat_path, clock())
            sleep(min(tick, max(0.0, (due - clock()).total_seconds())))
        ok = _safe(job, clock())
        runs += 1
        heartbeat(heartbeat_path, clock())
        last_success = clock() if ok else last_success
    return runs


def _safe(job, now) -> bool:
    try:
        return bool(job(now))
    except Exception:
        log.exception("scheduled run crashed; the loop continues")
        return False
