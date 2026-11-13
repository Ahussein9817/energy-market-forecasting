"""The scheduled daily flow, shared by the CLI (`run-daily`) and the scheduler:
ingest -> live weather snapshot -> forecast for tomorrow -> evaluate -> monitor. Each step is isolated so one failing
step does not stop the rest; the result says what happened and whether a forecast exists."""
from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from energy_service import evaluate, forecast, monitor
from energy_service.config import Settings
from energy_service.db import JobRun
from energy_service.ingest import nyiso, weather

ET = ZoneInfo("America/New_York")
log = logging.getLogger("energy.daily")


def run_daily(conn, s: Settings, *, now: datetime, fetch, lookback_days: int = 3, post=None) -> tuple[dict, int]:
    """Returns (summary, exit_code); exit_code is 2 when no forecast could be produced."""
    summary: dict = {}
    code = 0
    with JobRun(conn, "run_daily", now) as job:
        for name, step in (
            ("ingest", lambda: nyiso.run_ingest(conn, s, now=now, fetch=fetch, lookback_days=lookback_days)),
            ("weather_snapshot", lambda: weather.snapshot_live_forecast(conn, s, now=now, fetch=fetch)),
        ):
            try:
                summary[name] = step()
            except Exception as e:  # a step failing must not stop the flow
                log.exception("step failed", extra={"fields": {"step": name}})
                summary[name] = {"error": f"{type(e).__name__}: {e}"}
                job.status = "partial"
        try:
            r = forecast.generate_forecast(conn, s, origin_date=now.astimezone(ET).date(), now=now, source="scheduled")
            summary["forecast"] = {"target_date": r.target_date, "served": r.served_kind, "fallback_reason": r.fallback_reason,
                                   "late": r.late, "already_existed": r.existing}
        except forecast.ForecastUnavailable as e:
            summary["forecast"] = {"error": str(e)}
            code = 2
            job.status = "partial"
        except Exception as e:
            log.exception("forecast step crashed")
            summary["forecast"] = {"error": f"{type(e).__name__}: {e}"}
            code = 2
            job.status = "partial"
        for name, step in (("evaluate", lambda: evaluate.evaluate_pending(conn, s, now=now)),
                           ("monitor", lambda: monitor.run_monitor(conn, s, now) if post is None else monitor.run_monitor(conn, s, now, post))):
            try:
                out = step()
                summary[name] = out if name == "evaluate" else {"status": out["status"], "transitions": len(out["transitions"]),
                                                                "alerts_sent": out["alerts_sent"]}
            except Exception as e:
                log.exception("step failed", extra={"fields": {"step": name}})
                summary[name] = {"error": f"{type(e).__name__}: {e}"}
                job.status = "partial"
        if code:
            job.status = "failed"
        job.details = summary
    return summary, code
