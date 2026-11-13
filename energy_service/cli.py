"""Command line entry point: python -m energy_service <command>. One scheduled command: `run-daily`."""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from energy_service import db, evaluate, forecast, registry, status, train
from energy_service.config import Settings
from energy_service.dataset import DataIncomplete
from energy_service.http import Fetched, get_bytes
from energy_service.ingest import nyiso, weather

ET = ZoneInfo("America/New_York")


def make_fetcher(s: Settings):
    session = requests.Session()
    headers = {"User-Agent": s.user_agent}

    def fetch(url: str) -> Fetched:
        return get_bytes(url, session=session, headers=headers, timeout=s.timeout, retries=s.retries, backoff=s.backoff)
    return fetch


class CachedFetcher:
    """Serve NYISO monthly zips from a local directory (the research download cache) when present."""
    PAT = re.compile(r"/(damlbmp|palIntegrated)/(\d{6})01\1(?:_zone)?_csv\.zip$")

    def __init__(self, inner, zip_dir: str):
        self.inner, self.dir = inner, Path(zip_dir)

    def __call__(self, url: str) -> Fetched:
        m = self.PAT.search(url)
        if m:
            name = ("damlbmp_zone" if m.group(1) == "damlbmp" else "palIntegrated") + f"_{m.group(2)}.zip"
            p = self.dir / name
            if p.exists() and p.stat().st_size > 0:
                return Fetched(p.read_bytes(), None, url)
        return self.inner(url)


def _now() -> datetime:
    return db.utcnow()


def _today_et(now: datetime) -> date:
    return now.astimezone(ET).date()


def cmd_run_daily(conn, s: Settings, args) -> int:
    """ingest -> weather snapshot -> forecast for tomorrow -> evaluate. Each step is isolated; the exit code
    is non-zero if no forecast could be produced."""
    now, fetch = _now(), make_fetcher(s)
    summary: dict = {}
    summary["ingest"] = nyiso.run_ingest(conn, s, now=now, fetch=fetch, lookback_days=args.lookback_days)
    sid = weather.snapshot_live_forecast(conn, s, now=now, fetch=fetch)
    summary["weather_snapshot"] = sid
    code = 0
    try:
        r = forecast.generate_forecast(conn, s, origin_date=_today_et(now), now=now, source="scheduled")
        summary["forecast"] = {"target_date": r.target_date, "served": r.served_kind, "fallback_reason": r.fallback_reason,
                               "late": r.late, "already_existed": r.existing}
    except forecast.ForecastUnavailable as e:
        summary["forecast"] = {"error": str(e)}
        code = 2
    summary["evaluate"] = evaluate.evaluate_pending(conn, s, now=now)
    print(json.dumps(summary, indent=2, default=str))
    return code


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="energy_service")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("db-init", help="create or migrate the SQLite database")
    b = sub.add_parser("backfill", help="load completed months from NYISO monthly zips")
    b.add_argument("--from", dest="start", required=True)
    b.add_argument("--to", dest="end", required=True)
    b.add_argument("--no-zip-cache", action="store_true", help="do not reuse zips from ENERGY_ZIP_CACHE_DIR")
    wb = sub.add_parser("weather-backfill", help="observed temperature and 2-day-lead archived forecasts for training")
    wb.add_argument("--start", required=True)
    wb.add_argument("--end", required=True)
    i = sub.add_parser("ingest", help="incremental ingestion of the recent window")
    i.add_argument("--lookback-days", type=int, default=3)
    sub.add_parser("weather-snapshot", help="store the current live forecast")
    t = sub.add_parser("train", help="train and register a model (offline)")
    t.add_argument("--feature-set", default="OD", choices=["OD", "OB"])
    t.add_argument("--origin-date")
    t.add_argument("--activate", action="store_true", help="activate if the backtest evidence makes it eligible")
    f = sub.add_parser("forecast", help="generate the forecast for origin-date + 1")
    f.add_argument("--origin-date")
    f.add_argument("--force", action="store_true")
    f.add_argument("--source", default="manual")
    sub.add_parser("evaluate", help="score forecasts whose actual prices are published")
    sub.add_parser("status", help="readiness and recent jobs")
    d = sub.add_parser("run-daily", help="ingest, snapshot weather, forecast, evaluate (the scheduled job)")
    d.add_argument("--lookback-days", type=int, default=3)
    sv = sub.add_parser("serve", help="run the read-only API")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    s = Settings.from_env()

    if args.cmd == "serve":
        import logging

        import uvicorn

        from energy_service.api import create_app
        logging.basicConfig(level=logging.INFO, format="%(message)s")   # one JSON object per request on stdout/stderr
        uvicorn.run(create_app(s), host=args.host, port=args.port, log_config=None)
        return 0
    conn = db.connect(s.db_path)
    try:
        if args.cmd == "db-init":
            print(json.dumps({"applied": db.migrate(conn), "schema_version": db.schema_version(conn), "db": s.db_path}))
            return 0
        db.migrate(conn)
        now = _now()
        if args.cmd == "backfill":
            fetch = make_fetcher(s)
            if not args.no_zip_cache:
                fetch = CachedFetcher(fetch, s.zip_cache_dir)
            print(json.dumps(nyiso.run_backfill(conn, s, start=args.start, end=args.end, fetch=fetch, now=now)))
        elif args.cmd == "weather-backfill":
            fetch = make_fetcher(s)
            a = weather.backfill_archive(conn, s, start=args.start, end=args.end, now=now, fetch=fetch)
            b2 = weather.backfill_previous_runs(conn, s, start=args.start, end=args.end, lead_days=2, now=now, fetch=fetch)
            print(json.dumps({"archive_snapshot": a, "previous_runs_snapshots": b2}))
        elif args.cmd == "ingest":
            print(json.dumps(nyiso.run_ingest(conn, s, now=now, fetch=make_fetcher(s), lookback_days=args.lookback_days)))
        elif args.cmd == "weather-snapshot":
            sid = weather.snapshot_live_forecast(conn, s, now=now, fetch=make_fetcher(s))
            print(json.dumps({"snapshot_id": sid}))
            return 0 if sid else 2
        elif args.cmd == "train":
            origin = date.fromisoformat(args.origin_date) if args.origin_date else _today_et(now)
            try:
                mid = train.train_model(conn, s, feature_set=args.feature_set, origin_date=origin, now=now)
            except DataIncomplete as e:
                print(json.dumps({"error": str(e)}))
                return 2
            out = {"model_id": mid}
            if args.activate:
                try:
                    registry.activate(conn, mid)
                    out["active"] = True
                except registry.NotEligible as e:
                    out.update(active=False, reason=str(e))
            print(json.dumps(out))
        elif args.cmd == "forecast":
            origin = date.fromisoformat(args.origin_date) if args.origin_date else _today_et(now)
            try:
                r = forecast.generate_forecast(conn, s, origin_date=origin, now=now, force=args.force, source=args.source)
            except forecast.ForecastUnavailable as e:
                print(json.dumps({"error": str(e)}))
                return 2
            print(json.dumps(r.__dict__))
        elif args.cmd == "evaluate":
            print(json.dumps(evaluate.evaluate_pending(conn, s, now=now)))
        elif args.cmd == "status":
            print(json.dumps({"readiness": status.readiness(conn, s, now), "recent_jobs": status.recent_jobs(conn)}, indent=2))
        elif args.cmd == "run-daily":
            return cmd_run_daily(conn, s, args)
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
