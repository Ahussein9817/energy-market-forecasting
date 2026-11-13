"""Operations: admin endpoint, monitoring endpoint, logging, scheduler, backup/restore, daily flow, deployment verification."""
import io
import json
import logging
import sqlite3
import sys
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from energy_service import backup, daily, db, logging_setup, scheduler
from energy_service.api import create_app
from energy_service.config import Settings
from energy_service.http import FetchError, NotYetPublished
from tests.service_helpers import (
    FakeSite,
    add_live_snapshot,
    build_world,
    populate_days,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import verify_deployment  # noqa: E402

D = date(2026, 11, 12)
MORNING = datetime(2026, 11, 12, 10, 30, tzinfo=UTC)         # 05:30 ET
KEY = "s3cr3t-test-key"


def settings(tmp_path, **kw):
    return Settings(db_path=str(tmp_path / "o.sqlite"), model_dir=str(tmp_path / "m"), results_dir=str(tmp_path / "r"), **kw)


@pytest.fixture
def world(tmp_path):
    s = settings(tmp_path, api_key=KEY)
    conn = db.connect(s.db_path)
    db.migrate(conn)
    build_world(conn, D)
    conn.close()
    return s


def client(s, now=MORNING):
    return TestClient(create_app(s, clock=lambda: now), raise_server_exceptions=False)


# ---- admin endpoint -------------------------------------------------------------------------
def test_admin_is_disabled_without_a_key_and_rejects_bad_keys(tmp_path, world):
    off = settings(tmp_path)                                  # no key configured
    r = client(off).post("/admin/forecasts/generate", json={})
    assert r.status_code == 403 and r.json()["error"]["code"] == "admin_disabled"
    c = client(world)
    assert c.post("/admin/forecasts/generate", json={}).status_code == 401
    assert c.post("/admin/forecasts/generate", json={}, headers={"X-API-Key": "wrong"}).status_code == 401
    conn = db.connect(world.db_path)
    assert conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == 0           # nothing happened
    conn.close()


def test_admin_generates_idempotently_and_force_makes_a_new_version(world):
    c = client(world)
    h = {"X-API-Key": KEY}
    a = c.post("/admin/forecasts/generate", json={}, headers=h)
    assert a.status_code == 200
    j = a.json()
    assert j["target_date"] == "2026-11-13" and j["already_existed"] is False and j["served_kind"] == "baseline" and j["version"] == 1
    assert j["fallback_reason"] == "no_active_model"
    b = c.post("/admin/forecasts/generate", json={}, headers=h).json()
    assert b["already_existed"] is True and b["forecast_id"] == j["forecast_id"]
    f = c.post("/admin/forecasts/generate", json={"force": True}, headers=h).json()
    assert f["version"] == 2
    assert c.get("/forecasts/latest").json()["source"] == "manual"


def test_admin_restricts_the_origin_date_and_rate_limits(world):
    s = Settings(**{**world.__dict__, "admin_rate_limit_per_hour": 2})
    c = client(s)
    h = {"X-API-Key": KEY}
    r = c.post("/admin/forecasts/generate", json={"origin_date": "2026-10-01"}, headers=h)
    assert r.status_code == 422 and r.json()["error"]["code"] == "origin_out_of_range"
    assert c.post("/admin/forecasts/generate", json={}, headers=h).status_code == 200
    assert c.post("/admin/forecasts/generate", json={}, headers=h).status_code == 200
    r3 = c.post("/admin/forecasts/generate", json={}, headers=h)
    assert r3.status_code == 429 and r3.json()["error"]["code"] == "rate_limited"


def test_admin_reports_a_missing_baseline_input_as_409(tmp_path):
    s = settings(tmp_path, api_key=KEY)
    conn = db.connect(s.db_path)
    db.migrate(conn)
    build_world(conn, date(2026, 11, 5))                      # prices end a week before "today"
    conn.close()
    r = client(s).post("/admin/forecasts/generate", json={}, headers={"X-API-Key": KEY})
    assert r.status_code == 409 and r.json()["error"]["code"] == "forecast_unavailable"


def test_the_api_key_never_appears_in_logs(world, caplog):
    out = io.StringIO()
    logging_setup.configure("INFO", stream=out, secrets=(KEY,))
    try:
        client(world).post("/admin/forecasts/generate", json={}, headers={"X-API-Key": KEY})
        client(world).post("/admin/forecasts/generate", json={}, headers={"X-API-Key": "wrong-" + KEY})
    finally:
        logging.getLogger().handlers.clear()
    assert KEY not in out.getvalue() and out.getvalue()


# ---- monitoring endpoint --------------------------------------------------------------------
def test_monitoring_summary_reports_checks_events_and_jobs(world):
    c = client(world)
    j = c.get("/monitoring/summary").json()
    assert j["status"] == "warn"                              # 05:30 ET: forecast not due yet; only "no active model" is off
    names = {x["name"]: x["status"] for x in j["checks"]}
    assert names["prices_fresh"] == "ok" and names["performance_vs_baseline"] == "unknown"
    assert "recent_events" in j and "recent_jobs" in j and j["last_snapshot_at"] is None
    conn = db.connect(world.db_path)
    before = conn.execute("SELECT COUNT(*) FROM monitor_snapshots").fetchone()[0]
    conn.close()
    c.get("/monitoring/summary")
    conn = db.connect(world.db_path)
    assert conn.execute("SELECT COUNT(*) FROM monitor_snapshots").fetchone()[0] == before    # the endpoint writes nothing
    conn.close()


# ---- logging --------------------------------------------------------------------------------
def test_json_formatter_emits_one_json_object_with_fields_and_redaction():
    out = io.StringIO()
    logging_setup.configure("INFO", stream=out, secrets=("topsecret",))
    try:
        logging.getLogger("t").info("hello topsecret", extra={"fields": {"job": "x", "api_key": "abc", "n": 2}})
    finally:
        logging.getLogger().handlers.clear()
    rec = json.loads(out.getvalue())
    assert rec["message"] == "hello ***" and rec["job"] == "x" and rec["api_key"] == "***" and rec["n"] == 2
    assert rec["level"] == "INFO" and rec["ts"].endswith("Z")


def test_jobs_log_a_structured_completion_line(tmp_path, caplog):
    conn = db.connect(str(tmp_path / "j.sqlite"))
    db.migrate(conn)
    with caplog.at_level("INFO", logger="energy.job"):
        with db.JobRun(conn, "demo"):
            pass
        with pytest.raises(RuntimeError), db.JobRun(conn, "boom"):
            raise RuntimeError("bad")
    f = [r.fields for r in caplog.records if hasattr(r, "fields")]
    assert f[0]["job"] == "demo" and f[0]["status"] == "success" and f[1]["status"] == "failed" and "RuntimeError" in f[1]["error"]


# ---- scheduler ------------------------------------------------------------------------------
T = [time(5, 0), time(9, 45)]


def utc(y, m, d, h, mi=0):
    return datetime(y, m, d, h, mi, tzinfo=UTC)


def test_parse_times_and_next_run_across_midnight():
    assert scheduler.parse_times("09:45, 05:00,05:00") == T
    with pytest.raises(ValueError):
        scheduler.parse_times("")
    n = scheduler.next_run(utc(2026, 11, 12, 4, 0), T)                  # 23:00 ET on the 11th
    assert n.astimezone(scheduler.ET).strftime("%Y-%m-%d %H:%M") == "2026-11-12 05:00"
    n2 = scheduler.next_run(utc(2026, 11, 12, 20, 0), T)                # 15:00 ET: next is tomorrow 05:00
    assert n2.astimezone(scheduler.ET).strftime("%Y-%m-%d %H:%M") == "2026-11-13 05:00"
    assert scheduler.next_run(utc(2026, 11, 12, 10, 0), T).astimezone(scheduler.ET).strftime("%H:%M") == "09:45"  # strictly after 05:00


def test_schedule_follows_wall_clock_across_dst():
    """05:00 ET stays 05:00 ET when clocks fall back on 2026-11-01, so the UTC time moves from 09:00 to 10:00."""
    sat = scheduler.next_run(utc(2026, 10, 30, 12, 0), [time(5, 0)])        # 08:00 EDT on Oct 30
    sun = scheduler.next_run(utc(2026, 11, 1, 12, 0), [time(5, 0)])         # 07:00 EST on Nov 1 (after the shift)
    assert sat.astimezone(UTC) == utc(2026, 10, 31, 9, 0)                   # 05:00 EDT
    assert sun.astimezone(UTC) == utc(2026, 11, 2, 10, 0)                   # 05:00 EST
    assert scheduler.next_run(utc(2026, 11, 2, 9, 0), [time(5, 0)]).astimezone(UTC) == utc(2026, 11, 2, 10, 0)   # 04:00 EST -> later today


def test_missed_run_detection():
    now = utc(2026, 11, 12, 16, 0)                                       # 11:00 ET: 09:45 slot has passed
    assert scheduler.last_scheduled(now, T).astimezone(scheduler.ET).strftime("%H:%M") == "09:45"
    assert scheduler.missed_run(now, T, None) is True
    assert scheduler.missed_run(now, T, utc(2026, 11, 12, 14, 50)) is False   # ran at 09:50 ET
    assert scheduler.missed_run(now, T, utc(2026, 11, 12, 11, 0)) is True      # last success was before the 09:45 slot


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now

    def sleep(self, secs):
        self.now += timedelta(seconds=secs)


def test_run_forever_runs_at_the_scheduled_times_and_survives_failures(tmp_path):
    clk = Clock(utc(2026, 11, 12, 8, 0))                                 # 03:00 ET
    calls, fail_next = [], [True]

    def job(now):
        calls.append(now.astimezone(scheduler.ET).strftime("%m-%d %H:%M"))
        if fail_next[0]:
            fail_next[0] = False
            raise RuntimeError("source down")
        return True

    hb = str(tmp_path / "hb")
    n = scheduler.run_forever(job, T, clock=clk, last_success=clk() - timedelta(hours=1), sleep=clk.sleep, heartbeat_path=hb,
                              max_runs=3, tick=600)
    assert n == 3 and calls == ["11-12 05:00", "11-12 09:45", "11-13 05:00"]     # the first run crashed; the loop went on
    assert Path(hb).exists()


def test_run_forever_catches_up_after_downtime():
    clk = Clock(utc(2026, 11, 12, 16, 0))                                # 11:00 ET, process was down since yesterday
    calls = []
    scheduler.run_forever(lambda now: calls.append(now) or True, T, clock=clk, last_success=utc(2026, 11, 11, 15, 0),
                          sleep=clk.sleep, max_runs=1)
    assert len(calls) == 1 and calls[0] == utc(2026, 11, 12, 16, 0)       # ran immediately, not at the next slot


# ---- backup / restore -----------------------------------------------------------------------
def test_backup_restore_roundtrip_preserves_data_and_protects_existing(tmp_path, world):
    out = tmp_path / "bk" / "b.sqlite"
    meta = backup.backup(world.db_path, str(out))
    assert meta["counts"]["prices_dam"] > 3000 and meta["sha256"] and Path(str(out) + ".json").exists()
    with pytest.raises(FileExistsError):
        backup.backup(world.db_path, str(out))                          # never overwrites a backup
    target = tmp_path / "restored.sqlite"
    res = backup.restore(str(out), str(target))
    assert res["counts"] == meta["counts"]
    with pytest.raises(FileExistsError):
        backup.restore(str(out), str(target))                           # existing db protected
    conn = sqlite3.connect(target)
    conn.execute("DELETE FROM prices_dam")
    conn.commit()
    conn.close()
    backup.restore(str(out), str(target), force=True)
    assert backup._counts(str(target))["prices_dam"] == meta["counts"]["prices_dam"]
    assert Path(str(target) + ".before-restore").exists()


def test_corrupted_backup_is_refused(tmp_path, world):
    out = tmp_path / "b.sqlite"
    backup.backup(world.db_path, str(out))
    out.write_bytes(out.read_bytes()[:-500] + b"x" * 500)
    with pytest.raises(ValueError):
        backup.restore(str(out), str(tmp_path / "never.sqlite"))
    assert not (tmp_path / "never.sqlite").exists()


# ---- daily flow -----------------------------------------------------------------------------
def weather_fetch(site):
    from energy_service.http import Fetched

    def f(url):
        if "api.open-meteo.com" in url:
            idx = [(datetime(2026, 11, 12, 0, tzinfo=UTC) + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(96)]
            return Fetched(json.dumps({"hourly": {"time": idx, "temperature_2m": [9.0] * 96}}).encode(), None, url)
        return site(url)
    return f


def test_daily_flow_isolates_failing_steps_and_still_forecasts(tmp_path):
    s = settings(tmp_path)
    conn = db.connect(s.db_path)
    db.migrate(conn)
    build_world(conn, D)
    site = FakeSite()
    populate_days(site, date(2026, 11, 9), date(2026, 11, 12), load_end=date(2026, 11, 11))

    def fetch(url):
        if "api.open-meteo.com" in url:
            raise FetchError("weather down")
        return site(url)
    summary, code = daily.run_daily(conn, s, now=MORNING, fetch=fetch)
    assert code == 0 and summary["forecast"]["served"] == "baseline"              # weather outage -> baseline, flow continues
    assert summary["weather_snapshot"] is None and "status" in summary["monitor"]
    assert conn.execute("SELECT status FROM job_runs WHERE job='run_daily'").fetchone()[0] in ("success", "partial")


def test_daily_flow_with_no_data_fails_with_exit_code_2_but_still_monitors(tmp_path):
    s = settings(tmp_path)
    conn = db.connect(s.db_path)
    db.migrate(conn)
    summary, code = daily.run_daily(conn, s, now=MORNING, fetch=lambda url: (_ for _ in ()).throw(NotYetPublished(url)))
    assert code == 2 and "error" in summary["forecast"] and summary["monitor"]["status"] in ("fail", "warn")
    assert conn.execute("SELECT status FROM job_runs WHERE job='run_daily'").fetchone()[0] == "failed"


def test_daily_flow_scores_yesterdays_forecast_when_prices_arrive(tmp_path):
    s = settings(tmp_path)
    conn = db.connect(s.db_path)
    db.migrate(conn)
    build_world(conn, D)
    add_live_snapshot(conn, date(2026, 11, 13), datetime(2026, 11, 12, 9, 0, tzinfo=UTC))
    from energy_service.forecast import generate_forecast
    generate_forecast(conn, s, origin_date=D, now=MORNING)
    site = FakeSite()
    populate_days(site, date(2026, 11, 9), date(2026, 11, 13), load_end=date(2026, 11, 11))
    summary, code = daily.run_daily(conn, s, now=datetime(2026, 11, 12, 15, 0, tzinfo=UTC), fetch=weather_fetch(site))
    assert code == 0 and summary["evaluate"]["evaluated"] == 1


# ---- deployment verification ----------------------------------------------------------------
def test_verify_deployment_passes_on_a_healthy_service_and_fails_on_a_broken_one(tmp_path):
    s = settings(tmp_path)
    conn = db.connect(s.db_path)
    db.migrate(conn)
    build_world(conn, D)
    add_live_snapshot(conn, date(2026, 11, 13), datetime(2026, 11, 12, 9, 0, tzinfo=UTC))
    from energy_service.forecast import generate_forecast
    generate_forecast(conn, s, origin_date=D, now=MORNING)
    conn.close()
    healthy = TestClient(create_app(s, clock=lambda: MORNING))
    res = verify_deployment.verify(healthy)
    assert all(ok for _, ok, _ in res), [r for r in res if not r[1]]
    assert len(res) >= 12
    stale = TestClient(create_app(s, clock=lambda: datetime(2026, 11, 20, 15, 0, tzinfo=UTC)))     # prices a week old
    bad = {n for n, ok, _ in verify_deployment.verify(stale) if not ok}
    assert "ready (200 ready/degraded)" in bad
    empty = settings(tmp_path / "empty")
    c2 = db.connect(empty.db_path)
    db.migrate(c2)
    c2.close()
    no_fc = verify_deployment.verify(TestClient(create_app(empty, clock=lambda: MORNING)), allow_no_forecast=True, allow_not_ready=True)
    assert all(ok for _, ok, _ in no_fc), [r for r in no_fc if not r[1]]


def test_verify_deployment_reports_unreachable_service():
    class Down(httpx.Client):
        def request(self, *a, **k):
            raise httpx.ConnectError("refused")
    res = verify_deployment.verify(Down(base_url="http://x"))
    assert not all(ok for _, ok, _ in res) and any("reachable" in n for n, ok, _ in res if not ok)


# ---- demo seed and secret scan --------------------------------------------------------------
def test_demo_seed_loads_synthetic_data_and_refuses_real_databases(tmp_path):
    from energy_service import demo
    s = settings(tmp_path)
    conn = db.connect(s.db_path)
    db.migrate(conn)
    out = demo.seed_demo(conn, s, datetime(2026, 11, 12, 11, 0, tzinfo=UTC))
    assert out["served"] == "baseline" and out["note"] == "synthetic data"
    assert conn.execute("SELECT COUNT(*) FROM events WHERE code='demo_seed'").fetchone()[0] == 1
    assert conn.execute("SELECT url FROM raw_files").fetchone()[0] == "synthetic"        # provenance says so
    with pytest.raises(RuntimeError, match="refusing"):
        demo.seed_demo(conn, s, datetime(2026, 11, 12, 11, 0, tzinfo=UTC))                  # never on a populated database
    c = TestClient(create_app(s, clock=lambda: datetime(2026, 11, 12, 15, 0, tzinfo=UTC)))
    assert c.get("/forecasts/latest").json()["target_date"] == "2026-11-13"


def test_secret_scanner_flags_credentials_and_ignores_normal_text():
    import check_secrets
    fake = {"key": "sk-" + "a1" * 15, "gh": "ghp_" + "B2" * 18, "aws": "AKIA" + "Z9" * 8,
            "assign": 'API_KEY = "' + "q" * 24 + '"', "pem": "-----BEGIN " + "RSA PRIVATE KEY-----",
            "hook": "https://hooks.slack.com/" + "services/T000/B000/" + "x" * 20}
    for name, text in fake.items():
        assert check_secrets.scan_text(text, "f.py"), name
    clean = 'api_key = os.environ["ENERGY_API_KEY"]\nKEY = "s3cr3t-test-key"\n# set ENERGY_API_KEY in .env\nPASSWORD_FIELD = None\n'
    assert check_secrets.scan_text(clean) == []
    assert check_secrets.scan_text("x\nx\n" + fake["key"], "f.py")[0].startswith("f.py:3:")
