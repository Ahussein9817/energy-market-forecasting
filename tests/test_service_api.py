"""API behaviour with a seeded database: success paths, error bodies, and predictable behaviour when data or models are missing."""
from datetime import UTC, date, datetime

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from energy_service import db, registry
from energy_service.api import create_app
from energy_service.config import Settings
from energy_service.evaluate import evaluate_pending
from energy_service.forecast import generate_forecast
from energy_service.train import train_model
from tests.service_helpers import add_live_snapshot, build_world

D = date(2026, 11, 12)
MORNING = datetime(2026, 11, 12, 10, 30, tzinfo=UTC)       # 05:30 ET
PARAMS = {"max_depth": 3, "n_estimators": 40}


def settings(tmp_path):
    return Settings(db_path=str(tmp_path / "a.sqlite"), model_dir=str(tmp_path / "m"), results_dir=str(tmp_path / "r"))


def client(s, now=MORNING):
    return TestClient(create_app(s, clock=lambda: now), raise_server_exceptions=False)


@pytest.fixture
def seeded(tmp_path):
    s = settings(tmp_path)
    conn = db.connect(s.db_path)
    db.migrate(conn)
    build_world(conn, D)
    mid = train_model(conn, s, feature_set="OD", origin_date=D, now=MORNING, params=PARAMS,
                      evidence={"eligible": True, "backtest_model": "xgb_OD_train_actual"})
    registry.activate(conn, mid)
    add_live_snapshot(conn, date(2026, 11, 13), datetime(2026, 11, 12, 9, 0, tzinfo=UTC))
    generate_forecast(conn, s, origin_date=D, now=MORNING)
    conn.close()
    return s, mid


def test_health_is_independent_of_the_database(tmp_path):
    c = client(settings(tmp_path))          # no database file exists
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    assert r.headers["x-request-id"]


def test_ready_ok_when_prices_current_and_model_active(seeded):
    s, mid = seeded
    r = client(s).get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready" and body["active_model"] == mid and body["latest_price_date"] == "2026-11-12"
    assert body["forecast_for_tomorrow"]["served_kind"] == "model"


def test_ready_is_503_with_reasons_when_prices_are_stale(seeded):
    s, _ = seeded
    r = client(s, now=datetime(2026, 11, 15, 15, 0, tzinfo=UTC)).get("/ready")        # three days later, no new ingest
    assert r.status_code == 503
    assert r.json()["status"] == "not_ready" and any("prices_stale" in x for x in r.json()["reasons"])


def test_ready_is_503_when_database_is_missing_or_empty(tmp_path):
    s = settings(tmp_path)
    r = client(s).get("/ready")
    assert r.status_code == 503 and r.json()["error"]["code"] == "db_unavailable"
    conn = db.connect(s.db_path)
    db.migrate(conn)
    conn.close()
    r2 = client(s).get("/ready")
    assert r2.status_code == 503 and "no_price_data" in r2.json()["reasons"]


def test_degraded_but_ready_when_no_model_is_active(seeded):
    s, _ = seeded
    conn = db.connect(s.db_path)
    conn.execute("UPDATE models SET active=0")
    conn.commit()
    conn.close()
    r = client(s).get("/ready")
    assert r.status_code == 200 and r.json()["status"] == "degraded"
    assert any("no_active_model" in x for x in r.json()["degraded"])


def test_models_endpoint_shows_baseline_candidate_and_evidence(seeded):
    s, mid = seeded
    j = client(s).get("/models").json()
    assert j["serving_now"] == mid and j["baseline"]["name"] == "same_hour_of_day_D"
    m = j["models"][0]
    assert m["model_id"] == mid and m["eligible"] and m["active"] and m["evidence"]["backtest_model"] == "xgb_OD_train_actual"


def test_models_endpoint_says_baseline_when_nothing_is_active(tmp_path):
    s = settings(tmp_path)
    conn = db.connect(s.db_path)
    db.migrate(conn)
    conn.close()
    assert client(s).get("/models").json()["serving_now"] == "baseline"


def test_latest_forecast_has_hours_provenance_and_no_evaluation_yet(seeded):
    s, mid = seeded
    r = client(s).get("/forecasts/latest")
    assert r.status_code == 200
    j = r.json()
    assert j["target_date"] == "2026-11-13" and j["served_kind"] == "model" and j["served_model_id"] == mid
    assert len(j["hours"]) == 24 and j["hours"][0]["baseline_usd_mwh"] > 0 and j["evaluation"] is None
    assert j["inputs"]["feature_version"] == registry.FEATURE_VERSION and j["late"] is False


def test_latest_forecast_404_when_none_exists(tmp_path):
    s = settings(tmp_path)
    conn = db.connect(s.db_path)
    db.migrate(conn)
    conn.close()
    r = client(s).get("/forecasts/latest")
    assert r.status_code == 404 and r.json() == {"error": {"code": "no_forecast", "message": "no forecast has been generated yet",
                                                           "details": None}}


def test_history_filters_and_includes_evaluation_once_actuals_arrive(seeded):
    s, _ = seeded
    conn = db.connect(s.db_path)
    rid = conn.execute("SELECT id FROM raw_files LIMIT 1").fetchone()[0]
    idx = pd.date_range("2026-11-13", periods=24, freq="h", tz="America/New_York").tz_convert("UTC")
    conn.executemany("INSERT INTO prices_dam (zone, ts_utc, lbmp, raw_file_id) VALUES ('N.Y.C.',?,?,?)",
                     [(t.strftime("%Y-%m-%dT%H:%M:%SZ"), 50.0, rid) for t in idx])
    conn.commit()
    evaluate_pending(conn, s, now=MORNING)
    conn.close()
    c = client(s)
    j = c.get("/forecasts", params={"target_date": "2026-11-13", "include_hours": "true"}).json()
    assert j["count"] == 1 and j["forecasts"][0]["evaluation"]["mae_baseline"] > 0
    assert len(j["forecasts"][0]["evaluation"]["actual_usd_mwh"]) == 24
    assert c.get("/forecasts", params={"date_from": "2026-11-14"}).json()["count"] == 0
    assert c.get("/forecasts").json()["forecasts"][0]["hours"] is None            # summaries omit hours by default


def test_history_returns_only_the_latest_version_per_day(seeded):
    s, _ = seeded
    conn = db.connect(s.db_path)
    generate_forecast(conn, s, origin_date=D, now=MORNING, force=True)
    conn.close()
    j = client(s).get("/forecasts").json()
    assert j["count"] == 1 and j["forecasts"][0]["version"] == 2


@pytest.mark.parametrize("params,code", [
    ({"target_date": "not-a-date"}, "invalid_request"),
    ({"limit": 0}, "invalid_request"),
    ({"limit": 9999}, "invalid_request"),
    ({"target_date": "2026-11-13", "date_from": "2026-11-01"}, "invalid_request"),
    ({"date_from": "2026-11-20", "date_to": "2026-11-01"}, "invalid_request"),
])
def test_invalid_requests_get_a_consistent_422_body(seeded, params, code):
    s, _ = seeded
    r = client(s).get("/forecasts", params=params)
    assert r.status_code == 422 and r.json()["error"]["code"] == code


def test_unknown_forecast_date_is_404_and_unknown_route_is_an_error_body(seeded):
    s, _ = seeded
    c = client(s)
    r = c.get("/forecasts", params={"target_date": "2030-01-01"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "no_forecast"
    r2 = c.get("/nope")
    assert r2.status_code == 404 and "error" in r2.json()


def test_no_endpoint_mutates_state_or_computes(seeded):
    s, _ = seeded
    c = client(s)
    for path in ("/forecasts", "/forecasts/latest", "/models", "/health", "/ready"):
        assert c.post(path).status_code in (404, 405)          # read-only: no write verbs are exposed
    conn = db.connect(s.db_path)
    before = conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0], conn.execute("SELECT COUNT(*) FROM job_runs").fetchone()[0]
    for _ in range(3):
        c.get("/forecasts/latest")
    after = conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0], conn.execute("SELECT COUNT(*) FROM job_runs").fetchone()[0]
    conn.close()
    assert before == after


def test_requests_are_logged_with_structured_fields(seeded, caplog):
    s, _ = seeded
    with caplog.at_level("INFO", logger="energy.api"):
        client(s).get("/health", headers={"X-Request-ID": "abc123"})
    rec = caplog.records[-1].fields
    assert rec["event"] == "request" and rec["path"] == "/health" and rec["status"] == 200 and rec["request_id"] == "abc123"
