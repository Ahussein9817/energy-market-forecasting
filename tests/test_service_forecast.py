"""Train -> register -> forecast -> evaluate, including every fallback path (offline, synthetic world in SQLite)."""
import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy_service import db, registry
from energy_service.config import Settings
from energy_service.evaluate import actual_day, evaluate_pending
from energy_service.forecast import ForecastUnavailable, generate_forecast
from energy_service.train import train_model
from tests.service_helpers import add_live_snapshot, build_world

D = date(2026, 11, 12)                                     # origin day; target = 2026-11-13
NOW = datetime(2026, 11, 12, 10, 30, tzinfo=UTC)           # 05:30 ET: before the day-ahead prices are public
ELIGIBLE = {"backtest_model": "xgb_OD_train_actual", "eligible": True, "rule": "test"}
PARAMS = {"max_depth": 3, "n_estimators": 60}


@pytest.fixture
def world(tmp_path):
    s = Settings(db_path=str(tmp_path / "w.sqlite"), model_dir=str(tmp_path / "models"), results_dir=str(tmp_path / "res"))
    conn = db.connect(s.db_path)
    db.migrate(conn)
    build_world(conn, D)
    yield conn, s
    conn.close()


def trained(conn, s, evidence=ELIGIBLE, activate=True):
    mid = train_model(conn, s, feature_set="OD", origin_date=D, now=NOW, params=PARAMS, evidence=evidence)
    if activate and evidence.get("eligible"):
        registry.activate(conn, mid)
    return mid


def fc_hours(conn, fid):
    return conn.execute("SELECT * FROM forecast_hours WHERE forecast_id=? ORDER BY hour", (fid,)).fetchall()


# ---- registry / training --------------------------------------------------------------------
def test_training_persists_a_versioned_hash_checked_model(world):
    conn, s = world
    mid = trained(conn, s, activate=False)
    d = Path(s.model_dir) / mid
    meta = json.loads((d / "metadata.json").read_text())
    assert meta["feature_set"] == "OD" and meta["feature_version"] == registry.FEATURE_VERSION
    assert meta["model_sha256"] == registry.sha256_file(d / "model.joblib") and meta["n_rows"] > 24 * 60
    assert {"temp_fc_t", "load_d2", "p_d1"} <= set(meta["columns"]) and "load_d1" not in meta["columns"]
    row = conn.execute("SELECT * FROM models WHERE model_id=?", (mid,)).fetchone()
    assert row["eligible"] == 1 and row["active"] == 0 and row["sha256"] == meta["model_sha256"]
    assert conn.execute("SELECT status FROM job_runs WHERE job='train'").fetchone()[0] == "success"


def test_ineligible_model_cannot_be_activated(world):
    conn, s = world
    mid = trained(conn, s, evidence={"eligible": False}, activate=False)
    with pytest.raises(registry.NotEligible):
        registry.activate(conn, mid)
    assert conn.execute("SELECT active FROM models WHERE model_id=?", (mid,)).fetchone()[0] == 0


def test_evidence_from_results_requires_files_and_follows_the_decision(tmp_path):
    assert registry.evidence_from_results(str(tmp_path), "OD")["eligible"] is False      # no files, no evidence
    (tmp_path / "operational_decision.json").write_text(json.dumps({"eligible_models": ["xgb_OD_train_actual"], "rule": "r"}))
    (tmp_path / "operational_metrics.json").write_text(json.dumps({"vs_baseline": {"xgb_OD_train_actual": {
        "mae_diff_ci": {"hi": -0.5}, "dm": {"p_value": 0.001}}}}))
    ev = registry.evidence_from_results(str(tmp_path), "OD")
    assert ev["eligible"] is True and ev["dm_p_value"] == 0.001 and ev["decision_sha256"]
    assert registry.evidence_from_results(str(tmp_path), "OB")["eligible"] is False


def test_too_little_history_is_refused(tmp_path):
    s = Settings(db_path=str(tmp_path / "x.sqlite"), model_dir=str(tmp_path / "m"), results_dir=str(tmp_path))
    conn = db.connect(s.db_path)
    db.migrate(conn)
    build_world(conn, D, days=40)
    from energy_service.dataset import DataIncomplete
    with pytest.raises(DataIncomplete):
        train_model(conn, s, feature_set="OD", origin_date=D, now=NOW, params=PARAMS, evidence=ELIGIBLE)
    assert conn.execute("SELECT status FROM job_runs WHERE job='train'").fetchone()[0] == "failed"


# ---- forecasting ----------------------------------------------------------------------------
def test_model_forecast_with_full_provenance(world):
    conn, s = world
    mid = trained(conn, s)
    snap = add_live_snapshot(conn, date(2026, 11, 13), datetime(2026, 11, 12, 9, 0, tzinfo=UTC))
    r = generate_forecast(conn, s, origin_date=D, now=NOW)
    assert (r.served_kind, r.fallback_reason, r.late, r.existing, r.version) == ("model", None, False, False, 1)
    f = conn.execute("SELECT * FROM forecasts WHERE id=?", (r.forecast_id,)).fetchone()
    inp = json.loads(f["input_json"])
    assert f["target_date"] == "2026-11-13" and f["origin_ts"] == "2026-11-12T10:00:00Z" and f["generated_at"] == "2026-11-12T10:30:00Z"
    assert f["served_model_id"] == mid and inp["model_id"] == mid and inp["weather_snapshot_id"] == snap
    assert inp["feature_version"] == registry.FEATURE_VERSION and inp["feature_hash"] and inp["weather_age_hours"] == 1.5
    h = fc_hours(conn, r.forecast_id)
    assert len(h) == 24 and all(x["model"] is not None and np.isfinite(x["served"]) for x in h)
    assert [x["served"] for x in h] == [x["model"] for x in h]
    assert all(0 < x["model"] < 200 for x in h)
    # baseline = price of day D, same local hour
    d_prices = [conn.execute("SELECT lbmp FROM prices_dam WHERE ts_utc=?", (t.strftime("%Y-%m-%dT%H:%M:%SZ"),)).fetchone()[0]
                for t in pd.date_range("2026-11-12", periods=24, freq="h", tz="America/New_York").tz_convert("UTC")]
    assert [x["baseline"] for x in h] == pytest.approx(d_prices)


def test_forecast_is_idempotent_and_force_creates_a_new_version(world):
    conn, s = world
    trained(conn, s)
    add_live_snapshot(conn, date(2026, 11, 13), datetime(2026, 11, 12, 9, 0, tzinfo=UTC))
    a = generate_forecast(conn, s, origin_date=D, now=NOW)
    b = generate_forecast(conn, s, origin_date=D, now=NOW)
    assert b.existing and b.forecast_id == a.forecast_id
    assert conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == 1
    c = generate_forecast(conn, s, origin_date=D, now=NOW, force=True)
    assert c.version == 2 and not c.existing and conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == 2


def test_forecast_does_not_use_information_from_after_the_origin(world):
    conn, s = world
    trained(conn, s)
    add_live_snapshot(conn, date(2026, 11, 13), datetime(2026, 11, 12, 9, 0, tzinfo=UTC))
    a = generate_forecast(conn, s, origin_date=D, now=NOW)
    before = [x["model"] for x in fc_hours(conn, a.forecast_id)]
    # later information appears: the target day's prices, and the load of day D
    build_extra = pd.date_range("2026-11-13", periods=24, freq="h", tz="America/New_York").tz_convert("UTC")
    rid = conn.execute("SELECT id FROM raw_files LIMIT 1").fetchone()[0]
    conn.executemany("INSERT INTO prices_dam (zone, ts_utc, lbmp, raw_file_id) VALUES ('N.Y.C.',?,?,?)",
                     [(t.strftime("%Y-%m-%dT%H:%M:%SZ"), 999.0, rid) for t in build_extra])
    conn.executemany("INSERT INTO load_actual (zone, ts_utc, load_mw, raw_file_id) VALUES ('N.Y.C.',?,?,?)",
                     [(t.strftime("%Y-%m-%dT%H:%M:%SZ"), 40000.0, rid)
                      for t in pd.date_range("2026-11-12", periods=24, freq="h", tz="America/New_York").tz_convert("UTC")])
    conn.commit()
    b = generate_forecast(conn, s, origin_date=D, now=NOW, force=True)
    assert [x["model"] for x in fc_hours(conn, b.forecast_id)] == before


@pytest.mark.parametrize("scenario,reason", [
    ("no_model", "no_active_model"),
    ("not_eligible", "model_not_eligible"),
    ("no_weather", "weather_missing"),
    ("stale_weather", "weather_stale"),
    ("null_weather_hours", "weather_missing"),
    ("no_load", "load_missing"),
    ("model_file_missing", "model_load_failed"),
    ("model_file_tampered", "model_load_failed"),
    ("old_feature_version", "feature_version_mismatch"),
])
def test_every_fallback_serves_the_baseline_and_records_why(world, scenario, reason):
    conn, s = world
    snap_time = datetime(2026, 11, 12, 9, 0, tzinfo=UTC)
    if scenario == "stale_weather":
        snap_time = datetime(2026, 11, 11, 3, 0, tzinfo=UTC)         # 31.5 h old
    if scenario != "no_model":
        mid = trained(conn, s)
    if scenario != "no_weather":
        add_live_snapshot(conn, date(2026, 11, 13), snap_time, null_hours=(5, 6) if scenario == "null_weather_hours" else ())
    if scenario == "not_eligible":
        conn.execute("UPDATE models SET eligible=0")
    if scenario == "no_load":
        conn.execute("DELETE FROM load_actual WHERE ts_utc >= '2026-11-10T00:00:00Z'")
    if scenario == "model_file_missing":
        (Path(s.model_dir) / mid / "model.joblib").unlink()
    if scenario == "model_file_tampered":
        with open(Path(s.model_dir) / mid / "model.joblib", "ab") as fh:
            fh.write(b"tamper")
    if scenario == "old_feature_version":
        conn.execute("UPDATE models SET feature_version='ancient'")
    conn.commit()
    r = generate_forecast(conn, s, origin_date=D, now=NOW)
    assert r.served_kind == "baseline" and r.fallback_reason == reason
    h = fc_hours(conn, r.forecast_id)
    assert all(x["model"] is None and x["served"] == x["baseline"] for x in h) and len(h) == 24
    assert conn.execute("SELECT COUNT(*) FROM events WHERE code='forecast_fallback'").fetchone()[0] == 1
    assert conn.execute("SELECT status FROM job_runs WHERE job='forecast'").fetchone()[0] == "success"


def test_missing_price_of_day_D_fails_loudly_without_a_forecast_row(world):
    conn, s = world
    conn.execute("DELETE FROM prices_dam WHERE ts_utc >= '2026-11-12T10:00:00Z'")      # part of day D missing
    conn.commit()
    with pytest.raises(ForecastUnavailable):
        generate_forecast(conn, s, origin_date=D, now=NOW)
    assert conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == 0
    assert conn.execute("SELECT status FROM job_runs WHERE job='forecast'").fetchone()[0] == "failed"
    assert conn.execute("SELECT COUNT(*) FROM events WHERE code='forecast_failed'").fetchone()[0] == 1


def test_late_flag_marks_forecasts_made_after_prices_are_public(world):
    conn, s = world
    early = generate_forecast(conn, s, origin_date=D, now=datetime(2026, 11, 12, 11, 0, tzinfo=UTC))   # 06:00 ET
    assert early.late is False
    late = generate_forecast(conn, s, origin_date=D, now=datetime(2026, 11, 12, 16, 0, tzinfo=UTC), force=True)  # 11:00 ET
    assert late.late is True


# ---- evaluation -----------------------------------------------------------------------------
def test_evaluation_waits_for_actuals_then_scores_each_forecast_once(world):
    conn, s = world
    trained(conn, s)
    add_live_snapshot(conn, date(2026, 11, 13), datetime(2026, 11, 12, 9, 0, tzinfo=UTC))
    r = generate_forecast(conn, s, origin_date=D, now=NOW)
    out = evaluate_pending(conn, s, now=NOW)
    assert out == {"evaluated": 0, "waiting_for_actuals": 1}
    # day-ahead prices for the target day are published (~09:32 ET)
    rid = conn.execute("SELECT id FROM raw_files LIMIT 1").fetchone()[0]
    idx = pd.date_range("2026-11-13", periods=24, freq="h", tz="America/New_York").tz_convert("UTC")
    actual = np.linspace(30, 60, 24)
    conn.executemany("INSERT INTO prices_dam (zone, ts_utc, lbmp, raw_file_id) VALUES ('N.Y.C.',?,?,?)",
                     [(t.strftime("%Y-%m-%dT%H:%M:%SZ"), float(v), rid) for t, v in zip(idx, actual)])
    conn.commit()
    out = evaluate_pending(conn, s, now=NOW)
    assert out == {"evaluated": 1, "waiting_for_actuals": 0}
    e = conn.execute("SELECT * FROM evaluations WHERE forecast_id=?", (r.forecast_id,)).fetchone()
    h = fc_hours(conn, r.forecast_id)
    assert e["mae_served"] == pytest.approx(np.mean([abs(x["served"] - a) for x, a in zip(h, actual)]))
    assert e["mae_baseline"] == pytest.approx(np.mean([abs(x["baseline"] - a) for x, a in zip(h, actual)]))
    assert e["mae_model"] is not None
    assert evaluate_pending(conn, s, now=NOW) == {"evaluated": 0, "waiting_for_actuals": 0}     # idempotent
    assert conn.execute("SELECT COUNT(*) FROM evaluation_hours").fetchone()[0] == 24


def test_actual_day_requires_all_hours(world):
    conn, s = world
    assert actual_day(conn, s, pd.Timestamp("2026-11-12")) is not None
    assert actual_day(conn, s, pd.Timestamp("2026-11-13")) is None
