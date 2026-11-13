"""Monitoring checks, transitions, alerts, and the shared queries."""
import json
from datetime import UTC, date, datetime

import pytest

from energy_service import db, monitor, queries
from energy_service.config import Settings
from energy_service.ingest import nyiso
from tests.service_helpers import FakeSite, build_world, load_csv, seed_evaluated

D = date(2026, 11, 12)
NOW = datetime(2026, 11, 12, 15, 0, tzinfo=UTC)          # 10:00 ET
S = Settings()


@pytest.fixture
def conn(tmp_path):
    c = db.connect(str(tmp_path / "m.sqlite"))
    db.migrate(c)
    build_world(c, D, days=60)
    yield c
    c.close()


def clear(conn):
    for t in ("evaluation_hours", "evaluations", "forecast_hours", "forecasts"):
        conn.execute(f"DELETE FROM {t}")
    conn.commit()


def by_name(checks):
    return {c.name: c for c in checks}


def ingest_ok(conn, when=NOW):
    conn.execute("INSERT INTO job_runs (job, started_at, finished_at, status) VALUES ('ingest',?,?, 'success')",
                 (db.iso(when), db.iso(when)))
    conn.commit()


def test_fresh_pipeline_is_ok_where_evidence_exists(conn):
    ingest_ok(conn)
    c = by_name(monitor.run_checks(conn, S, NOW))
    assert c["prices_fresh"].status == "ok" and c["ingest_recent"].status == "ok"
    assert c["performance_vs_baseline"].status == "unknown"            # nothing evaluated yet: not claimed
    assert c["forecast_fallbacks"].status == "unknown"
    assert c["model_integrity"].status == "warn"                       # no active model: baseline only
    assert c["forecast_available"].status == "fail"                    # 10:00 ET and nothing generated: the one real failure
    assert monitor.overall(list(c.values())) == "fail"


def test_stale_prices_and_missing_ingest_fail(conn):
    later = datetime(2026, 11, 16, 15, 0, tzinfo=UTC)
    c = by_name(monitor.run_checks(conn, S, later))
    assert c["prices_fresh"].status == "fail" and "stale" in c["prices_fresh"].message
    assert c["ingest_recent"].status == "unknown"
    ingest_ok(conn, NOW)
    assert by_name(monitor.run_checks(conn, S, later))["ingest_recent"].status == "fail"   # success too old


def test_partial_ingest_warns(conn):
    ingest_ok(conn, datetime(2026, 11, 12, 14, 0, tzinfo=UTC))
    conn.execute("INSERT INTO job_runs (job, started_at, finished_at, status) VALUES ('ingest',?,?, 'partial')",
                 (db.iso(NOW), db.iso(NOW)))
    conn.commit()
    assert by_name(monitor.run_checks(conn, S, NOW))["ingest_recent"].status == "warn"


def test_forecast_availability_depends_on_the_time_of_day(conn):
    early = datetime(2026, 11, 12, 10, 0, tzinfo=UTC)               # 05:00 ET: not yet due
    assert by_name(monitor.run_checks(conn, S, early))["forecast_available"].status == "ok"
    assert by_name(monitor.run_checks(conn, S, NOW))["forecast_available"].status == "fail"   # 10:00 ET, nothing generated


def test_performance_is_only_judged_with_enough_evaluated_days(conn):
    seed_evaluated(conn, D, days=5)
    assert by_name(monitor.run_checks(conn, S, NOW))["performance_vs_baseline"].status == "unknown"


def test_model_better_than_baseline_is_ok_worse_warns_much_worse_fails(conn):
    seed_evaluated(conn, D, days=15, served_mae=3.0, baseline_mae=5.0)
    c = by_name(monitor.run_checks(conn, S, NOW))["performance_vs_baseline"]
    assert c.status == "ok" and c.value["ratio"] == 0.6 and c.value["days"] == 15
    clear(conn)
    seed_evaluated(conn, D, days=15, served_mae=5.3, baseline_mae=5.0)
    assert by_name(monitor.run_checks(conn, S, NOW))["performance_vs_baseline"].status == "warn"
    clear(conn)
    seed_evaluated(conn, D, days=15, served_mae=6.0, baseline_mae=5.0)
    assert by_name(monitor.run_checks(conn, S, NOW))["performance_vs_baseline"].status == "fail"


def test_late_and_replayed_forecasts_never_count_towards_performance(conn):
    seed_evaluated(conn, D, days=15, served_mae=9.0, baseline_mae=5.0, late=1)              # would fail if counted
    seed_evaluated(conn, D, days=15, served_mae=9.0, baseline_mae=5.0, source="replay", version=2)
    c = by_name(monitor.run_checks(conn, S, NOW))["performance_vs_baseline"]
    assert c.status == "unknown" and c.value["days"] == 0


def test_error_spike_detection(conn):
    seed_evaluated(conn, date(2026, 11, 11), days=14, served_mae=3.0, baseline_mae=5.0, start_id=1000)
    seed_evaluated(conn, D, days=1, served_mae=3.0, baseline_mae=5.0, start_id=3000)
    # vary prior errors slightly so sigma > 0, then make the last day far worse
    for i, r in enumerate(conn.execute("SELECT forecast_id FROM evaluations ORDER BY target_date").fetchall()[:-1]):
        conn.execute("UPDATE evaluations SET mae_served=? WHERE forecast_id=?", (3.0 + 0.1 * (i % 3), r[0]))
    conn.execute("UPDATE evaluations SET mae_served=20.0 WHERE target_date=?", (str(D),))
    conn.commit()
    assert by_name(monitor.run_checks(conn, S, NOW))["error_spike"].status == "warn"


def test_fallback_rate_warns_when_most_forecasts_use_the_baseline(conn):
    seed_evaluated(conn, D, days=8, kind="baseline", fallback="weather_missing")
    c = by_name(monitor.run_checks(conn, S, NOW))["forecast_fallbacks"]
    assert c.status == "warn" and c.value["reasons"] == {"weather_missing": 8}


def test_model_file_tampering_fails_the_integrity_check(conn, tmp_path):
    f = tmp_path / "model.joblib"
    import hashlib
    f.write_bytes(b"abc")
    conn.execute("INSERT INTO models (model_id, kind, feature_set, feature_version, path, sha256, trained_at, eligible, active) "
                 "VALUES ('m','xgb','OD','v',?,?, '2026-01-01T00:00:00Z', 1, 1)", (str(f), hashlib.sha256(b"abc").hexdigest()))
    conn.commit()
    assert by_name(monitor.run_checks(conn, S, NOW))["model_integrity"].status == "ok"
    f.write_bytes(b"abcd")
    assert by_name(monitor.run_checks(conn, S, NOW))["model_integrity"].status == "fail"
    f.unlink()
    assert by_name(monitor.run_checks(conn, S, NOW))["model_integrity"].status == "fail"


def test_resolved_missing_file_flags_do_not_alarm(conn):
    nyiso._add_flags(conn, NOW, "dam_price", [nyiso.Flag("expected_file_missing", "warning", "u", "2026-11-12")], None, "2026-11-12")
    nyiso._add_flags(conn, NOW, "dam_price", [nyiso.Flag("expected_file_missing", "warning", "u", "2030-01-01")], None, "2030-01-01")
    c = by_name(monitor.run_checks(conn, S, NOW))["data_quality_48h"]
    assert c.status == "warn" and c.value["by_code"] == {"expected_file_missing": 1}        # only the day that is still missing


def test_transitions_are_recorded_once_and_alerts_go_out_for_failures(conn):
    sent = []

    class R:
        def raise_for_status(self):
            pass

    def post(url, json=None, timeout=None):
        sent.append((url, json))
        return R()

    s = Settings(alert_webhook="https://hooks.example/test")
    later = datetime(2026, 11, 16, 15, 0, tzinfo=UTC)
    r1 = monitor.run_monitor(conn, s, later, post=post)                 # prices stale -> fail
    assert any(t["check"] == "prices_fresh" and t["to"] == "fail" for t in r1["transitions"])
    assert len(sent) >= 1 and "prices_fresh" in sent[0][1]["text"] and sent[0][0] == s.alert_webhook
    n_events = conn.execute("SELECT COUNT(*) FROM events WHERE code='monitor_transition'").fetchone()[0]
    r2 = monitor.run_monitor(conn, s, later, post=post)                 # same state again: nothing new
    assert r2["transitions"] == [] and r2["alerts_sent"] == 0
    assert conn.execute("SELECT COUNT(*) FROM events WHERE code='monitor_transition'").fetchone()[0] == n_events
    assert conn.execute("SELECT COUNT(*) FROM monitor_snapshots").fetchone()[0] == 2
    # data returns: recovery is also reported
    site = FakeSite()
    site.put_daily("dam_price", date(2026, 11, 16), b"x")
    from tests.service_helpers import dam_csv
    for d in (date(2026, 11, 13), date(2026, 11, 14), date(2026, 11, 15), date(2026, 11, 16)):
        site.put_daily("dam_price", d, dam_csv(d))
        site.put_daily("actual_load", d, load_csv(d))
    nyiso.run_ingest(conn, S, now=later, fetch=site, lookback_days=4)
    r3 = monitor.run_monitor(conn, s, later, post=post)
    assert any(t["check"] == "prices_fresh" and t["from"] == "fail" for t in r3["transitions"])
    assert any("fail -> " in x[1]["text"] for x in sent)


def test_alert_delivery_failure_is_logged_and_never_raises(conn):
    def boom(url, json=None, timeout=None):
        raise RuntimeError("webhook down")
    s = Settings(alert_webhook="https://hooks.example/test")
    out = monitor.run_monitor(conn, s, datetime(2026, 11, 16, 15, 0, tzinfo=UTC), post=boom)
    assert out["alerts_sent"] == 0
    assert conn.execute("SELECT COUNT(*) FROM events WHERE code='alert_failed'").fetchone()[0] >= 1
    assert monitor.send_alerts(conn, Settings(), [{"check": "x", "from": "ok", "to": "fail", "message": "m"}], NOW) == 0   # disabled


def test_snapshot_is_valid_json_with_every_check(conn):
    monitor.run_monitor(conn, S, NOW)
    snap = json.loads(conn.execute("SELECT checks_json FROM monitor_snapshots").fetchone()[0])
    assert {c["name"] for c in snap} >= {"prices_fresh", "load_fresh", "ingest_recent", "forecast_available", "model_integrity",
                                         "performance_vs_baseline", "data_quality_48h"}
    assert conn.execute("SELECT status FROM job_runs WHERE job='monitor'").fetchone()[0] == "success"


# ---- shared queries -------------------------------------------------------------------------
def test_performance_and_spike_queries(conn):
    seed_evaluated(conn, D, days=6, served_mae=2.0, baseline_mae=4.0)
    h = queries.hourly_errors(conn, 90, today=datetime(2026, 11, 13))
    assert len(h) == 6 * 24 and set(h.columns) >= {"actual", "baseline", "model", "served"}
    t = queries.performance_table(h).set_index("forecast")
    assert t.loc["served", "MAE"] == pytest.approx(2.0) and t.loc["baseline", "MAE"] == pytest.approx(4.0)
    assert t.loc["served", "bias"] == pytest.approx(2.0)
    sp = queries.spike_analysis(h, threshold=60.0)
    assert sp["hours"] == 6 * 4 and sp["served_mae"] == pytest.approx(2.0) and sp["baseline_mae"] == pytest.approx(4.0)
    assert queries.spike_analysis(h, threshold=1e9) == {"threshold": 1e9, "hours": 0}


def test_day_detail_includes_forecast_actuals_and_inputs(conn):
    seed_evaluated(conn, D, days=2)
    d = queries.day_detail(conn, str(D), S.zone)
    assert d["forecast"]["target_date"] == str(D) and len(d["hours"]) == 24 and d["hours"]["actual"].notna().all()
    assert d["evaluation"]["mae_baseline"] == 5.0
    assert queries.day_detail(conn, "2030-01-01", S.zone) is None
