"""Dashboard smoke tests with Streamlit's AppTest against seeded databases."""
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from energy_service import db
from tests.service_helpers import add_live_snapshot, build_world, seed_evaluated

APP = str(Path(__file__).resolve().parent.parent / "dashboard" / "app.py")
D = date(2026, 11, 12)


def run(monkeypatch, path):
    monkeypatch.setenv("ENERGY_DB_PATH", str(path))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def texts(at):
    out = []
    for kind in ("warning", "info", "success", "error", "markdown", "caption"):
        out += [str(getattr(e, "value", "")) for e in getattr(at, kind)]
    return " | ".join(out)


@pytest.fixture
def seeded(tmp_path):
    p = tmp_path / "d.sqlite"
    conn = db.connect(str(p))
    db.migrate(conn)
    build_world(conn, D)
    seed_evaluated(conn, D, days=14, served_mae=3.0, baseline_mae=5.0)
    conn.close()
    return p


def test_dashboard_renders_all_tabs_with_data(monkeypatch, seeded):
    at = run(monkeypatch, seeded)
    assert not at.exception, [str(e.value) for e in at.exception]
    assert len(at.tabs) == 4 and [t.label for t in at.tabs] == ["Tomorrow", "Performance", "Data health", "Investigate a day"]
    assert any(m.label == "System status" for m in at.metric)
    t = texts(at)
    assert "Scored" in t                                       # the latest seeded day has actuals
    assert at.selectbox, "the investigate tab offers a day to pick"


def test_performance_is_not_overclaimed_with_little_data(monkeypatch, tmp_path):
    p = tmp_path / "few.sqlite"
    conn = db.connect(str(p))
    db.migrate(conn)
    build_world(conn, D)
    seed_evaluated(conn, D, days=3)
    conn.close()
    at = run(monkeypatch, p)
    assert not at.exception
    assert "Nothing below is evidence of performance yet" in texts(at)


def test_fallback_and_late_banners_are_shown(monkeypatch, tmp_path):
    p = tmp_path / "fb.sqlite"
    conn = db.connect(str(p))
    db.migrate(conn)
    build_world(conn, D)
    from energy_service.config import Settings
    from energy_service.forecast import generate_forecast
    s = Settings(db_path=str(p))
    add_live_snapshot(conn, date(2026, 11, 13), datetime(2026, 11, 12, 9, 0, tzinfo=UTC))
    generate_forecast(conn, s, origin_date=D, now=datetime(2026, 11, 12, 16, 0, tzinfo=UTC))      # 11:00 ET, no model: baseline and late
    conn.close()
    at = run(monkeypatch, p)
    assert not at.exception
    t = texts(at)
    assert "Baseline served: no model is active" in t and "Generated after the day-ahead prices were public" in t


def test_empty_database_and_missing_database_do_not_crash(monkeypatch, tmp_path):
    p = tmp_path / "e.sqlite"
    conn = db.connect(str(p))
    db.migrate(conn)
    conn.close()
    at = run(monkeypatch, p)
    assert not at.exception and "No forecast has been generated yet" in texts(at)
    at2 = run(monkeypatch, tmp_path / "does-not-exist.sqlite")
    assert not at2.exception and "No database" in texts(at2)
