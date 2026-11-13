"""The scheduled `run-daily` flow end to end with fake network sources, and the zip-cache fetcher."""
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from energy_service import cli, db, registry
from energy_service.config import Settings
from energy_service.http import Fetched, NotYetPublished
from energy_service.ingest import nyiso
from tests.service_helpers import (
    FakeSite,
    build_world,
    dam_csv,
    load_csv,
    populate_days,
    zip_of,
)

NOW = datetime(2026, 11, 12, 10, 30, tzinfo=UTC)          # 05:30 ET on 2026-11-12


class FrozenClock:
    @staticmethod
    def now():
        return NOW


def settings(tmp_path):
    return Settings(db_path=str(tmp_path / "c.sqlite"), model_dir=str(tmp_path / "m"), results_dir=str(tmp_path / "r"),
                    zip_cache_dir=str(tmp_path / "zips"))


def weather_body(target: date):
    idx = pd.date_range(pd.Timestamp(target) - pd.Timedelta(days=1), periods=96, freq="h", tz="UTC")
    return json.dumps({"hourly": {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in idx], "temperature_2m": [9.0] * len(idx)}}).encode()


def test_run_daily_ingests_forecasts_and_is_idempotent(tmp_path, monkeypatch, capsys):
    s = settings(tmp_path)
    monkeypatch.setenv("ENERGY_DB_PATH", s.db_path)
    monkeypatch.setenv("ENERGY_MODEL_DIR", s.model_dir)
    monkeypatch.setenv("ENERGY_RESULTS_DIR", s.results_dir)
    conn = db.connect(s.db_path)
    db.migrate(conn)
    build_world(conn, date(2026, 11, 11))                    # history through yesterday; today's files arrive via ingest
    conn.close()
    site = FakeSite()
    populate_days(site, date(2026, 11, 9), date(2026, 11, 12), load_end=date(2026, 11, 11))
    live = Fetched(weather_body(date(2026, 11, 13)), None, "weather")

    def fetch(url):
        return live if "api.open-meteo.com" in url else site(url)
    monkeypatch.setattr(cli, "make_fetcher", lambda settings: fetch)
    monkeypatch.setattr(cli, "_now", lambda: NOW)

    assert cli.main(["run-daily"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["forecast"]["target_date"] == "2026-11-13" and out["forecast"]["served"] == "baseline"
    assert out["forecast"]["fallback_reason"] == "no_active_model"            # nothing trained/eligible: baseline is the default
    assert out["weather_snapshot"] and out["ingest"]["status"] == "success"
    conn = db.connect(s.db_path)
    n = conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
    prices = conn.execute("SELECT COUNT(*) FROM prices_dam").fetchone()[0]
    conn.close()
    assert n == 1
    assert cli.main(["run-daily"]) == 0                                       # second run: no duplicates
    out2 = json.loads(capsys.readouterr().out)
    assert out2["forecast"]["already_existed"] is True
    conn = db.connect(s.db_path)
    assert conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM prices_dam").fetchone()[0] == prices
    conn.close()


def test_run_daily_exits_nonzero_when_no_forecast_is_possible(tmp_path, monkeypatch, capsys):
    s = settings(tmp_path)
    monkeypatch.setenv("ENERGY_DB_PATH", s.db_path)
    monkeypatch.setattr(cli, "make_fetcher", lambda settings: (lambda url: (_ for _ in ()).throw(NotYetPublished(url))))
    monkeypatch.setattr(cli, "_now", lambda: NOW)
    assert cli.main(["run-daily"]) == 2
    assert "error" in json.loads(capsys.readouterr().out)["forecast"]


def test_status_and_db_init_commands(tmp_path, monkeypatch, capsys):
    s = settings(tmp_path)
    monkeypatch.setenv("ENERGY_DB_PATH", s.db_path)
    assert cli.main(["db-init"]) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == len(list(db.MIGRATIONS_DIR.glob("*.sql")))
    monkeypatch.setattr(cli, "_now", lambda: NOW)
    assert cli.main(["status"]) == 0
    assert json.loads(capsys.readouterr().out)["readiness"]["status"] == "not_ready"      # empty database


def test_train_command_reports_ineligible_without_evidence(tmp_path, monkeypatch, capsys):
    s = settings(tmp_path)
    monkeypatch.setenv("ENERGY_DB_PATH", s.db_path)
    monkeypatch.setenv("ENERGY_MODEL_DIR", s.model_dir)
    monkeypatch.setenv("ENERGY_RESULTS_DIR", s.results_dir)           # empty results dir: no backtest evidence
    conn = db.connect(s.db_path)
    db.migrate(conn)
    build_world(conn, date(2026, 11, 12))
    conn.close()
    monkeypatch.setattr(cli, "_now", lambda: NOW)
    Path(s.results_dir).mkdir()
    (Path(s.results_dir) / "operational_run_metadata.json").write_text(
        json.dumps({"chosen_params": {"OB": {"max_depth": 3, "n_estimators": 30}, "OD": {"max_depth": 3, "n_estimators": 30}}}))
    assert cli.main(["train", "--feature-set", "OD", "--activate"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["active"] is False and "serving rule" in out["reason"]
    conn = db.connect(s.db_path)
    assert conn.execute("SELECT eligible, active FROM models").fetchone()[:] == (0, 0)
    conn.close()
    assert registry.evidence_from_results(s.results_dir, "OD")["eligible"] is False


def test_cached_fetcher_serves_local_monthly_zips_and_falls_back_to_the_network(tmp_path):
    zdir = tmp_path / "zips"
    zdir.mkdir()
    (zdir / "damlbmp_zone_202409.zip").write_bytes(zip_of({"a.csv": dam_csv(date(2024, 9, 1))}))
    site = FakeSite()
    site.put_monthly("actual_load", 2024, 9, zip_of({"a.csv": load_csv(date(2024, 9, 1))}))
    f = cli.CachedFetcher(site, str(zdir))
    got = f(nyiso.monthly_url("dam_price", 2024, 9))
    assert got.body.startswith(b"PK") and site.calls == []                       # served from disk
    assert f(nyiso.monthly_url("actual_load", 2024, 9)).body.startswith(b"PK") and len(site.calls) == 1
    with pytest.raises(NotYetPublished):
        f(nyiso.monthly_url("dam_price", 2024, 10))


def test_settings_from_env_overrides_only_what_is_set():
    s = Settings.from_env({"ENERGY_ZONE": "WEST", "ENERGY_HTTP_RETRIES": "5", "ENERGY_API_KEY": "k"})
    assert (s.zone, s.retries, s.api_key) == ("WEST", 5, "k") and s.db_path == Settings().db_path


def test_scheduler_catchup_switch_parses_booleans():
    assert Settings().scheduler_catchup is True
    for v, want in (("false", False), ("0", False), ("no", False), ("TRUE", True), ("1", True)):
        assert Settings.from_env({"ENERGY_SCHEDULER_CATCHUP": v}).scheduler_catchup is want
