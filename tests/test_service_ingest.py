"""Ingestion: idempotency, provenance, validation, failure handling (offline, synthetic NYISO-format files)."""
import json
from datetime import UTC, date, datetime

import pandas as pd
import pytest
import requests

from energy_service import db
from energy_service.config import Settings
from energy_service.http import FetchError, NotYetPublished, get_bytes
from energy_service.ingest import nyiso, weather
from energy_service.ingest.validate import (
    ConflictError,
    SchemaError,
    expected_hours,
    validate_frame,
)
from src import data_loader as dl
from tests.service_helpers import FakeSite, dam_csv, load_csv, populate_days, zip_of

S = Settings()
NOW = datetime(2026, 11, 12, 15, 0, tzinfo=UTC)  # 10:00 ET on 2026-11-12: today's DAM file for tomorrow is out


@pytest.fixture
def conn(tmp_path):
    c = db.connect(str(tmp_path / "t.sqlite"))
    db.migrate(c)
    yield c
    c.close()


def count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


# ---- migrations -----------------------------------------------------------------------------
def test_migrations_apply_once(tmp_path):
    c = db.connect(str(tmp_path / "m.sqlite"))
    assert db.migrate(c) == [1]
    assert db.migrate(c) == []          # idempotent
    assert db.schema_version(c) == 1
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"raw_files", "prices_dam", "forecasts", "models", "events", "job_runs"} <= tables


def test_iso_refuses_naive_timestamps():
    with pytest.raises(ValueError):
        db.iso(datetime(2026, 1, 1))


# ---- http helper ----------------------------------------------------------------------------
class Resp:
    def __init__(self, code, body=b"x", headers=None):
        self.status_code, self.content, self.headers = code, body, headers or {}


class Seq:
    def __init__(self, *items):
        self.items, self.n = list(items), 0

    def get(self, url, headers=None, params=None, timeout=None):
        self.n += 1
        it = self.items.pop(0) if len(self.items) > 1 else self.items[0]
        if isinstance(it, Exception):
            raise it
        return it


def test_http_404_is_not_retried():
    sess = Seq(Resp(404))
    with pytest.raises(NotYetPublished):
        get_bytes("u", session=sess, sleep=lambda s: None)
    assert sess.n == 1


def test_http_retries_5xx_and_connection_errors_then_succeeds_with_last_modified():
    sleeps = []
    sess = Seq(Resp(503), requests.ConnectionError("boom"),
               Resp(200, b"ok", {"Last-Modified": "Thu, 12 Nov 2026 14:33:03 GMT"}))
    f = get_bytes("u", session=sess, retries=3, backoff=2.0, sleep=sleeps.append)
    assert f.body == b"ok" and f.last_modified == datetime(2026, 11, 12, 14, 33, 3, tzinfo=UTC)
    assert sess.n == 3 and sleeps == [2.0, 4.0]       # exponential backoff


def test_http_gives_up_after_retries_and_does_not_retry_other_4xx():
    with pytest.raises(FetchError, match="after 2 attempts"):
        get_bytes("u", session=Seq(Resp(500)), retries=2, sleep=lambda s: None)
    sess = Seq(Resp(403))
    with pytest.raises(FetchError):
        get_bytes("u", session=sess, sleep=lambda s: None)
    assert sess.n == 1


# ---- validation -----------------------------------------------------------------------------
def frame(hours=24, **over):
    ts = pd.date_range("2026-11-12 05:00", periods=hours, freq="h", tz="UTC")
    d = pd.DataFrame({"timestamp_utc": ts, "zone": "N.Y.C.", "lbmp": 30.0})
    for k, v in over.items():
        d[k] = v
    return d


def test_expected_hours_handles_dst():
    assert expected_hours("2026-11-12") == 24
    assert expected_hours("2026-03-08") == 23   # spring forward
    assert expected_hours("2026-11-01") == 25   # fall back


def test_validate_flags_incomplete_day_and_keeps_rows():
    v = validate_frame(frame(hours=20), zone="N.Y.C.", value_col="lbmp", lo=-1000, hi=10000)
    assert len(v.df) == 20
    assert [f.code for f in v.flags] == ["incomplete_day"] and "20 of 24" in v.flags[0].detail


def test_validate_drops_out_of_range_and_nulls():
    d = frame()
    d.loc[3, "lbmp"] = 99999.0
    d.loc[4, "lbmp"] = None
    v = validate_frame(d, zone="N.Y.C.", value_col="lbmp", lo=-1000, hi=10000)
    codes = {f.code for f in v.flags}
    assert {"out_of_range", "null_value", "incomplete_day"} <= codes and len(v.df) == 22


def test_validate_identical_duplicates_dropped_conflicting_rejected():
    d = pd.concat([frame(), frame().iloc[[5]]], ignore_index=True)
    v = validate_frame(d, zone="N.Y.C.", value_col="lbmp", lo=-1000, hi=10000)
    assert any(f.code == "duplicate_rows" for f in v.flags) and len(v.df) == 24
    c = pd.concat([frame(), frame().iloc[[5]].assign(lbmp=77.0)], ignore_index=True)
    with pytest.raises(ConflictError):
        validate_frame(c, zone="N.Y.C.", value_col="lbmp", lo=-1000, hi=10000)


def test_validate_schema_errors():
    with pytest.raises(SchemaError):
        validate_frame(frame().drop(columns=["lbmp"]), zone="N.Y.C.", value_col="lbmp", lo=0, hi=1)
    with pytest.raises(SchemaError, match="zone"):
        validate_frame(frame(), zone="WEST", value_col="lbmp", lo=0, hi=100)
    odd = frame()
    odd["timestamp_utc"] = odd["timestamp_utc"] + pd.Timedelta(minutes=7)
    with pytest.raises(SchemaError, match="non-hourly"):
        validate_frame(odd, zone="N.Y.C.", value_col="lbmp", lo=-1000, hi=10000)


# ---- ingestion ------------------------------------------------------------------------------
def test_ingest_stores_rows_with_provenance_and_is_idempotent(conn):
    site = FakeSite()
    populate_days(site, date(2026, 11, 9), date(2026, 11, 13), load_end=date(2026, 11, 11))
    r1 = nyiso.run_ingest(conn, S, now=NOW, fetch=site, lookback_days=3)
    assert r1["status"] == "success"
    n_prices, n_load, n_raw = count(conn, "prices_dam"), count(conn, "load_actual"), count(conn, "raw_files")
    assert n_prices == 5 * 24 and n_load == 3 * 24                      # only the N.Y.C. zone is stored
    row = conn.execute("SELECT * FROM raw_files WHERE dataset='dam_price' LIMIT 1").fetchone()
    assert row["sha256"] and row["bytes"] > 0 and row["url"].startswith("https://mis.nyiso.com") \
        and row["last_modified"] and row["status"] == "ingested"
    r2 = nyiso.run_ingest(conn, S, now=NOW, fetch=site, lookback_days=3)   # run again: nothing changes
    assert (count(conn, "prices_dam"), count(conn, "load_actual"), count(conn, "raw_files")) == (n_prices, n_load, n_raw)
    assert r2["counts"].get("dam_price:unchanged") == 5 and "dam_price:ingested" not in r2["counts"]


def test_unpublished_files_are_not_errors_but_overdue_files_are_flagged(conn):
    site = FakeSite()
    populate_days(site, date(2026, 11, 9), date(2026, 11, 12))       # tomorrow's (11-13) price file is overdue
    out = nyiso.run_ingest(conn, S, now=NOW, fetch=site, lookback_days=3)
    assert out["counts"]["dam_price:expected_missing"] == 1           # 11-13 was due 09:35 ET on 11-12
    flags = conn.execute("SELECT code, delivery_date FROM quality_flags WHERE code='expected_file_missing'").fetchall()
    assert [(f[0], f[1]) for f in flags] == [("expected_file_missing", "2026-11-13")]
    # a file that is not yet due is simply "not published"
    early = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)                 # 07:00 ET, before 09:35
    c2 = db.connect(":memory:")
    db.migrate(c2)
    out2 = nyiso.run_ingest(c2, S, now=early, fetch=site, lookback_days=3)
    assert out2["counts"].get("dam_price:not_published") == 1 and "dam_price:expected_missing" not in out2["counts"]


def test_fetch_errors_make_job_partial_and_are_logged_other_files_still_load(conn):
    site = FakeSite()
    populate_days(site, date(2026, 11, 9), date(2026, 11, 13), load_end=date(2026, 11, 11))
    site.fail(nyiso.daily_url("dam_price", date(2026, 11, 10)))
    out = nyiso.run_ingest(conn, S, now=NOW, fetch=site, lookback_days=3)
    assert out["status"] == "partial" and out["counts"]["dam_price:error"] == 1
    assert count(conn, "prices_dam") == 4 * 24                        # the rest was ingested
    assert conn.execute("SELECT COUNT(*) FROM events WHERE code='fetch_failed'").fetchone()[0] == 1
    assert conn.execute("SELECT status FROM job_runs WHERE job='ingest'").fetchone()[0] == "partial"


def test_total_outage_fails_the_job_without_touching_stored_data(conn):
    site = FakeSite()
    populate_days(site, date(2026, 11, 9), date(2026, 11, 13), load_end=date(2026, 11, 11))
    nyiso.run_ingest(conn, S, now=NOW, fetch=site, lookback_days=3)
    before = count(conn, "prices_dam")
    down = FakeSite()
    for ds in nyiso.DATASETS:
        for k in range(-1, 6):
            down.fail(nyiso.daily_url(ds, date(2026, 11, 12 + k if 12 + k <= 30 else 30)))
    for d in range(9, 14):
        for ds in nyiso.DATASETS:
            down.fail(nyiso.daily_url(ds, date(2026, 11, d)))
    out = nyiso.run_ingest(conn, S, now=NOW, fetch=down, lookback_days=3)
    assert out["status"] == "failed" and count(conn, "prices_dam") == before


def test_malformed_file_is_rejected_and_flagged_without_storing_rows(conn):
    site = FakeSite()
    d = date(2026, 11, 12)
    site.put_daily("dam_price", d, b"<html>Service unavailable</html>")
    site.put_daily("actual_load", d, load_csv(d))
    nyiso.run_ingest(conn, S, now=NOW, fetch=site, lookback_days=0)
    assert count(conn, "prices_dam") == 0
    raw = conn.execute("SELECT status, detail FROM raw_files WHERE dataset='dam_price'").fetchone()
    assert raw["status"] == "rejected"
    assert conn.execute("SELECT COUNT(*) FROM quality_flags WHERE code='file_rejected'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM events WHERE code='file_rejected'").fetchone()[0] == 1
    assert count(conn, "load_actual") == 24                           # the good file still loaded


def test_wrong_columns_are_rejected(conn):
    site = FakeSite()
    d = date(2026, 11, 12)
    site.put_daily("dam_price", d, b"Time Stamp,Name,Price\n11/12/2026 00:00,N.Y.C.,5\n")
    nyiso.run_ingest(conn, S, now=NOW, fetch=site, lookback_days=0)
    assert count(conn, "prices_dam") == 0
    assert conn.execute("SELECT status FROM raw_files WHERE dataset='dam_price'").fetchone()[0] == "rejected"


def test_conflicting_duplicate_rows_reject_the_file(conn):
    site = FakeSite()
    d = date(2026, 11, 12)
    extra = "11/12/2026 05:00,N.Y.C.,1,999.0,1.0,0.5\n"            # same hour, different price
    site.put_daily("dam_price", d, dam_csv(d, extra_rows=extra))
    nyiso.run_ingest(conn, S, now=NOW, fetch=site, lookback_days=0)
    assert count(conn, "prices_dam") == 0


def test_missing_hour_and_out_of_range_are_flagged(conn):
    site = FakeSite()
    d = date(2026, 11, 12)
    site.put_daily("dam_price", d, dam_csv(d, drop_hours=(7,), price=lambda h: 99999.0 if h == 3 else 30.0))
    nyiso.run_ingest(conn, S, now=NOW, fetch=site, lookback_days=0)
    codes = {r[0] for r in conn.execute("SELECT code FROM quality_flags WHERE dataset='dam_price'")}
    assert {"incomplete_day", "out_of_range"} <= codes
    assert count(conn, "prices_dam") == 22


def test_dst_days_have_23_and_25_hours_without_incomplete_flag(conn):
    site = FakeSite()
    for d in (date(2026, 3, 8), date(2026, 11, 1)):
        site.put_daily("dam_price", d, dam_csv(d))
    for d in (date(2026, 3, 8), date(2026, 11, 1)):
        nyiso.store_file(conn, S, "dam_price", site(nyiso.daily_url("dam_price", d)), granularity="daily", delivery=d, now=NOW)
    assert count(conn, "prices_dam") == 23 + 25
    assert conn.execute("SELECT COUNT(*) FROM quality_flags WHERE code IN ('incomplete_day','overfull_day')").fetchone()[0] == 0


def test_revised_file_updates_values_and_records_vintage(conn):
    site = FakeSite()
    d = date(2026, 11, 12)
    site.put_daily("actual_load", d, load_csv(d, load=lambda h: 5000.0))
    nyiso.run_ingest(conn, S, now=NOW, fetch=site, lookback_days=0)
    site.put_daily("actual_load", d, load_csv(d, load=lambda h: 5100.0 if h == 4 else 5000.0),
                   last_modified=datetime(2026, 11, 13, 17, 25, tzinfo=UTC))
    nyiso.run_ingest(conn, S, now=NOW, fetch=site, lookback_days=0)
    assert count(conn, "raw_files") >= 2 and count(conn, "load_actual") == 24
    v = conn.execute("SELECT load_mw FROM load_actual ORDER BY ts_utc LIMIT 1 OFFSET 4").fetchone()[0]
    assert v == 5100.0
    assert conn.execute("SELECT COUNT(*) FROM quality_flags WHERE code='revised_values'").fetchone()[0] == 1


def test_window_crossing_a_month_boundary_uses_the_monthly_zip_for_the_earlier_month(conn):
    site = FakeSite()
    oct_days = {f"d{k}.csv": dam_csv(date(2026, 10, 30 + k)) for k in (0, 1)}
    site.put_monthly("dam_price", 2026, 10, zip_of(oct_days))
    populate_days(site, date(2026, 11, 1), date(2026, 11, 3))
    now = datetime(2026, 11, 2, 15, 0, tzinfo=UTC)
    nyiso.run_ingest(conn, S, now=now, fetch=site, lookback_days=3)
    urls = [u for u in site.calls if "zone_csv.zip" in u and "20261001" in u]
    assert len(urls) == 1                                             # one monthly zip fetch, not one per day
    assert count(conn, "prices_dam") == 48 + 25 + 24 + 24             # Oct 30-31 (zip) + Nov 1 (25 h: fall-back day) + Nov 2-3


def test_backfill_loads_completed_months_skips_current_and_is_idempotent(conn):
    site = FakeSite()
    site.put_monthly("dam_price", 2026, 9, zip_of({"a.csv": dam_csv(date(2026, 9, 1)), "b.csv": dam_csv(date(2026, 9, 2))}))
    site.put_monthly("actual_load", 2026, 9, zip_of({"a.csv": load_csv(date(2026, 9, 1))}))
    out = nyiso.run_backfill(conn, S, start="2026-09", end="2026-11", fetch=site, now=NOW)
    assert count(conn, "prices_dam") == 48 and count(conn, "load_actual") == 24
    assert not any("20261101" in u for u in site.calls)               # current month is not backfilled from zips
    again = nyiso.run_backfill(conn, S, start="2026-09", end="2026-11", fetch=site, now=NOW)
    assert again["counts"].get("dam_price:unchanged") == 1 and count(conn, "prices_dam") == 48
    assert out["status"] == "success"


def test_expected_availability_times():
    assert not nyiso.expected_available("dam_price", date(2026, 11, 13), datetime(2026, 11, 12, 14, 0, tzinfo=UTC), S)   # 09:00 ET
    assert nyiso.expected_available("dam_price", date(2026, 11, 13), datetime(2026, 11, 12, 14, 40, tzinfo=UTC), S)      # 09:40 ET
    assert not nyiso.expected_available("actual_load", date(2026, 11, 12), datetime(2026, 11, 13, 5, 0, tzinfo=UTC), S)  # 00:00 ET
    assert nyiso.expected_available("actual_load", date(2026, 11, 12), datetime(2026, 11, 13, 7, 0, tzinfo=UTC), S)


# ---- research downloader: stale-cache fix ---------------------------------------------------
def test_download_month_refetches_an_incomplete_month_but_reuses_a_complete_one(tmp_path):
    calls = []

    class Sess:
        def get(self, url, timeout=None):
            calls.append(url)
            r = requests.Response()
            r.status_code, r._content = 200, b"PK" + bytes(len(calls))
            return r

    cur = pd.Timestamp("2026-11-12")
    dl.download_month("damlbmp_zone", 2026, 11, tmp_path, session=Sess(), today=cur)
    dl.download_month("damlbmp_zone", 2026, 11, tmp_path, session=Sess(), today=cur)
    assert len(calls) == 2                                             # current month: never served from cache
    dl.download_month("damlbmp_zone", 2026, 9, tmp_path, session=Sess(), today=cur)
    dl.download_month("damlbmp_zone", 2026, 9, tmp_path, session=Sess(), today=cur)
    assert len(calls) == 3                                             # completed month: cached after the first fetch


# ---- weather --------------------------------------------------------------------------------
def om_body(times, vals, var="temperature_2m"):
    return json.dumps({"hourly": {"time": times, var: vals}}).encode()


def fake_fetch(body):
    from energy_service.http import Fetched

    def f(url):
        f.urls.append(url)
        return Fetched(body, None, url)
    f.urls = []
    return f


def test_weather_snapshot_stores_nulls_as_null_and_records_retrieval_time(conn):
    times = [f"2026-11-13T{h:02d}:00" for h in range(24)]
    vals = [5.0] * 24
    vals[3] = None
    sid = weather.snapshot_live_forecast(conn, S, now=NOW, fetch=fake_fetch(om_body(times, vals)))
    assert sid and conn.execute("SELECT retrieved_at, kind, n_hours FROM weather_snapshots WHERE id=?", (sid,)).fetchone()[:] \
        == ("2026-11-12T15:00:00Z", "live_forecast", 24)
    assert conn.execute("SELECT COUNT(*) FROM weather_hours WHERE temp_c IS NULL").fetchone()[0] == 1


def test_weather_snapshot_failure_is_logged_and_returns_none(conn):
    def boom(url):
        raise FetchError("down")
    assert weather.snapshot_live_forecast(conn, S, now=NOW, fetch=boom) is None
    assert conn.execute("SELECT status FROM job_runs WHERE job='weather_snapshot'").fetchone()[0] == "failed"
    assert conn.execute("SELECT COUNT(*) FROM events WHERE code='weather_snapshot_failed'").fetchone()[0] == 1
    bad = fake_fetch(b'{"oops": 1}')
    assert weather.snapshot_live_forecast(conn, S, now=NOW, fetch=bad) is None


def test_previous_runs_backfill_one_request_per_year(conn):
    f = fake_fetch(om_body(["2022-12-31T00:00", "2023-01-01T00:00"], [1.0, 2.0], "temperature_2m_previous_day2"))
    ids = weather.backfill_previous_runs(conn, S, start="2022-12-01", end="2023-02-01", lead_days=2, now=NOW, fetch=f)
    assert len(ids) == 2 and len(f.urls) == 2 and "previous_day2" in f.urls[0] and "gfs_global" in f.urls[0]


def test_expired_daily_file_is_recovered_from_the_current_month_zip(conn):
    """NYISO's daily files are a rolling ~12-day window; the current month's zip still has the day."""
    site = FakeSite()
    populate_days(site, date(2026, 11, 2), date(2026, 11, 13), load_end=date(2026, 11, 11))
    site.put_monthly("dam_price", 2026, 11, zip_of({f"d{k}.csv": dam_csv(date(2026, 11, k)) for k in (1, 2)}))
    site.put_monthly("actual_load", 2026, 11, zip_of({f"d{k}.csv": load_csv(date(2026, 11, k)) for k in (1, 2)}))
    out = nyiso.run_ingest(conn, S, now=NOW, fetch=site, lookback_days=11)       # window starts 2026-11-01
    assert out["counts"]["dam_price:recovered_from_monthly_zip"] == 1
    assert "dam_price:expected_missing" not in out["counts"]
    assert nyiso.day_present(conn, "dam_price", S.zone, date(2026, 11, 1))
    assert conn.execute("SELECT COUNT(*) FROM quality_flags WHERE code='expected_file_missing'").fetchone()[0] == 0
    # idempotent: a second run changes nothing
    n = count(conn, "prices_dam")
    nyiso.run_ingest(conn, S, now=NOW, fetch=site, lookback_days=11)
    assert count(conn, "prices_dam") == n


def test_day_missing_from_both_daily_file_and_monthly_zip_is_flagged(conn):
    site = FakeSite()
    populate_days(site, date(2026, 11, 2), date(2026, 11, 13), load_end=date(2026, 11, 11))
    site.put_monthly("dam_price", 2026, 11, zip_of({"d2.csv": dam_csv(date(2026, 11, 2))}))     # no Nov 1 here either
    out = nyiso.run_ingest(conn, S, now=NOW, fetch=site, lookback_days=11)
    assert out["counts"]["dam_price:expected_missing"] == 1
    flags = conn.execute("SELECT delivery_date FROM quality_flags WHERE code='expected_file_missing' AND dataset='dam_price'").fetchall()
    assert [f[0] for f in flags] == ["2026-11-01"]
