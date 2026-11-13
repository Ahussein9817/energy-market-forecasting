"""Synthetic NYISO-format files and a fake site for offline tests (no real NYISO data is committed)."""
from __future__ import annotations

import io
import zipfile
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from energy_service.http import Fetched, FetchError, NotYetPublished
from energy_service.ingest import nyiso

DAM_HEAD = "Time Stamp,Name,PTID,LBMP ($/MWHr),Marginal Cost Losses ($/MWHr),Marginal Cost Congestion ($/MWHr)\n"
LOAD_HEAD = "Time Stamp,Time Zone,Name,PTID,Integrated Load\n"


def local_hours(day: date) -> list[pd.Timestamp]:
    d = pd.Timestamp(day)
    return list(pd.date_range(d, d + pd.Timedelta(days=1), freq="h", tz="America/New_York", inclusive="left"))


def dam_csv(day: date, price=lambda h: 30.0 + h, zones=("N.Y.C.", "WEST"), drop_hours=(), extra_rows="") -> bytes:
    rows = []
    for z in zones:
        for h, t in enumerate(local_hours(day)):
            if h in drop_hours:
                continue
            rows.append(f"{t.strftime('%m/%d/%Y %H:%M')},{z},1,{price(h) if z == 'N.Y.C.' else 20.0},1.0,0.5\n")
    return (DAM_HEAD + "".join(rows) + extra_rows).encode()


def load_csv(day: date, load=lambda h: 5000.0 + 10 * h, zone="N.Y.C.") -> bytes:
    rows = []
    for h, t in enumerate(local_hours(day)):
        rows.append(f"{t.strftime('%m/%d/%Y %H:%M:%S')},{t.tzname()},{zone},1,{load(h)}\n")
    return (LOAD_HEAD + "".join(rows)).encode()


def zip_of(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for n, c in files.items():
            zf.writestr(n, c)
    return buf.getvalue()


class FakeSite:
    """Callable fetch(url) -> Fetched backed by a dict. Missing URL -> NotYetPublished; value Exception -> raised."""

    def __init__(self):
        self.files: dict[str, object] = {}
        self.calls: list[str] = []

    def put_daily(self, dataset, day, body, last_modified=None):
        self.files[nyiso.daily_url(dataset, day)] = (body, last_modified)

    def put_monthly(self, dataset, year, month, body, last_modified=None):
        self.files[nyiso.monthly_url(dataset, year, month)] = (body, last_modified)

    def fail(self, url, exc=None):
        self.files[url] = exc or FetchError(f"{url}: HTTP 503")

    def __call__(self, url):
        self.calls.append(url)
        v = self.files.get(url)
        if v is None:
            raise NotYetPublished(url)
        if isinstance(v, Exception):
            raise v
        body, lm = v
        return Fetched(body, lm or datetime(2026, 11, 12, 14, 33, tzinfo=UTC), url)


def populate_days(site: FakeSite, start: date, end: date, load_end: date | None = None):
    d = start
    while d <= end:
        site.put_daily("dam_price", d, dam_csv(d))
        if load_end is None or d <= load_end:
            site.put_daily("actual_load", d, load_csv(d))
        d += timedelta(days=1)


from energy_service.demo import add_live_snapshot, build_world  # noqa: E402,F401  (synthetic world lives in the package)


def seed_evaluated(conn, end: date, days: int, served_mae=3.0, baseline_mae=5.0, kind="model", late=0, source="scheduled",
                   fallback=None, start_id=1000, version=1):
    """Insert `days` evaluated forecasts (target dates end-days+1 .. end) with controlled errors, bypassing the jobs.
    Hour-level rows are consistent with the stored MAEs (constant absolute error)."""
    import json as _json
    for k in range(days):
        t = end - timedelta(days=days - 1 - k)
        fid = start_id + k + (0 if source != "replay" else 5000)
        actual = [40.0 + h for h in range(24)]
        sv = [a + served_mae for a in actual]
        bv = [a + baseline_mae for a in actual]
        conn.execute("INSERT INTO forecasts (id, origin_date, target_date, version, origin_ts, generated_at, source, served_kind, "
                     "served_model_id, fallback_reason, late, input_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                     (fid, str(t - timedelta(days=1)), str(t), version, "x", "2026-01-01T00:00:00Z", source, kind, None, fallback, late,
                      _json.dumps({})))
        conn.executemany("INSERT INTO forecast_hours (forecast_id, hour, baseline, model, served) VALUES (?,?,?,?,?)",
                         [(fid, h, bv[h], sv[h] if kind == "model" else None, sv[h] if kind == "model" else bv[h]) for h in range(24)])
        conn.executemany("INSERT INTO evaluation_hours (forecast_id, hour, actual) VALUES (?,?,?)", [(fid, h, actual[h]) for h in range(24)])
        ms = served_mae if kind == "model" else baseline_mae
        conn.execute("INSERT INTO evaluations (forecast_id, evaluated_at, target_date, mae_served, mae_baseline, mae_model, bias_served) "
                     "VALUES (?,?,?,?,?,?,?)", (fid, "2026-01-02T00:00:00Z", str(t), ms, baseline_mae,
                                                served_mae if kind == "model" else None, ms if kind == "model" else baseline_mae))
    conn.commit()
