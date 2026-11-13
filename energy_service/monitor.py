"""Monitoring of the scheduled pipeline: data freshness, job health, forecast availability and quality,
performance against the baseline, and basic drift. `run_checks` only reads; `record_snapshot` stores the result
and emits an event when a check changes status. This monitors a scheduled demonstration pipeline; replayed or
backtest rows are never mixed in (see queries.evaluation_frame)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from energy_service import db, queries, registry, status
from energy_service.config import Settings
from energy_service.db import iso, log_event
from energy_service.ingest.nyiso import day_present

ET = ZoneInfo("America/New_York")
ORDER = {"ok": 0, "unknown": 1, "warn": 2, "fail": 3}


@dataclass
class Check:
    name: str
    status: str            # ok | warn | fail | unknown
    message: str
    value: dict | None = None


def overall(checks: list[Check]) -> str:
    worst = max((c.status for c in checks), key=lambda x: ORDER[x], default="unknown")
    return "ok" if worst == "unknown" and all(c.status in ("ok", "unknown") for c in checks) and any(
        c.status == "ok" for c in checks) else worst


def _age_hours(ts: str, now: datetime) -> float:
    return (now - pd.Timestamp(ts).to_pydatetime()).total_seconds() / 3600


def run_checks(conn, s: Settings, now: datetime) -> list[Check]:
    r = status.readiness(conn, s, now)
    now_et = now.astimezone(ET)
    out: list[Check] = []

    # data freshness
    out.append(Check("prices_fresh", "fail" if r["reasons"] else "ok",
                     "; ".join(r["reasons"]) or f"latest price day {r['latest_price_date']}", {"latest": r.get("latest_price_date")}))
    stale_load = [d for d in r["degraded"] if d.startswith(("load_stale", "no_load"))]
    out.append(Check("load_fresh", "warn" if stale_load else "ok", "; ".join(stale_load) or f"latest load day {r.get('latest_load_date')}"))

    # job health
    last = conn.execute("SELECT status, started_at, finished_at FROM job_runs WHERE job='ingest' ORDER BY id DESC LIMIT 1").fetchone()
    ok_job = conn.execute("SELECT finished_at FROM job_runs WHERE job='ingest' AND status='success' ORDER BY id DESC LIMIT 1").fetchone()
    if last is None:
        out.append(Check("ingest_recent", "unknown", "no ingest run recorded"))
    elif ok_job is None or _age_hours(ok_job["finished_at"], now) > s.job_max_age_hours:
        out.append(Check("ingest_recent", "fail", f"no successful ingest in the last {s.job_max_age_hours:.0f} h (last status {last['status']})"))
    elif last["status"] != "success":
        out.append(Check("ingest_recent", "warn", f"latest ingest {last['status']}"))
    else:
        out.append(Check("ingest_recent", "ok", f"last success {_age_hours(ok_job['finished_at'], now):.1f} h ago"))

    snap = conn.execute("SELECT retrieved_at FROM weather_snapshots WHERE kind='live_forecast' ORDER BY retrieved_at DESC LIMIT 1").fetchone()
    if snap is None:
        out.append(Check("weather_snapshot_recent", "warn", "no live weather snapshot (model cannot serve)"))
    else:
        age = _age_hours(snap["retrieved_at"], now)
        out.append(Check("weather_snapshot_recent", "ok" if age <= s.job_max_age_hours else "warn", f"latest snapshot {age:.1f} h old",
                         {"age_hours": round(age, 1)}))

    # data quality
    fl = queries.recent_flags(conn, now, 48)
    bad = fl[fl["severity"].isin(["warning", "error"])]
    # a "file missing" flag for a day that has since been recovered is history, not a current problem
    resolved = bad.apply(lambda r: r["code"] == "expected_file_missing" and r["delivery_date"] is not None
                         and day_present(conn, r["dataset"], s.zone, pd.Timestamp(r["delivery_date"]).date()), axis=1) \
        if len(bad) else pd.Series([], dtype=bool)
    bad = bad[~resolved] if len(bad) else bad
    out.append(Check("data_quality_48h", "warn" if len(bad) else "ok",
                     f"{len(bad)} warning/error flags in 48 h" if len(bad) else "no warning or error flags in 48 h",
                     {"by_code": bad["code"].value_counts().to_dict()} if len(bad) else None))

    # forecast availability and fallbacks
    tomorrow = str(now_et.date() + timedelta(days=1))
    f = queries.forecast_for(conn, tomorrow)
    if f is not None:
        out.append(Check("forecast_available", "ok", f"forecast for {tomorrow} generated {f['generated_at']}"))
    elif now_et.time() >= time(6, 0):
        out.append(Check("forecast_available", "fail", f"no forecast for {tomorrow} after 06:00 ET"))
    else:
        out.append(Check("forecast_available", "ok", f"forecast for {tomorrow} not yet due (06:00 ET)"))
    rows = conn.execute("SELECT fallback_reason FROM forecasts f WHERE source!='replay' AND late=0 AND " + queries.LATEST_VERSION +
                        " ORDER BY target_date DESC LIMIT 14").fetchall()
    if len(rows) < 3:
        out.append(Check("forecast_fallbacks", "unknown", f"only {len(rows)} recent non-late forecasts"))
    else:
        reasons = [x[0] for x in rows if x[0]]
        rate = len(reasons) / len(rows)
        out.append(Check("forecast_fallbacks", "warn" if rate > s.fallback_warn_rate else "ok",
                         f"{len(reasons)} of {len(rows)} recent forecasts fell back to the baseline",
                         {"rate": round(rate, 2), "reasons": pd.Series(reasons).value_counts().to_dict() if reasons else {}}))

    # model integrity
    m = conn.execute("SELECT * FROM models WHERE active=1 AND feature_set='OD'").fetchone()
    if m is None:
        out.append(Check("model_integrity", "warn", "no active model: the baseline serves"))
    else:
        try:
            ok = registry.sha256_file(m["path"]) == m["sha256"]
            out.append(Check("model_integrity", "ok" if ok else "fail", "model file hash matches" if ok else "model file hash MISMATCH"))
        except OSError as e:
            out.append(Check("model_integrity", "fail", f"model file unreadable: {e}"))

    # performance (evaluated, non-late, non-replay days only)
    ev = queries.evaluation_frame(conn, 28, today=now_et.replace(tzinfo=None))
    if len(ev) < s.perf_min_days:
        out.append(Check("performance_vs_baseline", "unknown", f"only {len(ev)} evaluated days (need {s.perf_min_days})", {"days": len(ev)}))
    else:
        ratio = float(ev["mae_served"].mean() / ev["mae_baseline"].mean())
        st = "fail" if ratio > s.perf_fail_ratio else ("warn" if ratio > s.perf_warn_ratio else "ok")
        out.append(Check("performance_vs_baseline", st, f"served MAE is {ratio:.2f}x the baseline's over {len(ev)} days",
                         {"ratio": round(ratio, 3), "days": len(ev), "served_mae": float(ev["mae_served"].mean()),
                          "baseline_mae": float(ev["mae_baseline"].mean())}))
        prior = ev.iloc[:-1]["mae_served"]
        if len(prior) >= s.perf_min_days and prior.std() > 0:
            z = float((ev.iloc[-1]["mae_served"] - prior.mean()) / prior.std())
            out.append(Check("error_spike", "warn" if z > s.spike_sigma else "ok",
                             f"latest day's MAE is {z:+.1f} sigma from the prior mean", {"z": round(z, 2)}))
        else:
            out.append(Check("error_spike", "unknown", "not enough evaluated days"))

    # Price-level drift (PSI) was evaluated on real data and NOT adopted: a 28-day window sits in one narrow price regime, so
    # PSI against any multi-year reference was ~1.3 (season-matched too). Performance against the baseline is the signal monitored.
    return out


def record_snapshot(conn, checks: list[Check], now: datetime) -> list[dict]:
    """Store the snapshot; emit an event for every check whose status changed since the previous snapshot."""
    prev = conn.execute("SELECT checks_json FROM monitor_snapshots ORDER BY id DESC LIMIT 1").fetchone()
    before = {c["name"]: c["status"] for c in json.loads(prev["checks_json"])} if prev else {}
    transitions = []
    for c in checks:
        old = before.get(c.name, "ok")           # a first-ever non-ok check is a transition from "ok"
        if c.status != old and not (c.status == "unknown" and old == "ok"):
            transitions.append({"check": c.name, "from": old, "to": c.status, "message": c.message})
    conn.execute("INSERT INTO monitor_snapshots (created_at, status, checks_json) VALUES (?,?,?)",
                 (iso(now), overall(checks), json.dumps([asdict(c) for c in checks])))
    conn.commit()
    for t in transitions:
        level = {"fail": "error", "warn": "warning"}.get(t["to"], "info")
        log_event(conn, level, "monitor_transition", f"{t['check']}: {t['from']} -> {t['to']}: {t['message']}", t, now)
    return transitions


def send_alerts(conn, s: Settings, transitions: list[dict], now: datetime, post=requests.post) -> int:
    """POST a message to the optional webhook for transitions into or out of `fail`. Never raises."""
    if not s.alert_webhook:
        return 0
    sent = 0
    for t in transitions:
        if t["to"] != "fail" and t["from"] != "fail":
            continue
        text = f"[energy-forecast] {t['check']}: {t['from']} -> {t['to']}: {t['message']}"
        try:
            r = post(s.alert_webhook, json={"text": text}, timeout=10)
            r.raise_for_status()
            sent += 1
        except Exception as e:  # alert delivery must never break the job
            log_event(conn, "warning", "alert_failed", f"{type(e).__name__}: {e}", {"check": t["check"]}, now)
    return sent


def run_monitor(conn, s: Settings, now: datetime, post=requests.post) -> dict:
    with db.JobRun(conn, "monitor", now) as job:
        checks = run_checks(conn, s, now)
        transitions = record_snapshot(conn, checks, now)
        sent = send_alerts(conn, s, transitions, now, post)
        job.details = {"status": overall(checks), "transitions": len(transitions), "alerts_sent": sent}
    return {"status": overall(checks), "checks": [asdict(c) for c in checks], "transitions": transitions, "alerts_sent": sent}
