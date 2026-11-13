"""Analyst dashboard for the day-ahead forecast service (NYISO N.Y.C. zone). Reads the SQLite database read-only.

Run:  ENERGY_DB_PATH=data/energy.sqlite streamlit run dashboard/app.py
Questions it answers: what is tomorrow's forecast and where did it come from; how reliable have forecasts been and
does the model beat the baseline; are there data problems; what happened on a given day. A local demonstration,
not trading advice and not a production system.
"""
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from energy_service import db, monitor, queries  # noqa: E402
from energy_service.config import Settings  # noqa: E402

ICON = {"ok": "✅", "warn": "⚠️", "fail": "❌", "unknown": "❔"}
WHY = {
    "no_active_model": "no model is active, so the baseline is the forecast",
    "model_not_eligible": "the model's backtest evidence does not meet the serving rule",
    "weather_missing": "the weather forecast was missing for some target hours",
    "weather_stale": "the latest weather snapshot was too old",
    "load_missing": "recent actual load was missing",
    "model_load_failed": "the model file could not be loaded or failed its hash check",
    "feature_version_mismatch": "the model's feature definition does not match the code",
    "features_incomplete": "some model inputs were missing for the target day",
}

st.set_page_config(page_title="Electricity forecast monitor", layout="wide")
s = Settings.from_env()
now = datetime.now(UTC)


@st.cache_resource(show_spinner=False)
def _noop():
    return None


def open_db():
    try:
        return db.connect(s.db_path, read_only=True)
    except sqlite3.Error:
        return None


conn = open_db()
st.title("Day-ahead electricity forecast: NYISO N.Y.C. zone")
st.caption("Demonstration system. The baseline repeats the price of day D for D+1; a model serves only when its backtest "
           "evidence qualifies it and its inputs are complete. Not trading advice; no operating history beyond what is shown.")

if conn is None or db.schema_version(conn) is None:
    st.error(f"No database at `{s.db_path}`. Create it with `python -m energy_service db-init` and load data "
             f"(see docs/operational-service.md).")
    st.stop()

checks = monitor.run_checks(conn, s, now)
overall = monitor.overall(checks)
c1, c2, c3 = st.columns([1, 1, 2])
c1.metric("System status", f"{ICON[overall]} {overall}")
latest = queries.latest_forecast_row(conn)
c2.metric("Latest forecast for", latest["target_date"] if latest else "none yet")
bad = [c for c in checks if c.status in ("warn", "fail")]
c3.write("**Attention:** " + ("; ".join(f"{c.name}: {c.message}" for c in bad) if bad else "nothing flagged"))

tab_tom, tab_perf, tab_health, tab_day = st.tabs(["Tomorrow", "Performance", "Data health", "Investigate a day"])

# ------------------------------------------------------------------------------------------- Tomorrow
with tab_tom:
    if latest is None:
        st.info("No forecast has been generated yet. Run `python -m energy_service forecast`.")
    else:
        fid = latest["id"]
        hours = queries.forecast_hours(conn, fid)
        inputs = json.loads(latest["input_json"])
        gen = pd.Timestamp(latest["generated_at"]).tz_convert("America/New_York")
        a, b, c, d = st.columns(4)
        a.metric("Target day", latest["target_date"])
        b.metric("Served by", "model" if latest["served_kind"] == "model" else "baseline")
        c.metric("Generated (ET)", gen.strftime("%a %H:%M"))
        d.metric("Version", latest["version"])
        if latest["fallback_reason"]:
            st.warning(f"Baseline served: {WHY.get(latest['fallback_reason'], latest['fallback_reason'])}.")
        if latest["late"]:
            st.info("Generated after the day-ahead prices were public (about 09:32 ET): shown for completeness, "
                    "excluded from performance statistics.")
        chart = hours.set_index("hour")[["baseline", "model", "served"]].rename(
            columns={"baseline": "baseline (price of day D)", "model": "model", "served": "served"})
        ev = queries.day_detail(conn, latest["target_date"], s.zone)
        if ev and ev["evaluation"]:
            chart["actual (published)"] = ev["hours"].set_index("hour")["actual"]
        st.line_chart(chart, y_label="$/MWh", x_label="local hour of the target day")
        if ev and ev["evaluation"]:
            e = ev["evaluation"]
            st.success(f"Scored: served MAE ${e['mae_served']:.2f}, baseline MAE ${e['mae_baseline']:.2f} on this day. "
                       f"One day says little about reliability; see the Performance tab.")
        else:
            st.caption("Actual prices for the target day are published about 09:32 ET on the day before; the forecast is scored then.")
        with st.expander("How this forecast was made"):
            st.json({k: inputs.get(k) for k in ("model_id", "feature_version", "weather_snapshot_id", "weather_retrieved_at",
                                                "weather_age_hours", "feature_hash", "baseline", "origin_nominal")})
        st.dataframe(hours.round(2), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------------------------------- Performance
with tab_perf:
    days = st.radio("Window", [14, 28, 90], index=1, horizontal=True, format_func=lambda x: f"last {x} days")
    ev = queries.evaluation_frame(conn, days, today=now.replace(tzinfo=None))
    h = queries.hourly_errors(conn, days, today=now.replace(tzinfo=None))
    st.caption("Only forecasts generated before the day-ahead prices were public, and not replays, are counted.")
    if len(ev) < s.perf_min_days:
        st.info(f"{len(ev)} evaluated day(s) in this window; at least {s.perf_min_days} are needed before judging "
                f"reliability. Nothing below is evidence of performance yet.")
    if ev.empty:
        st.write("No evaluated forecasts in this window.")
    else:
        st.subheader("Is the served forecast beating the baseline?")
        st.dataframe(queries.performance_table(h).round(2), hide_index=True, use_container_width=True)
        daily = ev.set_index("target_date")[["mae_served", "mae_baseline"]].rename(
            columns={"mae_served": "served MAE", "mae_baseline": "baseline MAE"})
        st.subheader("Forecast error over time (daily MAE, $/MWh)")
        st.line_chart(daily)
        st.subheader("Which hours have the largest errors?")
        byh = h.assign(served_err=(h["served"] - h["actual"]).abs(), base_err=(h["baseline"] - h["actual"]).abs()) \
            .groupby("hour")[["served_err", "base_err"]].mean().rename(columns={"served_err": "served", "base_err": "baseline"})
        st.bar_chart(byh, y_label="MAE $/MWh", x_label="local hour")
        thr = queries.price_quantile(conn, s.zone, 0.95)
        if thr is not None and not h.empty:
            sp = queries.spike_analysis(h, thr)
            st.subheader("Price spikes")
            if sp["hours"] == 0:
                st.write(f"No evaluated hour reached the spike threshold (${thr:.0f}/MWh, the 95th percentile of the last year).")
            else:
                st.write(f"{sp['hours']} evaluated hours at or above ${thr:.0f}/MWh: served MAE ${sp['served_mae']:.2f}, "
                         f"baseline MAE ${sp['baseline_mae']:.2f}, served bias {sp['served_bias']:+.2f}. "
                         f"{'Few hours: treat with caution.' if sp['hours'] < 30 else ''}")

# -------------------------------------------------------------------------------------------- Health
with tab_health:
    st.subheader("Checks")
    st.dataframe(pd.DataFrame([{"": ICON[c.status], "check": c.name, "status": c.status, "detail": c.message} for c in checks]),
                 hide_index=True, use_container_width=True)
    st.subheader("Day-ahead prices, last 14 days")
    ph = queries.price_history(conn, s.zone, 14)
    if not ph.empty:
        st.line_chart(ph.assign(ts=ph["ts"].dt.tz_localize(None)).set_index("ts")["lbmp"], y_label="$/MWh")
    st.subheader("Data-quality flags, last 48 hours")
    fl = queries.recent_flags(conn, now, 48)
    st.dataframe(fl, hide_index=True, use_container_width=True) if len(fl) else st.write("None.")
    c1, c2 = st.columns(2)
    c1.subheader("Recent jobs")
    c1.dataframe(pd.DataFrame(queries.recent_jobs(conn, 10)), hide_index=True, use_container_width=True)
    c2.subheader("Recent events")
    c2.dataframe(pd.DataFrame(queries.recent_events(conn, 10)), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------------------------------ Investigate
with tab_day:
    dates = [r[0] for r in conn.execute("SELECT DISTINCT target_date FROM forecasts ORDER BY target_date DESC LIMIT 120")]
    if not dates:
        st.info("No forecasts stored yet.")
    else:
        pick = st.selectbox("Target day", dates)
        d = queries.day_detail(conn, pick, s.zone)
        f = d["forecast"]
        st.write(f"**{pick}**: served by **{f['served_kind']}** (version {f['version']}, source {f['source']}), generated {f['generated_at']}"
                 + ("; late" if f["late"] else "") + (f"; fallback: {WHY.get(f['fallback_reason'], f['fallback_reason'])}" if f["fallback_reason"] else ""))
        hrs = d["hours"].copy()
        cols = [c for c in ("baseline", "model", "served", "actual") if hrs[c].notna().any()]
        st.line_chart(hrs.set_index("hour")[cols], y_label="$/MWh")
        if hrs["actual"].notna().any():
            hrs["served error"] = hrs["served"] - hrs["actual"]
            hrs["baseline error"] = hrs["baseline"] - hrs["actual"]
            st.write("Largest served errors first:")
            st.dataframe(hrs.reindex(hrs["served error"].abs().sort_values(ascending=False).index).round(2).head(8),
                         hide_index=True, use_container_width=True)
        else:
            st.caption("Not scored yet: actual prices for this day are not stored.")
        st.write("Inputs and provenance")
        st.json(d["inputs"])
        st.write("Data-quality flags for the origin and target days")
        st.dataframe(pd.DataFrame(d["flags"]), hide_index=True, use_container_width=True) if d["flags"] else st.write("None.")
