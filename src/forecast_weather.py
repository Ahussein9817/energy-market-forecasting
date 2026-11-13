"""Day-ahead forecast temperature (Open-Meteo Previous Runs API, GFS global model).

`temperature_2m_previous_dayN` at valid time t is the value forecast N days before t. It is a
genuine archived forecast, unlike the stitched `temperature_2m` series of the Forecast or
Historical Forecast APIs, which must not be used for an operational claim.

Caveats (see docs/data-sources.md):
  * The run initialisation time behind `previous_day1` is not documented. `lead_days=2` is the
    leakage-safe sensitivity: it is certainly available before the end-of-day-D cutoff.
  * GFS coverage has holes (one 492 h outage 2023-12-30..2024-01-19 UTC). Missing hours stay NaN;
    days that cannot be fully built are EXCLUDED, never imputed or filled with actuals.

Weather data by Open-Meteo.com (CC BY 4.0). The free API is non-commercial; do not commit bulk raw data.
"""

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from src.data_loader import NYC_LAT, NYC_LON, DataError
from src.panel import HOURS, TZ

logger = logging.getLogger(__name__)

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "gfs_global"  # not gfs_seamless, which can blend in HRRR for the first days


def fetch_previous_runs(start: str, end: str, lead_days: int = 1, model: str = MODEL,
                        lat: float = NYC_LAT, lon: float = NYC_LON,
                        session: requests.Session | None = None, timeout: int = 60,
                        retries: int = 3, backoff: float = 2.0) -> pd.DataFrame:
    """Hourly forecast 2 m temperature (deg C, UTC) issued `lead_days` before the valid time.
    Hours the API reports as null are returned as NaN (never filled)."""
    var = f"temperature_2m_previous_day{lead_days}"
    params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
              "hourly": var, "models": model, "timezone": "UTC"}
    get = (session or requests).get
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            r = get(PREVIOUS_RUNS_URL, params=params, timeout=timeout)
            r.raise_for_status()
            h = r.json().get("hourly")
            if not h or "time" not in h or var not in h:
                raise DataError(f"Open-Meteo previous-runs response missing {var}")
            return pd.DataFrame({"timestamp_utc": pd.to_datetime(h["time"], utc=True),
                                 "temp_fc_c": pd.to_numeric(pd.Series(h[var]), errors="coerce")})
        except (requests.RequestException, DataError) as e:
            last = e
            logger.warning("previous-runs %s..%s attempt %d/%d failed: %s", start, end, attempt, retries, e)
            if attempt < retries:
                time.sleep(backoff * 2 ** (attempt - 1))
    raise DataError(f"previous-runs {start}..{end}: failed after {retries} attempts: {last}")


def fetch_range(start: str, end: str, lead_days: int, cache_dir: Path, **kw) -> pd.DataFrame:
    """One request per calendar year (<= 366 days), cached under cache_dir (gitignored)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    parts = []
    for yr in range(s.year, e.year + 1):
        lo, hi = max(s, pd.Timestamp(yr, 1, 1)), min(e, pd.Timestamp(yr, 12, 31))
        f = cache_dir / f"gfs_previous_day{lead_days}_{lo.date()}_{hi.date()}.parquet"
        if f.exists():
            parts.append(pd.read_parquet(f))
            continue
        df = fetch_previous_runs(str(lo.date()), str(hi.date()), lead_days=lead_days, **kw)
        df.to_parquet(f)
        parts.append(df)
    return pd.concat(parts, ignore_index=True).drop_duplicates("timestamp_utc").sort_values("timestamp_utc")


def to_forecast_panel(df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Local-date x hour panel over `index`, NaN where the forecast is missing.
    The only cells filled are local hours that do not exist (the spring-forward hour), linearly from
    their neighbours, as in src.panel. A forecast hole the API reports is never filled."""
    local = df["timestamp_utc"].dt.tz_convert(TZ)
    t = pd.DataFrame({"date": local.dt.tz_localize(None).dt.normalize(), "hour": local.dt.hour,
                      "v": df["temp_fc_c"].to_numpy()})
    wide = t.groupby(["date", "hour"])["v"].mean().unstack("hour").reindex(columns=HOURS)
    wide = wide.reindex(index)
    nonexistent = pd.DataFrame(
        {h: [pd.isna(pd.Timestamp(d + pd.Timedelta(hours=h)).tz_localize(TZ, ambiguous=True, nonexistent="NaT"))
             for d in index] for h in HOURS}, index=index)
    filled = wide.interpolate(axis=1, limit=1, limit_direction="both")
    wide = wide.where(~nonexistent, filled)
    wide.index.name = "date"
    return wide


def missing_days(panel: pd.DataFrame) -> pd.DatetimeIndex:
    """Days with at least one missing forecast hour (these are excluded, not imputed)."""
    return panel.index[panel.isna().any(axis=1)]
