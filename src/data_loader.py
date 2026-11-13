"""Real-data ingestion: NYISO day-ahead zonal prices + integrated load, Open-Meteo temperature.

Sources (all key-free, see docs/data-sources.md):
  * NYISO day-ahead LBMP by zone  https://mis.nyiso.com/public/csv/damlbmp/
  * NYISO integrated actual load   https://mis.nyiso.com/public/csv/palIntegrated/
  * Open-Meteo historical weather  https://archive-api.open-meteo.com/v1/archive

All timestamps are converted to UTC. NYISO price files carry local (Eastern)
timestamps with NO timezone column, so daylight-saving transitions are resolved
from row order (the repeated 01:00 on fall-back day appears twice, in order).
"""

import io
import logging
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

NYISO_BASE = "https://mis.nyiso.com/public/csv"
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
NYC_LAT, NYC_LON = 40.7128, -74.0060
TZ = "America/New_York"


class DataError(Exception):
    """Download failed or a file did not have the expected structure."""


def nyiso_url(dataset: str, year: int, month: int) -> str:
    """Monthly zip containing one CSV per day. dataset: 'damlbmp_zone' or 'palIntegrated'."""
    folder = "damlbmp" if dataset == "damlbmp_zone" else dataset
    return f"{NYISO_BASE}/{folder}/{year:04d}{month:02d}01{dataset}_csv.zip"


def month_is_complete(year: int, month: int, today=None) -> bool:
    """True once the month ended at least 3 days ago (NYISO then no longer appends to its zip)."""
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    end = pd.Period(f"{year:04d}-{month:02d}", "M").end_time.normalize()
    return today.normalize() >= end + pd.Timedelta(days=3)


def download_month(dataset: str, year: int, month: int, cache_dir: Path,
                   retries: int = 3, backoff: float = 2.0, timeout: int = 60,
                   session: requests.Session | None = None, today=None) -> Path:
    """Download (or reuse from cache) one monthly zip. A cached zip is reused only for a completed
    month; the current month is always re-fetched. Raises DataError after `retries`."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{dataset}_{year:04d}{month:02d}.zip"
    if dest.exists() and dest.stat().st_size > 0 and month_is_complete(year, month, today):
        return dest  # an incomplete (current) month is re-downloaded: NYISO updates its zip daily
    get = (session or requests).get
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            r = get(nyiso_url(dataset, year, month), timeout=timeout)
            r.raise_for_status()
            if not r.content.startswith(b"PK"):
                raise DataError(f"response is not a zip file ({len(r.content)} bytes)")
            dest.write_bytes(r.content)
            return dest
        except (requests.RequestException, DataError) as e:
            last = e
            logger.warning("%s %d-%02d attempt %d/%d failed: %s", dataset, year, month, attempt, retries, e)
            if attempt < retries:
                time.sleep(backoff * 2 ** (attempt - 1))
    raise DataError(f"{dataset} {year}-{month:02d}: failed after {retries} attempts: {last}")


def _read_zip_csvs(zip_path: Path | io.BytesIO) -> list[pd.DataFrame]:
    frames = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(n for n in zf.namelist() if n.lower().endswith(".csv")):
            with zf.open(name) as f:
                frames.append(pd.read_csv(f))
    if not frames:
        raise DataError("zip contains no CSV files")
    return frames


def parse_dam_lbmp(zip_path) -> pd.DataFrame:
    """-> columns: timestamp_utc, zone, lbmp, mcl, mcc ($/MWh). Duplicate 01:00 on
    fall-back day is resolved by row order within each zone and day file."""
    out = []
    for df in _read_zip_csvs(zip_path):
        need = {"Time Stamp", "Name", "LBMP ($/MWHr)"}
        if not need <= set(df.columns):
            raise DataError(f"unexpected LBMP columns: {list(df.columns)}")
        for zone, g in df.groupby("Name", sort=False):
            ts = pd.to_datetime(g["Time Stamp"], format="%m/%d/%Y %H:%M")
            local = ts.dt.tz_localize(TZ, ambiguous="infer", nonexistent="shift_forward")
            out.append(pd.DataFrame({
                "timestamp_utc": local.dt.tz_convert("UTC").to_numpy(),
                "zone": zone,
                "lbmp": g["LBMP ($/MWHr)"].to_numpy(dtype=float),
                "mcl": g["Marginal Cost Losses ($/MWHr)"].to_numpy(dtype=float),
                "mcc": g["Marginal Cost Congestion ($/MWHr)"].to_numpy(dtype=float),
            }))
    return pd.concat(out, ignore_index=True)


_OFFSET_H = {"EST": 5, "EDT": 4}


def parse_integrated_load(zip_path) -> pd.DataFrame:
    """-> timestamp_utc, zone, load_mw. Uses the file's own EST/EDT column, so DST is exact."""
    out = []
    for df in _read_zip_csvs(zip_path):
        if not {"Time Stamp", "Time Zone", "Name", "Integrated Load"} <= set(df.columns):
            raise DataError(f"unexpected load columns: {list(df.columns)}")
        bad = set(df["Time Zone"]) - set(_OFFSET_H)
        if bad:
            raise DataError(f"unknown time zone labels: {bad}")
        local = pd.to_datetime(df["Time Stamp"], format="%m/%d/%Y %H:%M:%S")
        utc = local + pd.to_timedelta(df["Time Zone"].map(_OFFSET_H), unit="h")
        out.append(pd.DataFrame({"timestamp_utc": utc.dt.tz_localize("UTC"), "zone": df["Name"],
                                 "load_mw": df["Integrated Load"].astype(float)}))
    return pd.concat(out, ignore_index=True)


def fetch_open_meteo(start: str, end: str, lat: float = NYC_LAT, lon: float = NYC_LON,
                     session: requests.Session | None = None, timeout: int = 60) -> pd.DataFrame:
    """Hourly 2 m temperature (deg C) in UTC for [start, end] (YYYY-MM-DD)."""
    params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
              "hourly": "temperature_2m", "timezone": "UTC"}
    r = (session or requests).get(OPEN_METEO_URL, params=params, timeout=timeout)
    r.raise_for_status()
    h = r.json().get("hourly")
    if not h or "time" not in h or "temperature_2m" not in h:
        raise DataError("Open-Meteo response missing hourly temperature_2m")
    return pd.DataFrame({"timestamp_utc": pd.to_datetime(h["time"], utc=True),
                         "temp_c": pd.to_numeric(pd.Series(h["temperature_2m"]), errors="coerce")})


def months(start: str, end: str):
    for p in pd.period_range(start, end, freq="M"):
        yield p.year, p.month
