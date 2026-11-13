"""Validation of parsed NYISO frames. Policy:

  * schema problems, non-hourly timestamps, or duplicate timestamps with CONFLICTING values reject the
    whole file (nothing from it is stored);
  * null or out-of-range values are dropped and flagged;
  * identical duplicate rows are dropped (info);
  * a local day with fewer hours than expected (24, or 23/25 on daylight-saving days) is kept but
    flagged `incomplete_day`, so downstream code can refuse to use it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.panel import TZ


class SchemaError(Exception):
    pass


class ConflictError(Exception):
    pass


@dataclass
class Flag:
    code: str
    severity: str            # info | warning | error
    detail: str
    delivery_date: str | None = None


@dataclass
class Validated:
    df: pd.DataFrame
    flags: list[Flag] = field(default_factory=list)


def expected_hours(local_date) -> int:
    d = pd.Timestamp(local_date).tz_localize(None).normalize()
    return len(pd.date_range(d, d + pd.Timedelta(days=1), freq="h", tz=TZ, inclusive="left"))


def validate_frame(df: pd.DataFrame, *, zone: str, value_col: str, lo: float, hi: float) -> Validated:
    need = {"timestamp_utc", "zone", value_col}
    if not need <= set(df.columns):
        raise SchemaError(f"missing columns {sorted(need - set(df.columns))}")
    ts = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    if ts.isna().any():
        raise SchemaError("unparseable timestamps")
    df = df.assign(timestamp_utc=ts)
    df = df[df["zone"] == zone]
    if df.empty:
        raise SchemaError(f"zone {zone!r} not present in file")
    if ((df["timestamp_utc"].dt.minute != 0) | (df["timestamp_utc"].dt.second != 0)).any():
        raise SchemaError("non-hourly timestamps")
    flags: list[Flag] = []
    v = pd.to_numeric(df[value_col], errors="coerce")
    df = df.assign(**{value_col: v})
    if df[value_col].isna().any():
        flags.append(Flag("null_value", "warning", f"{int(df[value_col].isna().sum())} null/non-numeric rows dropped"))
        df = df[df[value_col].notna()]
    bad = (df[value_col] < lo) | (df[value_col] > hi)
    if bad.any():
        flags.append(Flag("out_of_range", "error", f"{int(bad.sum())} rows outside [{lo}, {hi}] dropped"))
        df = df[~bad]
    dup = df.duplicated("timestamp_utc", keep=False)
    if dup.any():
        if (df[dup].groupby("timestamp_utc")[value_col].nunique() > 1).any():
            raise ConflictError("duplicate timestamps with conflicting values")
        flags.append(Flag("duplicate_rows", "info", f"{int(df.duplicated('timestamp_utc').sum())} identical duplicates dropped"))
        df = df.drop_duplicates("timestamp_utc")
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    local_date = df["timestamp_utc"].dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    for d, n in local_date.groupby(local_date).size().items():
        exp = expected_hours(d)
        if n < exp:
            flags.append(Flag("incomplete_day", "warning", f"{n} of {exp} hours", str(d.date())))
        elif n > exp:
            flags.append(Flag("overfull_day", "warning", f"{n} hours, expected {exp}", str(d.date())))
    return Validated(df, flags)
