"""Local-date x hour panels (24 columns per day) from the UTC series.

Working in local day-hour space keeps "same hour yesterday / last week" aligned
across daylight-saving changes (a UTC lag of 24 h is 23 or 25 local hours on a
transition week). Approximation on the two DST days per year: the repeated
fall-back hour is averaged and the missing spring-forward hour is interpolated
from its neighbours.
"""

import pandas as pd

TZ = "America/New_York"
HOURS = list(range(24))


def to_panel(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """df: columns timestamp_utc, value_col (one zone). -> index local date, columns 0..23."""
    local = df["timestamp_utc"].dt.tz_convert(TZ)
    t = pd.DataFrame({"date": local.dt.tz_localize(None).dt.normalize(), "hour": local.dt.hour,
                      "v": df[value_col].to_numpy()})
    wide = t.groupby(["date", "hour"])["v"].mean().unstack("hour").reindex(columns=HOURS)
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq="D"))
    wide = wide.interpolate(axis=1, limit=1, limit_direction="both")
    # trim incomplete days at the two ends (e.g. a UTC-aligned series starts mid local day);
    # gaps INSIDE the range are real data problems and still raise below.
    bad = wide.isna().any(axis=1).to_numpy()
    first, last = 0, len(wide)
    while first < last and bad[first]:
        first += 1
    while last > first and bad[last - 1]:
        last -= 1
    wide = wide.iloc[first:last]
    if wide.isna().any().any():
        raise ValueError(f"{value_col}: gaps remain after DST interpolation "
                         f"({int(wide.isna().sum().sum())} cells)")
    wide.index.name = "date"
    return wide


def build_panels(prices: pd.DataFrame, load: pd.DataFrame, temp: pd.DataFrame,
                 zone: str = "N.Y.C.") -> dict[str, pd.DataFrame]:
    p = to_panel(prices[prices["zone"] == zone], "lbmp")
    ld = to_panel(load[load["zone"] == zone], "load_mw")
    tp = to_panel(temp, "temp_c")
    idx = p.index.intersection(ld.index).intersection(tp.index)
    return {"price": p.loc[idx], "load": ld.loc[idx], "temp": tp.loc[idx]}
