"""Leakage-safe day-ahead feature construction.

Forecast origin: end of local day D. Target: the 24 hourly prices of day T = D + 1.
Every price/load feature for target day T is built from days <= D (shift >= 1 day on the
day-indexed panel). Calendar facts of T (hour, weekday, month, holiday) are known in
advance. `temp_t` is the temperature OF THE TARGET DAY taken from historical actuals,
i.e. a perfect-foresight weather forecast: an optimistic assumption, so it lives in its
own feature set ('C') and results with it must be labelled as such.

Feature sets: A = prices + calendar; B = A + load lags; C = B + temperature (oracle).
"""

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

FEATURE_SETS = ("A", "B", "C")


def _long(frame: pd.DataFrame, name: str) -> pd.Series:
    s = frame.stack()
    s.index.names = ["date", "hour"]
    return s.rename(name)


def make_features(panels: dict[str, pd.DataFrame], feature_set: str):
    """Returns (X, y). Index: (target_date, hour). Rows whose lags are unavailable are dropped."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
    price, load, temp = panels["price"], panels["load"], panels["temp"]
    if not (price.index.is_monotonic_increasing and (price.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()):
        raise ValueError("panel must have contiguous, sorted daily index")

    cols = {}
    for k, nm in ((1, "p_d1"), (2, "p_d2"), (7, "p_d7"), (14, "p_d14")):
        cols[nm] = _long(price.shift(k), nm)
    cols["p_wk_mean"] = _long(sum(price.shift(k) for k in range(1, 8)) / 7, "p_wk_mean")
    prev = price.shift(1)  # day D
    for nm, ser in (("pD_mean", prev.mean(axis=1)), ("pD_max", prev.max(axis=1)), ("pD_min", prev.min(axis=1))):
        cols[nm] = _long(pd.DataFrame({h: ser for h in price.columns}), nm)
    if feature_set in ("B", "C"):
        cols["load_d1"] = _long(load.shift(1), "load_d1")
        cols["load_d7"] = _long(load.shift(7), "load_d7")
    if feature_set == "C":
        cols["temp_t"] = _long(temp, "temp_t")            # target day: oracle weather
        cols["temp_d1"] = _long(temp.shift(1), "temp_d1")
    X = pd.concat(cols.values(), axis=1)
    dates = X.index.get_level_values("date")
    hol = USFederalHolidayCalendar().holidays(start=dates.min(), end=dates.max())
    X["hour"] = X.index.get_level_values("hour")
    X["dow"] = dates.day_of_week
    X["month"] = dates.month
    X["is_holiday"] = dates.isin(hol).astype(int)
    y = _long(price, "y")
    X = X.dropna()
    return X, y.loc[X.index]
