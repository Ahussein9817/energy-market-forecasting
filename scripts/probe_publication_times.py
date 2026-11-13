"""Record when NYISO actually publishes the files this project depends on.

    python scripts/probe_publication_times.py [--days 14] [--out results/publication_times.json]

HEAD requests only (no file bodies). For each recent delivery day it records HTTP status and the
server `Last-Modified` time for the day-ahead price file, the day-ahead load forecast (isolf) and
the actual integrated load file, plus Last-Modified of the current and previous monthly zips.
`Last-Modified` is the file's last modification, i.e. the latest revision, not necessarily the first
posting; treat the earliest plausible posting time as <= the value shown. NYISO keeps daily files only
for the current month, so this can only sample the last few weeks.
Network needed; not run in CI.
"""
import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

BASE = "https://mis.nyiso.com/public/csv"
ET = ZoneInfo("America/New_York")
DAILY = {"dam_price": "damlbmp/{d}damlbmp_zone.csv", "load_forecast": "isolf/{d}isolf.csv",
         "actual_load": "palIntegrated/{d}palIntegrated.csv"}


def head(url: str, ua: str):
    r = requests.head(url, headers={"User-Agent": ua}, timeout=30, allow_redirects=True)
    lm = r.headers.get("Last-Modified")
    return r.status_code, (parsedate_to_datetime(lm).astimezone(UTC) if lm else None)


def entry(status, lm):
    return {"status": status, "last_modified_utc": lm.isoformat() if lm else None,
            "last_modified_et": lm.astimezone(ET).strftime("%Y-%m-%d %H:%M %Z") if lm else None,
            "et_clock": lm.astimezone(ET).strftime("%H:%M") if lm else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--out", default="results/publication_times.json")
    ap.add_argument("--user-agent", default="energy-market-forecasting timing probe (research; contact via GitHub Ahussein9817)")
    a = ap.parse_args()
    today = datetime.now(UTC).astimezone(ET).date()
    rows = []
    for k in range(-1, a.days):
        delivery = today + timedelta(days=1 - k - 1)  # from tomorrow back `days` days
        d = delivery.strftime("%Y%m%d")
        row = {"delivery_date": str(delivery)}
        for name, pat in DAILY.items():
            row[name] = entry(*head(f"{BASE}/{pat.format(d=d)}", a.user_agent))
        rows.append(row)
    zips = {}
    for m in (today.replace(day=1), (today.replace(day=1) - timedelta(days=1)).replace(day=1)):
        d = m.strftime("%Y%m01")
        zips[m.strftime("%Y-%m")] = {
            "dam_price": entry(*head(f"{BASE}/damlbmp/{d}damlbmp_zone_csv.zip", a.user_agent)),
            "actual_load": entry(*head(f"{BASE}/palIntegrated/{d}palIntegrated_csv.zip", a.user_agent))}
    out = {"probed_at_utc": datetime.now(UTC).isoformat(), "daily": rows, "monthly_zips": zips,
           "note": "Last-Modified is the latest revision time, an upper bound on first posting."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    ok = [r for r in rows if r["dam_price"]["status"] == 200]
    print(f"{len(ok)} delivery days with a price file; written {a.out}")


if __name__ == "__main__":
    sys.exit(main())
