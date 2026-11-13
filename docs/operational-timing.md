# Operational timing: what is known when

Why this document exists: the original backtest forecasts the 24 prices of day D+1 "at the end of day D". NYISO publishes those prices much earlier (below), so that protocol is a valid test of how predictable the series is but cannot be run as a real forecast. The operational forecast defined here uses an earlier origin and only information available then. The end-of-day backtest results are kept, unchanged, as a labelled research experiment.

Evidence labels: **VERIFIED** (observed by probe, `results/publication_times.json`, 2026-11-13), **DOCUMENTED** (published by the source), **TO CONFIRM** (not yet verified from a current official source).

## Publication times (NYISO, America/New_York)

| Data | When it is published | Evidence |
|---|---|---|
| Day-ahead LBMP for delivery day D+1 | **09:32-09:33 ET on day D**, on all 12 sampled days (file `Last-Modified`, 2026-11-02..11-13 delivery days). Monthly-zip timestamps from other months (Sep, Oct, Jan) fall at the same ET clock time in both daylight and standard time. | VERIFIED. NYISO's 2013 market presentation lists a 5 am DAM close and 11 am latest posting with a ~10 am best-effort target ([NYISO document](https://www.nyiso.com/documents/20142/1406794/DAMPostingTime_Aug6thMIWG.pdf/ea980810-9280-7e2f-edef-ff1acda43964)); this is old, so the **5 am close is TO CONFIRM** against the current tariff or Market Participant User's Guide |
| Day-ahead load forecast (`isolf`) for D+1 | **07:15-09:00 ET on day D** (12 samples). One NYISO training deck says "by 8 a.m."; the sampled range is later on some days | VERIFIED range |
| Actual integrated load for day D | **~00:17-00:52 ET on D+1** (10 of 11 samples). **Can be revised later**: the file for 2026-11-11 was last modified at 12:25 ET on 11-12 | VERIFIED; so load values have vintages |
| Daily files | A **rolling window of about 12 days** (2026-11-01's daily file was already gone on 11-13). The current month's zip holds the whole month and is updated daily, at the same time as the daily file; completed months are monthly zips | VERIFIED (a first version of this table said "current month only"; the real-data run showed that was wrong) |
| Day-ahead price file for D+2 | Not present (404) | VERIFIED |

`Last-Modified` is the latest revision time, so first posting is at or before these times.

## Operational forecast definition

- **Zone and target:** NYISO N.Y.C. zone, 24 hourly day-ahead prices (local hours 0-23) of delivery day T = D+1.
- **Nominal origin:** 05:00 ET on day D (the assumed DAM close; **TO CONFIRM**). Forecasts generated after about 09:30 ET on D are flagged `late`, because the answer is then public; they are stored but excluded from operational performance metrics.
- **Baseline:** same hour of day D (price of D), available at the origin.

| Input | Legal at the origin? | Why |
|---|---|---|
| Day-ahead prices through D | Yes | published 09:32 ET on D-1 |
| Actual load through D-1 | Yes | posted ~00:20 ET on D (may be revised later) |
| Actual load of D, temperature observed on D | **No** | day D is not over |
| NYISO load forecast for D+1 | **No** at 05:00 | posted 07:15-09:00 |
| Weather forecast for D+1 | Yes, if issued before the origin | see below |
| Calendar | Yes | |

The end-of-day feature sets (B, C, D) use the load of day D and the observed temperature of day D, which are **not** legal at this origin. The operational sets `OB` (prices + calendar + load through D-1) and `OD` (`OB` + forecast temperature) drop them.

## Weather forecast availability

Open-Meteo Previous Runs API ([documentation](https://open-meteo.com/en/docs/previous-runs-api)): `temperature_2m_previous_dayN` is the value predicted N x 24 hours before the valid time; `previous_day0` is the current run (the live forecast API). The run cycle used for each offset is **not documented**.

- Valid hours of D+1 run from 05:00 UTC on D+1 to 04:00 UTC on D+2 (EST). A 48-hour lead therefore means runs initialised by about 04:00 UTC on D, comfortably before a 05:00 ET (10:00 UTC) origin even allowing hours of model publication delay.
- A 24-hour lead (`previous_day1`) means runs initialised up to about 04:00 UTC on D+1, **after** the origin for most target hours. It is not safe at the operational origin; the earlier end-of-day experiment used it, which is acceptable only at the end-of-day origin.
- **Backtest proxy:** `previous_day2`. **Live input:** the live forecast API snapshot taken at generation time, stored with its retrieval time. The two are related but not identical (the live run available at 05:00 ET is the 00 UTC cycle, about 29-52 hours before the target hours). Daily snapshots accumulate so the proxy can be checked against what was actually available once enough days exist (**TO CONFIRM**, needs weeks of scheduled data).
- GFS 2 m temperature is archived from 2021-03 in the Previous Runs API; the 2023-12-29..2024-01-20 hole documented in `docs/data-sources.md` remains.

## What is not claimed

No live monitoring or trading use. The operational protocol has been defined from public timestamps and documentation, not from NYISO's current official market rules, and a single week of scheduled data says nothing about performance.
