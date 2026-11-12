import io
import zipfile
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests

from src import data_loader as dl


def _zip(files: dict[str, str]) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for n, c in files.items():
            zf.writestr(n, c)
    buf.seek(0)
    return buf


LB_HEAD = "Time Stamp,Name,PTID,LBMP ($/MWHr),Marginal Cost Losses ($/MWHr),Marginal Cost Congestion ($/MWHr)\n"


def test_nyiso_url_shapes():
    assert dl.nyiso_url("damlbmp_zone", 2024, 3) == \
        "https://mis.nyiso.com/public/csv/damlbmp/20240301damlbmp_zone_csv.zip"
    assert dl.nyiso_url("palIntegrated", 2019, 12).endswith("/palIntegrated/20191201palIntegrated_csv.zip")


def test_parse_lbmp_regular_day_converts_to_utc():
    rows = "01/15/2024 00:00,N.Y.C.,1,30.0,1.0,0.5\n01/15/2024 01:00,N.Y.C.,1,31.0,1.0,0.5\n"
    df = dl.parse_dam_lbmp(_zip({"20240115damlbmp_zone.csv": LB_HEAD + rows}))
    assert df["timestamp_utc"].iloc[0] == pd.Timestamp("2024-01-15 05:00", tz="UTC")  # EST = UTC-5
    assert df["lbmp"].tolist() == [30.0, 31.0] and (df["zone"] == "N.Y.C.").all()


def test_parse_lbmp_fall_back_day_has_25_unique_utc_hours():
    hours = ["00:00", "01:00", "01:00"] + [f"{h:02d}:00" for h in range(2, 24)]
    rows = "".join(f"11/03/2024 {h},N.Y.C.,1,{i}.0,0,0\n" for i, h in enumerate(hours))
    df = dl.parse_dam_lbmp(_zip({"d.csv": LB_HEAD + rows}))
    assert len(df) == 25 and df["timestamp_utc"].is_unique and df["timestamp_utc"].is_monotonic_increasing
    assert (df["timestamp_utc"].diff().dropna() == pd.Timedelta(hours=1)).all()


def test_parse_lbmp_spring_forward_day_has_23_hours_contiguous_utc():
    hours = [f"{h:02d}:00" for h in range(24) if h != 2]
    rows = "".join(f"03/10/2024 {h},N.Y.C.,1,1.0,0,0\n" for h in hours)
    df = dl.parse_dam_lbmp(_zip({"d.csv": LB_HEAD + rows}))
    assert len(df) == 23
    assert (df["timestamp_utc"].diff().dropna() == pd.Timedelta(hours=1)).all()


def test_parse_lbmp_multiple_zones_handled_independently():
    rows = ("11/03/2024 01:00,A,1,1,0,0\n11/03/2024 01:00,A,1,2,0,0\n"
            "11/03/2024 01:00,B,2,3,0,0\n11/03/2024 01:00,B,2,4,0,0\n")
    df = dl.parse_dam_lbmp(_zip({"d.csv": LB_HEAD + rows}))
    a = df[df.zone == "A"]["timestamp_utc"]
    assert a.iloc[1] - a.iloc[0] == pd.Timedelta(hours=1)


def test_parse_lbmp_bad_columns_raise():
    with pytest.raises(dl.DataError, match="unexpected LBMP"):
        dl.parse_dam_lbmp(_zip({"d.csv": "a,b\n1,2\n"}))
    with pytest.raises(dl.DataError, match="no CSV"):
        dl.parse_dam_lbmp(_zip({"readme.txt": "x"}))


def test_parse_load_uses_file_timezone_label_exactly():
    csv = ('"Time Stamp","Time Zone","Name","PTID","Integrated Load"\n'
           '"11/03/2024 01:00:00","EDT","N.Y.C.",1,5000.5\n'
           '"11/03/2024 01:00:00","EST","N.Y.C.",1,4900.5\n')
    df = dl.parse_integrated_load(_zip({"l.csv": csv}))
    assert df["timestamp_utc"].tolist() == [pd.Timestamp("2024-11-03 05:00", tz="UTC"),
                                            pd.Timestamp("2024-11-03 06:00", tz="UTC")]
    assert df["load_mw"].tolist() == [5000.5, 4900.5]


def test_parse_load_unknown_tz_label_raises():
    csv = '"Time Stamp","Time Zone","Name","PTID","Integrated Load"\n"11/03/2024 01:00:00","XYZ","N",1,1\n'
    with pytest.raises(dl.DataError, match="unknown time zone"):
        dl.parse_integrated_load(_zip({"l.csv": csv}))


def _resp(content=b"PK123", ok=True):
    r = MagicMock()
    r.content = content
    if not ok:
        r.raise_for_status.side_effect = requests.HTTPError("503")
    return r


def test_download_month_retries_then_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    s = MagicMock()
    s.get.side_effect = [_resp(ok=False), _resp()]
    p = dl.download_month("palIntegrated", 2024, 1, tmp_path, session=s)
    assert p.read_bytes() == b"PK123" and s.get.call_count == 2
    dl.download_month("palIntegrated", 2024, 1, tmp_path, session=s)  # cached: no new call
    assert s.get.call_count == 2


def test_download_month_rejects_non_zip_and_gives_up(tmp_path, monkeypatch):
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    s = MagicMock()
    s.get.return_value = _resp(content=b"<html>404</html>")
    with pytest.raises(dl.DataError, match="failed after 3"):
        dl.download_month("palIntegrated", 2024, 1, tmp_path, session=s)
    assert not list(tmp_path.glob("*.zip"))


def test_fetch_open_meteo_parses_and_validates():
    s = MagicMock()
    r = MagicMock()
    r.json.return_value = {"hourly": {"time": ["2024-01-01T00:00", "2024-01-01T01:00"],
                                      "temperature_2m": [1.5, None]}}
    s.get.return_value = r
    df = dl.fetch_open_meteo("2024-01-01", "2024-01-01", session=s)
    assert df["timestamp_utc"].iloc[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")
    assert df["temp_c"].isna().tolist() == [False, True]
    r.json.return_value = {"error": True}
    with pytest.raises(dl.DataError):
        dl.fetch_open_meteo("2024-01-01", "2024-01-01", session=s)


def test_months_inclusive():
    assert list(dl.months("2023-11-15", "2024-02-01")) == [(2023, 11), (2023, 12), (2024, 1), (2024, 2)]
