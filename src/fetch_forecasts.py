"""Download ERCOT's archived load, wind and solar forecasts as they stood before 9 AM.

For every forecast origin day D, this keeps the last vintage of each report posted
at or before 09:00 Central on D, and stores only the rows for delivery days D+1 and
D+2. That is the information a forecaster would have had at the proposal's 9 AM
cutoff, so later features can be joined without leaking the future.

Requires ERCOT_API_USERNAME, ERCOT_API_PASSWORD and ERCOT_PUBLIC_API_SUBSCRIPTION_KEY
in the environment or in .env. Output goes to data/raw/forecast_<name>_<year>.parquet,
one file per year, and years whose file already exists are skipped so an interrupted
run can resume.
"""

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from gridstatus.ercot_api.ercot_api import PUBLIC_BASE_URL, ErcotAPI

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
CUTOFF_HOUR = 9

# report id -> (short name, column holding the system-wide forecast)
REPORTS = {
    "np3-565-cd": ("load", "SystemTotal"),
    "np4-732-cd": ("wind", "STWPF_SYSTEM_WIDE"),
    "np4-737-cd": ("solar", "STPPF_SYSTEM_WIDE"),
}


def list_archives(api, emil, year):
    rows = []
    for month in range(1, 13):
        start = pd.Timestamp(year, month, 1)
        end = start + pd.offsets.MonthBegin(1)
        page, pages = 1, 1
        while page <= pages:
            resp = api.make_api_call(
                f"{PUBLIC_BASE_URL}/archive/{emil}",
                api_params={
                    "postDatetimeFrom": start.strftime("%Y-%m-%dT%H:%M:%S"),
                    "postDatetimeTo": end.strftime("%Y-%m-%dT%H:%M:%S"),
                    "size": 1000,
                    "page": page,
                },
            )
            pages = resp["_meta"]["totalPages"]
            rows += [(a["docId"], a["postDatetime"]) for a in resp["archives"]]
            page += 1
    df = pd.DataFrame(rows, columns=["doc_id", "posted"])
    df["posted"] = pd.to_datetime(df["posted"], format="ISO8601")
    return df


def pick_vintages(archives):
    """Keep, for each origin day, the last posting at or before the 9 AM cutoff."""
    a = archives.copy()
    a["origin"] = a["posted"].dt.normalize()
    a = a[a["posted"] <= a["origin"] + pd.Timedelta(hours=CUTOFF_HOUR)]
    return a.sort_values("posted").groupby("origin").tail(1).set_index("doc_id")


def download(api, emil, doc_ids):
    """Bulk-download documents in batches of 1000; returns {doc_id: csv bytes}."""
    out = {}
    for i in range(0, len(doc_ids), 1000):
        batch = [str(d) for d in doc_ids[i : i + 1000]]
        blob = api.make_api_call(
            f"{PUBLIC_BASE_URL}/archive/{emil}/download",
            api_params={"docIds": batch},
            parse_json=False,
            method="POST",
        )
        with zipfile.ZipFile(io.BytesIO(blob)) as outer:
            for name in outer.namelist():
                with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
                    out[int(name.split(".")[0])] = inner.read(inner.namelist()[0])
    return out


def parse(raw, value_col, origin):
    df = pd.read_csv(io.BytesIO(raw))
    if "InUseFlag" in df.columns:
        df = df[df["InUseFlag"] == "Y"]
    date_col = "DeliveryDate" if "DeliveryDate" in df.columns else "DELIVERY_DATE"
    hour_col = "HourEnding" if "HourEnding" in df.columns else "HOUR_ENDING"
    # The repeated 1-2 AM hour on the November DST change is flagged "Y"; drop it
    # so every day indexes hours 1-24 the same way the price data does.
    df = df[df["DSTFlag"] != "Y"]
    out = pd.DataFrame(
        {
            "origin": origin,
            "date": pd.to_datetime(df[date_col], format="%m/%d/%Y"),
            "hour": df[hour_col].astype(str).str.split(":").str[0].astype(int),
            "value": df[value_col].astype(float),
        }
    )
    horizon = (out["date"] - origin).dt.days
    return out[horizon.isin([1, 2])]


def fetch_year(api, emil, year):
    name, value_col = REPORTS[emil]
    path = RAW / f"forecast_{name}_{year}.parquet"
    if path.exists():
        return
    vintages = pick_vintages(list_archives(api, emil, year))
    docs = download(api, emil, vintages.index.tolist())
    missing = set(vintages.index) - set(docs)
    if missing:
        print(f"{name} {year}: {len(missing)} documents missing from download")
    frames = [parse(raw, value_col, vintages.loc[d, "origin"]) for d, raw in docs.items()]
    pd.concat(frames).sort_values(["origin", "date", "hour"]).to_parquet(path)
    print(f"{name} {year}: {len(frames)} origin days", flush=True)


def main(years):
    load_dotenv()
    RAW.mkdir(parents=True, exist_ok=True)
    api = ErcotAPI(sleep_seconds=1.0)
    for emil in REPORTS:
        for year in years:
            fetch_year(api, emil, year)


if __name__ == "__main__":
    years = [int(y) for y in sys.argv[1:]] or list(range(2019, 2026))
    main(years)
