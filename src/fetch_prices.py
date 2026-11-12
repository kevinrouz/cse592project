"""Download ERCOT day-ahead hub prices and Henry Hub natural gas prices.

Prices come from ERCOT report NP4-180-ER (Historical DAM Load Zone and Hub Prices),
published as one zip per year containing an Excel workbook with a sheet per month.
That report needs no API key. Gas prices are EIA's daily Henry Hub spot series
(RNGWHHD), also keyless.

Writes data/raw/dam_prices.parquet (all hubs and load zones, long format) and
data/raw/henry_hub.parquet.
"""

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
YEARS = range(2018, 2026)
DOC_LIST = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS?reportTypeId=13060"
DOC_URL = "https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId={}"
HENRY_HUB = "https://www.eia.gov/dnav/ng/hist_xls/RNGWHHDd.xls"


def fetch_dam():
    docs = requests.get(DOC_LIST, timeout=60).json()["ListDocsByRptTypeRes"]["DocumentList"]
    by_year = {d["Document"]["FriendlyName"][-4:]: d["Document"]["DocID"] for d in docs}
    frames = []
    for year in YEARS:
        blob = requests.get(DOC_URL.format(by_year[str(year)]), timeout=300).content
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            sheets = pd.read_excel(z.open(z.namelist()[0]), sheet_name=None)
        frames += sheets.values()
        print(f"DAM {year}: {sum(len(s) for s in sheets.values())} rows", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["Repeated Hour Flag"] != "Y"]
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["Delivery Date"], format="%m/%d/%Y"),
            "hour": df["Hour Ending"].astype(str).str.split(":").str[0].astype(int),
            "point": df["Settlement Point"],
            "price": df["Settlement Point Price"].astype(float),
        }
    )
    out.to_parquet(RAW / "dam_prices.parquet")


def fetch_gas():
    blob = requests.get(HENRY_HUB, timeout=60).content
    df = pd.read_excel(io.BytesIO(blob), sheet_name="Data 1", skiprows=2)
    df.columns = ["date", "gas_price"]
    df = df.dropna()
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(RAW / "henry_hub.parquet")
    print(f"Henry Hub: {len(df)} days through {df['date'].max().date()}")


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    fetch_gas()
    fetch_dam()
