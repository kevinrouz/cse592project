"""Build the modeling table: one row per (forecast origin day, target hour).

A forecast is issued at 9 AM Central on origin day D for the 48 hours of D+1 and
D+2. Every feature below is restricted to what a participant could see at that
moment:

- Day-ahead prices for day D itself are known, because the day-ahead market for D
  clears on D-1 around 1:30 PM. Prices for D+1 are not, since that auction closes
  at 10 AM on D. The latest price day available is therefore D, for both horizons.
- Load, wind and solar forecasts come from the last vintage posted at or before
  9 AM on D (see fetch_forecasts.py).
- Henry Hub gas uses the most recent spot price dated D-1 or earlier. EIA dates the
  series by trade date, and D's own trade has not settled by 9 AM.

Rows are assigned to train, validation or test by the year of the target date, so
no target hour in the validation or test years is ever used for training.

Writes data/processed/dataset.parquet.
"""

from pathlib import Path

import holidays
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "dataset.parquet"
HUB = "HB_NORTH"
SPLITS = {"train": range(2019, 2024), "val": [2024], "test": [2025]}


def hourly_grid(start, end):
    days = pd.date_range(start, end, freq="D")
    return pd.MultiIndex.from_product([days, range(1, 25)], names=["date", "hour"])


def load_prices():
    p = pd.read_parquet(RAW / "dam_prices.parquet")
    p = p[p["point"] == HUB].set_index(["date", "hour"])["price"]
    # The spring DST day has no hour ending 3; carry hour 2 forward so every day
    # has 24 rows and lagged lookups never hit a hole.
    return p.reindex(hourly_grid(p.index.levels[0].min(), p.index.levels[0].max())).ffill()


def load_forecasts():
    frames = []
    for name in ["load", "wind", "solar"]:
        files = sorted(RAW.glob(f"forecast_{name}_*.parquet"))
        f = pd.concat(pd.read_parquet(x) for x in files)
        # The 2022-07-01 solar posting, from the day ERCOT changed that report's
        # format, repeats each delivery hour several times.
        f = f.drop_duplicates(["origin", "date", "hour"], keep="last")
        frames.append(f.set_index(["origin", "date", "hour"])["value"].rename(name))
    f = pd.concat(frames, axis=1).groupby(level=[0, 1]).ffill()
    f["net_load"] = f["load"] - f["wind"] - f["solar"]
    return f


def build():
    prices = load_prices()
    forecasts = load_forecasts()
    gas = pd.read_parquet(RAW / "henry_hub.parquet").set_index("date")["gas_price"]

    origins = pd.date_range("2019-01-01", "2025-12-29", freq="D")
    rows = pd.MultiIndex.from_product(
        [origins, [1, 2], range(1, 25)], names=["origin", "horizon", "hour"]
    ).to_frame(index=False)
    rows["date"] = rows["origin"] + pd.to_timedelta(rows["horizon"], unit="D")

    def price_at(days, hours):
        return prices.reindex(pd.MultiIndex.from_arrays([days, hours])).to_numpy()

    rows["price"] = price_at(rows["date"], rows["hour"])

    # Same hour on the most recent known day (D), and one and six days before it.
    for lag in [0, 1, 6]:
        rows[f"price_d{lag}"] = price_at(rows["origin"] - pd.Timedelta(days=lag), rows["hour"])
    # Same hour one week before the target day; always on or before D.
    rows["price_week"] = price_at(rows["date"] - pd.Timedelta(days=7), rows["hour"])

    # Same-hour statistics over the week ending on D, which are steadier than any
    # single day's price.
    by_hour = prices.unstack("hour")
    for stat in ["mean", "max"]:
        rolled = by_hour.rolling(7).agg(stat).stack().rename(f"price_7d_{stat}")
        rows = rows.join(rolled, on=["origin", "hour"])

    daily = prices.groupby(level="date").agg(["mean", "max", "min"])
    daily.columns = [f"price_d0_{c}" for c in daily.columns]
    rows = rows.join(daily, on="origin")

    fc = forecasts.reindex(pd.MultiIndex.from_frame(rows[["origin", "date", "hour"]]))
    for col in fc.columns:
        rows[col] = fc[col].to_numpy()
    day_stats = forecasts.groupby(level=["origin", "date"])["net_load"].agg(["mean", "max", "min"])
    day_stats.columns = [f"net_load_day_{c}" for c in day_stats.columns]
    rows = rows.join(day_stats, on=["origin", "date"])
    # Day D's prices are the model's strongest input, so it helps to say how the
    # target hour's conditions differ from D's. D's net load forecast comes from the
    # D-1 vintage, posted before D's prices were known.
    today = forecasts["net_load"].reindex(
        pd.MultiIndex.from_arrays(
            [rows["origin"] - pd.Timedelta(days=1), rows["origin"], rows["hour"]]
        )
    )
    rows["net_load_d0"] = today.to_numpy()
    rows["net_load_change"] = rows["net_load"] - rows["net_load_d0"]
    rows["solar_share"] = rows["solar"] / rows["load"]
    rows["wind_share"] = rows["wind"] / rows["load"]

    gas_daily = gas.reindex(pd.date_range(gas.index.min(), "2026-01-01")).ffill()
    rows["gas_price"] = gas_daily.reindex(rows["origin"] - pd.Timedelta(days=1)).to_numpy()

    us_holidays = holidays.US(years=range(2018, 2027))
    rows["dow"] = rows["date"].dt.dayofweek
    rows["month"] = rows["date"].dt.month
    rows["holiday"] = rows["date"].dt.date.map(lambda d: d in us_holidays).astype(int)
    doy = rows["date"].dt.dayofyear
    rows["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    rows["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    rows["split"] = None
    for split, years in SPLITS.items():
        rows.loc[rows["date"].dt.year.isin(years), "split"] = split
    rows = rows.dropna(subset=["split", "price"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(OUT)
    return rows


if __name__ == "__main__":
    df = build()
    print(df.groupby(["split", "horizon"]).size())
    print(df.isna().mean().loc[lambda s: s > 0].round(4))
