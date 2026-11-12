"""Exploratory figures for the North Hub day-ahead price series.

Writes PNGs to reports/figures/ and prints the summary numbers they show, so the
numbers quoted in the report can be traced back to this script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from dataset import load_prices

FIG = Path(__file__).resolve().parents[1] / "reports" / "figures"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    p = load_prices().reset_index()
    p = p[p["date"].dt.year.between(2019, 2025)]
    p["year"] = p["date"].dt.year
    threshold = p.loc[p["year"] <= 2023, "price"].quantile(0.95)

    yearly = p.groupby("year")["price"].agg(
        median="median",
        p95=lambda s: s.quantile(0.95),
        max="max",
        spike_hours=lambda s: (s > threshold).sum(),
    )
    print(f"spike threshold: ${threshold:.2f}/MWh")
    print(yearly.round(1))

    fig, ax = plt.subplots(figsize=(7, 3))
    daily = p.groupby("date")["price"].mean()
    ax.plot(daily.index, daily.values, lw=0.6, color="#1f4e79")
    ax.set_yscale("log")
    ax.set_ylabel("Daily mean price ($/MWh)")
    ax.set_title("HB_NORTH day-ahead price, 2019-2025")
    fig.tight_layout()
    fig.savefig(FIG / "daily_price.png", dpi=200)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(yearly.index, yearly["spike_hours"], color="#c55a11")
    ax.set_ylabel(f"Hours above ${threshold:.0f}/MWh")
    ax.set_title("Spike hours per year (train 95th percentile)")
    fig.tight_layout()
    fig.savefig(FIG / "spikes_per_year.png", dpi=200)

    summer = p[p["date"].dt.month.isin([6, 7, 8]) & p["year"].isin([2019, 2022, 2025])]
    shape = summer.groupby(["year", "hour"])["price"].median().unstack(0)
    fig, ax = plt.subplots(figsize=(5, 3))
    shape.plot(ax=ax, lw=1.5)
    ax.set_xlabel("Hour ending")
    ax.set_ylabel("Median price ($/MWh)")
    ax.set_title("Summer (Jun-Aug) hourly price shape")
    fig.tight_layout()
    fig.savefig(FIG / "summer_hourly_shape.png", dpi=200)
    print(shape.round(1))


if __name__ == "__main__":
    main()
