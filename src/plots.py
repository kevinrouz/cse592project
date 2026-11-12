"""Plot one well-forecast and one badly forecast 48-hour window from validation.

Picks, among 2024 origins, the one where LightGBM's 48-hour MAE is lowest relative
to the naive same-hour-yesterday forecast (a success case) and the one with the
largest LightGBM MAE overall (a failure case). Ranking by improvement over naive
avoids picking a flat, easy day as the success example. The plotted model is the
conformally calibrated one, whose median is the same as the uncalibrated model's.

Usage: python src/plots.py [config]   (reads val_predictions_<config>.parquet; default "final")
Writes reports/figures/example_success.png and example_failure.png.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
MODEL = "lightgbm_cqr"


def plot_origin(preds, origin, title, path):
    lgbm = preds[(preds["model"] == MODEL) & (preds["origin"] == origin)]
    naive = preds[(preds["model"] == "naive_day") & (preds["origin"] == origin)]
    lgbm = lgbm.sort_values(["horizon", "hour"])
    naive = naive.sort_values(["horizon", "hour"])
    x = range(1, 49)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.fill_between(x, lgbm["q10"], lgbm["q90"], color="#9dc3e6", alpha=0.6, label="LightGBM 10-90%")
    ax.plot(x, lgbm["q50"], color="#1f4e79", lw=1.5, label="LightGBM median")
    ax.plot(x, naive["q50"], color="#7f7f7f", lw=1, ls="--", label="Naive (same hour, day D)")
    ax.plot(x, lgbm["price"], color="#c00000", lw=1.5, label="Actual")
    ax.axvline(24.5, color="black", lw=0.5)
    ax.set_xticks([1, 7, 13, 19, 25, 31, 37, 43, 48])
    ax.set_xticklabels(["D+1 1", "7", "13", "19", "D+2 1", "7", "13", "19", "24"])
    ax.set_xlabel("Hour ending")
    ax.set_ylabel("$/MWh")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main(config="final"):
    preds = pd.read_parquet(ROOT / "data" / "processed" / f"val_predictions_{config}.parquet")
    preds["err"] = (preds["price"] - preds["q50"]).abs()
    mae = preds.groupby(["origin", "model"])["err"].mean().unstack()
    # Require 48 complete hours so the plot is never missing a horizon.
    counts = preds[preds["model"] == MODEL].groupby("origin").size()
    mae = mae[counts.reindex(mae.index) == 48]

    success = (mae[MODEL] - mae["naive_day"]).idxmin()
    failure = mae[MODEL].idxmax()
    for name, origin in [("success", success), ("failure", failure)]:
        row = mae.loc[origin]
        print(
            f"{name}: origin {origin.date()}, LightGBM MAE {row[MODEL]:.1f}, "
            f"naive MAE {row['naive_day']:.1f}"
        )
        title = f"Forecast issued {origin:%a %b %d, %Y} 9 AM (LightGBM MAE ${row[MODEL]:.0f})"
        plot_origin(preds, origin, title, FIG / f"example_{name}.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "final")
