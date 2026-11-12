"""Plot one well-forecast and one badly forecast 48-hour window from validation.

Picks, among 2024 origins, the one where LightGBM's 48-hour MAE is lowest relative
to the naive same-hour-yesterday forecast (a success case) and the one with the
largest LightGBM MAE overall (a failure case). Ranking by improvement over naive
avoids picking a flat, easy day as the success example. The plotted model is the
conformally calibrated one, whose median is the same as the uncalibrated model's.

Usage: python src/plots.py [config]   (reads val_predictions_<config>.parquet; default "final")
Writes reports/figures/examples.png, with the success case on the left.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
MODEL = "lightgbm_cqr"


def draw_origin(ax, preds, origin, title, legend):
    lgbm = preds[(preds["model"] == MODEL) & (preds["origin"] == origin)]
    naive = preds[(preds["model"] == "naive_day") & (preds["origin"] == origin)]
    lgbm = lgbm.sort_values(["horizon", "hour"])
    naive = naive.sort_values(["horizon", "hour"])
    x = range(1, 49)

    ax.fill_between(x, lgbm["q10"], lgbm["q90"], color="#9dc3e6", alpha=0.6, label="10-90% interval")
    ax.plot(x, lgbm["q50"], color="#1f4e79", lw=1.5, label="Median")
    ax.plot(x, naive["q50"], color="#7f7f7f", lw=1, ls="--", label="Naive (same hour, day D)")
    ax.plot(x, lgbm["price"], color="#c00000", lw=1.5, label="Actual")
    ax.axvline(24.5, color="black", lw=0.5)
    ax.set_xticks([1, 7, 13, 19, 25, 31, 37, 43, 48])
    ax.set_xticklabels(["D+1 1", "7", "13", "19", "D+2 1", "7", "13", "19", "24"], fontsize=8)
    ax.set_xlabel("Hour ending", fontsize=9)
    ax.set_title(title, fontsize=10)
    if legend:
        ax.legend(fontsize=7.5, loc="upper left", frameon=False)


def main(config="final"):
    preds = pd.read_parquet(ROOT / "data" / "processed" / f"val_predictions_{config}.parquet")
    preds["err"] = (preds["price"] - preds["q50"]).abs()
    mae = preds.groupby(["origin", "model"])["err"].mean().unstack()
    # Require 48 complete hours so the plot is never missing a horizon.
    counts = preds[preds["model"] == MODEL].groupby("origin").size()
    mae = mae[counts.reindex(mae.index) == 48]

    success = (mae[MODEL] - mae["naive_day"]).idxmin()
    failure = mae[MODEL].idxmax()
    # One two-panel figure, since page space in the report is limited.
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.6), sharey=True)
    for ax, name, origin in [(axes[0], "Success", success), (axes[1], "Failure", failure)]:
        row = mae.loc[origin]
        print(
            f"{name}: origin {origin.date()}, LightGBM MAE {row[MODEL]:.1f}, "
            f"naive MAE {row['naive_day']:.1f}"
        )
        draw_origin(ax, preds, origin, f"{name}: issued {origin:%a %b %d, %Y}", legend=ax is axes[1])
    axes[0].set_ylabel("$/MWh", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "examples.png", dpi=250)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "final")
