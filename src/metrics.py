"""Forecast metrics, reported per horizon and per price regime.

A forecast table holds the columns `price` (actual), `horizon` (1 or 2) and
`q10`, `q50`, `q90`. A point-only forecaster such as a naive baseline fills just
`q50`, and its interval metrics come out as NaN.

Spike hours are those whose actual price exceeds a fixed threshold, the 95th
percentile of training-set prices. The threshold is computed once from training
data and reused for every split so that the definition of a spike does not drift
with the data being scored.
"""

import numpy as np
import pandas as pd

QUANTILES = {"q10": 0.1, "q50": 0.5, "q90": 0.9}


def pinball(y, pred, q):
    diff = y - pred
    return np.mean(np.maximum(q * diff, (q - 1) * diff))


def score(df):
    y, med = df["price"], df["q50"]
    out = {
        "n": len(df),
        "MAE": np.mean(np.abs(y - med)),
        "RMSE": np.sqrt(np.mean((y - med) ** 2)),
    }
    if df["q10"].notna().all():
        out["pinball"] = np.mean([pinball(y, df[c], q) for c, q in QUANTILES.items()])
        out["coverage"] = np.mean((y >= df["q10"]) & (y <= df["q90"]))
        out["width"] = np.mean(df["q90"] - df["q10"])
    else:
        out.update(pinball=np.nan, coverage=np.nan, width=np.nan)
    return out


def report(forecasts, spike_threshold):
    """Score every model in a {name: forecast table} dict, split by horizon and regime."""
    rows = []
    for name, df in forecasts.items():
        df = df.assign(regime=np.where(df["price"] > spike_threshold, "spike", "normal"))
        groups = [("all", "all", df)]
        groups += [(f"D+{h}", "all", g) for h, g in df.groupby("horizon")]
        groups += [(f"D+{h}", r, g) for (h, r), g in df.groupby(["horizon", "regime"])]
        for horizon, regime, g in groups:
            rows.append({"model": name, "horizon": horizon, "regime": regime, **score(g)})
    return pd.DataFrame(rows)
