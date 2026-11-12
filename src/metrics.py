"""Forecast metrics, reported per horizon and per price regime.

A forecast table holds the columns `origin`, `price` (actual), `horizon` (1 or 2)
and `q10`, `q50`, `q90`. A point-only forecaster such as a naive baseline fills
just `q50`, and its interval metrics come out as NaN.

Spike hours are those whose actual price exceeds a fixed threshold, the 95th
percentile of training-set prices. The threshold is computed once from training
data and reused for every split so that the definition of a spike does not drift
with the data being scored.
"""

import numpy as np
import pandas as pd
from scipy import stats

QUANTILES = {"q10": 0.1, "q50": 0.5, "q90": 0.9}
ALPHA = 0.2  # the 10-90 interval has nominal miss rate 0.2


def pinball(y, pred, q):
    diff = y - pred
    return np.mean(np.maximum(q * diff, (q - 1) * diff))


def winkler(y, lo, hi, alpha=ALPHA):
    """Interval score: width plus 2/alpha times the distance by which y falls outside.

    Lower is better. Unlike coverage alone, it cannot be gamed by widening the
    interval, because width is paid for on every hour.
    """
    below = np.maximum(lo - y, 0)
    above = np.maximum(y - hi, 0)
    return np.mean((hi - lo) + (2 / alpha) * (below + above))


def detection(y, pred, threshold):
    """Precision and recall of `pred > threshold` as an alarm for `y > threshold`."""
    actual, flagged = y > threshold, pred > threshold
    hits = np.sum(actual & flagged)
    precision = hits / flagged.sum() if flagged.sum() else np.nan
    recall = hits / actual.sum() if actual.sum() else np.nan
    return precision, recall


def score(df, spike_threshold):
    y, med = df["price"], df["q50"]
    out = {
        "n": len(df),
        "MAE": np.mean(np.abs(y - med)),
        "RMSE": np.sqrt(np.mean((y - med) ** 2)),
    }
    out["spike_prec_q50"], out["spike_rec_q50"] = detection(y, med, spike_threshold)
    if df["q10"].notna().all():
        out["pinball"] = np.mean([pinball(y, df[c], q) for c, q in QUANTILES.items()])
        out["coverage"] = np.mean((y >= df["q10"]) & (y <= df["q90"]))
        out["width"] = np.mean(df["q90"] - df["q10"])
        out["winkler"] = winkler(y, df["q10"], df["q90"])
        out["spike_prec_q90"], out["spike_rec_q90"] = detection(y, df["q90"], spike_threshold)
    return out


def report(forecasts, spike_threshold, reference="naive_day"):
    """Score every model in a {name: forecast table} dict, split by horizon and regime.

    rMAE is each row's MAE divided by the reference model's MAE on the same rows,
    the scale-free comparison recommended by Lago et al. (2021).
    """
    rows = []
    for name, df in forecasts.items():
        df = df.assign(regime=np.where(df["price"] > spike_threshold, "spike", "normal"))
        groups = [("all", "all", df)]
        groups += [(f"D+{h}", "all", g) for h, g in df.groupby("horizon")]
        groups += [(f"D+{h}", r, g) for (h, r), g in df.groupby(["horizon", "regime"])]
        for horizon, regime, g in groups:
            rows.append(
                {"model": name, "horizon": horizon, "regime": regime, **score(g, spike_threshold)}
            )
    out = pd.DataFrame(rows)
    ref = out[out["model"] == reference].set_index(["horizon", "regime"])["MAE"]
    out.insert(5, "rMAE", out["MAE"] / ref.reindex(pd.MultiIndex.from_frame(out[["horizon", "regime"]])).to_numpy())
    return out


def diebold_mariano(a, b, horizon):
    """Test whether forecast `a` has lower absolute error than `b` at one horizon.

    Errors are averaged over the 24 hours of each target day, giving one loss
    differential per forecast origin, and the test uses a Newey-West variance with
    a 7-day lag because errors on nearby days are correlated. Returns the DM
    statistic and the one-sided p-value for "a is more accurate than b"; a negative
    statistic favors `a`. Raises KeyError if the two tables share no origins.
    """
    def daily(df):
        df = df[df["horizon"] == horizon]
        return (df["price"] - df["q50"]).abs().groupby(df["origin"]).mean()

    d = (daily(a) - daily(b)).dropna()
    if d.empty:
        raise KeyError("no common origins")
    n, lag = len(d), 7
    centered = d - d.mean()
    var = np.sum(centered**2) / n
    for k in range(1, lag + 1):
        cov = np.sum(centered[k:].to_numpy() * centered[:-k].to_numpy()) / n
        var += 2 * (1 - k / (lag + 1)) * cov
    stat = d.mean() / np.sqrt(var / n)
    return stat, stats.norm.cdf(stat)
