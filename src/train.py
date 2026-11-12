"""Fit the naive baselines and LightGBM quantile models, then score them on validation.

LightGBM is trained in a variance-stabilized space z = asinh((price - c) / s), with
c and s the median and median absolute deviation of training prices. Because
quantiles commute with monotone transforms, the q10/q50/q90 predicted for z map
back through the inverse transform to q10/q50/q90 of price. The transform keeps a
handful of $5,000-$9,000 hours (most of them from Winter Storm Uri in February
2021) from dominating every split the trees make.

The number of boosting rounds is chosen by early stopping with 2023 held out, and
the model is then refit on all of 2019-2023 with that many rounds. The validation
year is never used for fitting or stopping, so validation scores are not tuned to it.

Usage: python src/train.py [--raw]   (--raw trains on untransformed prices)
Writes reports/val_metrics.csv and data/processed/val_predictions.parquet.
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from metrics import QUANTILES, report

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

FEATURES = [
    "horizon", "hour", "dow", "month", "holiday", "doy_sin", "doy_cos",
    "price_d0", "price_d1", "price_d6", "price_week",
    "price_d0_mean", "price_d0_max", "price_d0_min",
    "load", "wind", "solar", "net_load", "solar_share", "wind_share",
    "net_load_day_mean", "net_load_day_max", "net_load_day_min",
    "gas_price",
]
PARAMS = {
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}


class Asinh:
    def __init__(self, y):
        self.c = np.median(y)
        self.s = np.median(np.abs(y - self.c))

    def forward(self, y):
        return np.arcsinh((y - self.c) / self.s)

    def inverse(self, z):
        return np.sinh(z) * self.s + self.c


class Identity:
    def forward(self, y):
        return np.asarray(y)

    def inverse(self, z):
        return z


def naive(df, column):
    return df[["origin", "date", "hour", "horizon", "price"]].assign(
        q10=np.nan, q50=df[column], q90=np.nan
    )


def fit_quantile(train, q, transform):
    inner, stop = train[train["date"].dt.year < 2023], train[train["date"].dt.year == 2023]
    params = {**PARAMS, "objective": "quantile", "alpha": q}
    probe = lgb.train(
        params,
        lgb.Dataset(inner[FEATURES], transform.forward(inner["price"])),
        num_boost_round=3000,
        valid_sets=[lgb.Dataset(stop[FEATURES], transform.forward(stop["price"]))],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    rounds = probe.best_iteration
    model = lgb.train(
        params, lgb.Dataset(train[FEATURES], transform.forward(train["price"])), rounds
    )
    print(f"  q={q}: {rounds} rounds", flush=True)
    return model


def lightgbm_forecast(train, target, transform):
    out = target[["origin", "date", "hour", "horizon", "price"]].copy()
    models = {}
    for col, q in QUANTILES.items():
        models[col] = fit_quantile(train, q, transform)
        out[col] = transform.inverse(models[col].predict(target[FEATURES]))
    # Independently fitted quantiles can cross; sorting restores q10 <= q50 <= q90.
    out[list(QUANTILES)] = np.sort(out[list(QUANTILES)].to_numpy(), axis=1)
    return out, models


def main(raw=False):
    df = pd.read_parquet(DATA / "dataset.parquet")
    train, val = df[df["split"] == "train"], df[df["split"] == "val"]
    spike = train["price"].quantile(0.95)
    print(f"spike threshold (train 95th pct): ${spike:.2f}/MWh")

    transform = Identity() if raw else Asinh(train["price"].to_numpy())
    lgbm, models = lightgbm_forecast(train, val, transform)
    forecasts = {
        "naive_day": naive(val, "price_d0"),
        "naive_week": naive(val, "price_week"),
        "lightgbm": lgbm,
    }
    metrics = report(forecasts, spike)

    REPORTS.mkdir(exist_ok=True)
    suffix = "_raw" if raw else ""
    metrics.to_csv(REPORTS / f"val_metrics{suffix}.csv", index=False)
    preds = pd.concat(f.assign(model=name) for name, f in forecasts.items())
    preds.to_parquet(DATA / f"val_predictions{suffix}.parquet")
    importance = pd.Series(models["q50"].feature_importance("gain"), FEATURES)
    importance.sort_values(ascending=False).to_csv(REPORTS / f"feature_importance{suffix}.csv")

    pd.set_option("display.width", 160)
    print(metrics.round(3).to_string(index=False))


if __name__ == "__main__":
    main(raw="--raw" in sys.argv)
