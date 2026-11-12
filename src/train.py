"""Fit the naive baselines and LightGBM quantile models, then score them on validation.

LightGBM is trained in a variance-stabilized space z = asinh((price - c) / s), with
c and s the median and median absolute deviation of training prices. Because
quantiles commute with monotone transforms, the q10/q50/q90 predicted for z map
back through the inverse transform to q10/q50/q90 of price. The transform keeps a
handful of $5,000-$9,000 hours (most of them from Winter Storm Uri in February
2021) from dominating every split the trees make.

Each quantile model is first fit on 2019-2022 with early stopping on 2023. That
"probe" fit sets the number of boosting rounds, and its 2023 predictions are the
held-out errors used to calibrate the intervals. The final model is then refit on
all of 2019-2023 with the same number of rounds. The validation year is never used
for fitting, stopping, model selection or the initial calibration.

Usage: python src/train.py [config]   (config is a key of CONFIGS; default "final")
Writes reports/val_metrics_<config>.csv and data/processed/val_predictions_<config>.parquet.
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from metrics import QUANTILES, diebold_mariano, report

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

BASE_FEATURES = [
    "horizon", "hour", "dow", "month", "holiday", "doy_sin", "doy_cos",
    "price_d0", "price_d1", "price_d6", "price_week",
    "price_d0_mean", "price_d0_max", "price_d0_min",
    "load", "wind", "solar", "net_load", "solar_share", "wind_share",
    "net_load_day_mean", "net_load_day_max", "net_load_day_min",
    "gas_price",
]
NEW_FEATURES = ["price_7d_mean", "price_7d_max", "net_load_d0", "net_load_change"]
BASE_PARAMS = {
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
}
REGULARIZED = {**BASE_PARAMS, "num_leaves": 15, "min_data_in_leaf": 500, "lambda_l2": 10.0}

# Candidate configurations, compared on 2023 by experiments.py. half_life is the
# age in years at which a training row's weight halves; None weights rows equally.
CONFIGS = {
    "base": dict(features=BASE_FEATURES, params=BASE_PARAMS, half_life=None),
    "features": dict(features=BASE_FEATURES + NEW_FEATURES, params=BASE_PARAMS, half_life=None),
    "recency": dict(features=BASE_FEATURES + NEW_FEATURES, params=BASE_PARAMS, half_life=2),
    "regularized": dict(features=BASE_FEATURES + NEW_FEATURES, params=REGULARIZED, half_life=None),
    "reg_recency": dict(features=BASE_FEATURES + NEW_FEATURES, params=REGULARIZED, half_life=2),
}
CONFIGS["final"] = CONFIGS["features"]


class Asinh:
    def __init__(self, y):
        self.c = np.median(y)
        self.s = np.median(np.abs(y - self.c))

    def forward(self, y):
        return np.arcsinh((y - self.c) / self.s)

    def inverse(self, z):
        return np.sinh(z) * self.s + self.c


def naive(df, column):
    return df[["origin", "date", "hour", "horizon", "price"]].assign(
        q10=np.nan, q50=df[column], q90=np.nan
    )


def weights(df, half_life):
    if half_life is None:
        return None
    age = df["date"].dt.year.max() - df["date"].dt.year
    return 0.5 ** (age / half_life)


def fit_probe(train, config, transform):
    """Fit on all but the last training year, stopping early on that year.

    Returns the round count per quantile and the z-space predictions for the
    held-out year, which double as the calibration set.
    """
    last = train["date"].dt.year.max()
    inner, held = train[train["date"].dt.year < last], train[train["date"].dt.year == last]
    feats = config["features"]
    rounds, z_held = {}, held[["origin", "date", "hour", "horizon"]].copy()
    z_held["z"] = transform.forward(held["price"].to_numpy())
    for col, q in QUANTILES.items():
        model = lgb.train(
            {**config["params"], "objective": "quantile", "alpha": q},
            lgb.Dataset(inner[feats], transform.forward(inner["price"]), weight=weights(inner, config["half_life"])),
            num_boost_round=3000,
            valid_sets=[lgb.Dataset(held[feats], transform.forward(held["price"]))],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        rounds[col] = model.best_iteration
        z_held[col] = model.predict(held[feats], num_iteration=model.best_iteration)
    return rounds, z_held


def fit_final(train, target, config, transform, rounds):
    feats = config["features"]
    out = target[["origin", "date", "hour", "horizon", "price"]].copy()
    models = {}
    for col, q in QUANTILES.items():
        models[col] = lgb.train(
            {**config["params"], "objective": "quantile", "alpha": q},
            lgb.Dataset(train[feats], transform.forward(train["price"]), weight=weights(train, config["half_life"])),
            rounds[col],
        )
        out[col] = models[col].predict(target[feats])
    out["z"] = transform.forward(target["price"].to_numpy())
    return out, models


def conformal_offsets(cal):
    """Asymmetric CQR offsets (Romano et al., 2019) from a calibration table in z space.

    The lower offset is the 90th percentile of how far actuals fell below q10, so
    that after shifting q10 down by it about 10% of hours land below; likewise for
    q90. Offsets can be negative, which narrows an interval that was too wide.
    """
    n = len(cal)
    level = min(1.0, (1 - 0.1) * (1 + 1 / n))
    return np.quantile(cal["q10"] - cal["z"], level), np.quantile(cal["z"] - cal["q90"], level)


def calibrate_static(pred, cal):
    out = pred.copy()
    for h in (1, 2):
        lo, hi = conformal_offsets(cal[cal["horizon"] == h])
        m = out["horizon"] == h
        out.loc[m, "q10"] -= lo
        out.loc[m, "q90"] += hi
    return out


def calibrate_rolling(pred, cal, window_days=90):
    """Recompute CQR offsets each origin day from the most recent realized errors.

    At origin D the only errors that may be used are those for target days on or
    before D, since later prices are not yet known at 9 AM. The pool starts with
    the held-out year's errors and grows with each realized validation error, and
    only the last `window_days` of target dates are used.
    """
    pool = pd.concat([cal, pred])
    out = pred.copy()
    for origin, idx in out.groupby("origin").groups.items():
        recent = pool[(pool["date"] <= origin) & (pool["date"] > origin - pd.Timedelta(days=window_days))]
        for h in (1, 2):
            lo, hi = conformal_offsets(recent[recent["horizon"] == h])
            rows = idx[out.loc[idx, "horizon"].to_numpy() == h]
            out.loc[rows, "q10"] -= lo
            out.loc[rows, "q90"] += hi
    return out


def to_price(pred, transform):
    out = pred.copy()
    cols = list(QUANTILES)
    # Independently fitted (and separately shifted) quantiles can cross; sorting
    # restores q10 <= q50 <= q90 (the rearrangement of Chernozhukov et al., 2010).
    out[cols] = np.sort(transform.inverse(out[cols].to_numpy()), axis=1)
    return out.drop(columns="z")


def run(df, config_name, target_split="val"):
    config = CONFIGS[config_name]
    train, target = df[df["split"] == "train"], df[df["split"] == target_split]
    transform = Asinh(train["price"].to_numpy())
    rounds, cal = fit_probe(train, config, transform)
    print(f"  {config_name} rounds: {rounds}", flush=True)
    pred, models = fit_final(train, target, config, transform, rounds)
    forecasts = {
        "naive_day": naive(target, "price_d0"),
        "naive_week": naive(target, "price_week"),
        "lightgbm": to_price(pred, transform),
        "lightgbm_cqr": to_price(calibrate_static(pred, cal), transform),
        "lightgbm_cqr_rolling": to_price(calibrate_rolling(pred, cal), transform),
    }
    return forecasts, models


def main(config_name="final"):
    df = pd.read_parquet(DATA / "dataset.parquet")
    spike = df.loc[df["split"] == "train", "price"].quantile(0.95)
    print(f"spike threshold (train 95th pct): ${spike:.2f}/MWh")
    forecasts, models = run(df, config_name)
    metrics = report(forecasts, spike)

    REPORTS.mkdir(exist_ok=True)
    metrics.to_csv(REPORTS / f"val_metrics_{config_name}.csv", index=False)
    preds = pd.concat(f.assign(model=name) for name, f in forecasts.items())
    preds.to_parquet(DATA / f"val_predictions_{config_name}.parquet")
    feats = CONFIGS[config_name]["features"]
    importance = pd.Series(models["q50"].feature_importance("gain"), feats)
    importance.sort_values(ascending=False).to_csv(REPORTS / f"feature_importance_{config_name}.csv")

    dm = []
    for h in (1, 2):
        for ref in ("naive_day", "naive_week"):
            stat, p = diebold_mariano(forecasts["lightgbm"], forecasts[ref], h)
            dm.append({"horizon": f"D+{h}", "versus": ref, "DM": stat, "p_value": p})
    dm = pd.DataFrame(dm)
    dm.to_csv(REPORTS / f"val_dm_{config_name}.csv", index=False)

    pd.set_option("display.width", 220)
    cols = ["model", "horizon", "regime", "MAE", "rMAE", "RMSE", "pinball", "coverage", "width", "winkler",
            "spike_prec_q50", "spike_rec_q50", "spike_prec_q90", "spike_rec_q90"]
    print(metrics[cols].round(3).to_string(index=False))
    print(dm.round(4).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "final")
