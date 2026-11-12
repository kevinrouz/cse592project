"""Compare model configurations on 2023, with each one fit on 2019-2022.

Selecting a configuration by its 2024 validation score would tune the model to
the year used to report results, so the choice is made on the last training year
instead, using the same probe fit that train.py runs for early stopping.

Writes reports/selection_2023.csv.
"""

import numpy as np
import pandas as pd

from metrics import QUANTILES, pinball
from train import CONFIGS, DATA, REPORTS, Asinh, fit_probe


def main():
    df = pd.read_parquet(DATA / "dataset.parquet")
    train = df[df["split"] == "train"]
    transform = Asinh(train["price"].to_numpy())
    rows = []
    for name in ["base", "features", "recency", "regularized", "reg_recency"]:
        rounds, held = fit_probe(train, CONFIGS[name], transform)
        y = transform.inverse(held["z"].to_numpy())
        q = {c: transform.inverse(held[c].to_numpy()) for c in QUANTILES}
        rows.append(
            {
                "config": name,
                "MAE": np.mean(np.abs(y - q["q50"])),
                "pinball": np.mean([pinball(y, q[c], a) for c, a in QUANTILES.items()]),
                "coverage": np.mean((y >= q["q10"]) & (y <= q["q90"])),
                **{f"rounds_{c}": r for c, r in rounds.items()},
            }
        )
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "selection_2023.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
