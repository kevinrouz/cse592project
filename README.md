# 48-hour day-ahead price forecasting for ERCOT North Hub

This project forecasts the 48 hourly day-ahead prices at ERCOT's HB_NORTH hub for the two days
after a 9 AM Central forecast time, along with a 10th-90th percentile interval for each hour.

## Setup

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
```

The forecast download needs a free ERCOT Public API account from https://apiexplorer.ercot.com/.
Put the credentials in `.env`, which git ignores:

```
ERCOT_API_USERNAME=...
ERCOT_API_PASSWORD=...
ERCOT_PUBLIC_API_SUBSCRIPTION_KEY=...
```

## Pipeline

Run the scripts from `src/` in this order. Each one writes to `data/` or `reports/`.

1. `fetch_prices.py` downloads day-ahead hub prices (2018-2025) and the Henry Hub gas price. It
   needs no API key and takes about a minute.
2. `fetch_forecasts.py` downloads ERCOT's archived load, wind and solar forecasts, keeping the last
   vintage posted before 9 AM each day. It takes about 20 minutes, writes one file per report and
   year, and skips files that already exist, so an interrupted run can be restarted.
3. `dataset.py` joins everything into one row per (origin day, target hour) and assigns rows to
   train (2019-2023), validation (2024) and test (2025) by the target date.
4. `train.py` fits the naive baselines and the LightGBM quantile models and writes the validation
   metrics to `reports/val_metrics.csv`. Pass `--raw` to train on untransformed prices.
5. `plots.py` and `eda.py` write the figures in `reports/figures/`.

The 2025 test year is not scored anywhere yet. It is reserved for the final evaluation.
