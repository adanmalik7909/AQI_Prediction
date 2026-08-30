"""
backfill/backfill_historical.py
----------------------------------
Fills the Hopsworks Feature Store with the full hourly history, so the feature
group is a genuine store of record rather than just a rolling recent window.

It reuses utils/data_source.load_history() - the SAME loader the training
pipeline uses - so the stored rows carry the identical columns and the identical
EPA-correct AQI. An earlier version of this script re-implemented the merge by
hand and produced the instantaneous AQI plus only the five original weather
columns, which is exactly the drift a feature store is supposed to prevent.

Run: python backfill/backfill_historical.py

Open-Meteo's archive lags real time by a few days; load_history() already leaves
that buffer. Inserts are chunked because a single ~35k-row insert is slow and an
interrupted one leaves no useful progress behind.
"""

import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "feature_pipeline"))

from config import CITY_NAME, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION
from data_source import load_history
from compute_features import STORE_COLUMNS
from push_to_store import push_feature_rows

CHUNK_ROWS = 5000


def build_historical_rows(refresh=False):
    """Full hourly history shaped exactly like the hourly pipeline's rows."""
    df = load_history(refresh=refresh)

    ts = pd.to_datetime(df["timestamp"])
    df = df.copy()
    df["city"] = CITY_NAME
    df["hour"] = ts.dt.hour
    df["day"] = ts.dt.day
    df["month"] = ts.dt.month
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    df["timestamp"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    for col in STORE_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")

    # Rows without an AQI have no target and cannot be trained on. The first
    # ~24 hours always drop out here: the EPA index needs a 24h window before
    # it can be computed at all.
    before = len(df)
    df = df[df["aqi"].notna()]
    print(f"Prepared {len(df)} rows ({before - len(df)} dropped - no AQI yet)")

    return df[STORE_COLUMNS].reset_index(drop=True)


def main():
    rows = build_historical_rows()

    if rows.empty:
        print("No rows to insert - check the archive response.")
        return 1

    print(f"Backfilling {len(rows)} hourly rows into "
          f"'{FEATURE_GROUP_NAME}' v{FEATURE_GROUP_VERSION} "
          f"({rows['timestamp'].iloc[0]} -> {rows['timestamp'].iloc[-1]}) "
          f"in chunks of {CHUNK_ROWS}...")

    for start in range(0, len(rows), CHUNK_ROWS):
        chunk = rows.iloc[start:start + CHUNK_ROWS]
        print(f"  rows {start}-{start + len(chunk) - 1}...")
        push_feature_rows(chunk)

    print("Backfill complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
