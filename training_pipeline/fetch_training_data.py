"""
training_pipeline/fetch_training_data.py
--------------------------------------------
Fetches ALL historical (features, targets) from the Hopsworks Feature
Store, then builds the actual TRAINING dataset:

  1. Cleans up (drops 'nh3' - always empty for our location, no signal)

  2. Adds DERIVED features that need HISTORY (lag, rolling average,
     change rate) - these were deferred all the way back in Phase 2,
     because they need the full historical sequence, not a single
     snapshot. Now that we have 2 years of data, it's easy with pandas.

  3. Builds the actual FORECASTING targets. Since the project goal is
     to predict AQI for the next 1/2/3 days, we shift the 'aqi' column
     BACKWARD in time to create 3 target columns:
       target_24h -> AQI exactly 24 hours AFTER this row
       target_48h -> AQI exactly 48 hours AFTER this row
       target_72h -> AQI exactly 72 hours AFTER this row
     Rows near the very end of the dataset won't have all 3 targets
     available yet (no future data exists) - those rows are dropped.
"""

import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))

from hopsworks_client import get_feature_store
from config import HOPSWORKS_PROJECT_NAME, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION


def fetch_and_prepare_training_data():
    fs = get_feature_store(HOPSWORKS_PROJECT_NAME)
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    print("Reading data from Feature Store...")
    try:
        df = fg.read()
    except Exception as e:
        # The Query Service (Arrow Flight) connection can occasionally drop
        # mid-transfer on large reads - fall back to the older, slower but
        # more stable Hive-based read path.
        print(f"Fast read failed ({type(e).__name__}), retrying with a more stable method...")
        df = fg.read(read_options={"use_hive": True})

    # --- Clean up ---
    df = df.sort_values("unix_time").reset_index(drop=True)
    df = df.drop(columns=["nh3"])  # 100% missing for this location - no signal

    # --- Derived features that need HISTORY (deferred from Phase 2) ---
    df["aqi_lag_1"] = df["aqi"].shift(1)                          # previous hour's AQI
    df["aqi_change_rate"] = df["aqi"] - df["aqi_lag_1"]           # how fast AQI is moving
    df["aqi_rolling_mean_24h"] = df["aqi"].shift(1).rolling(window=24).mean()
    df["pm2_5_rolling_mean_24h"] = df["pm2_5"].shift(1).rolling(window=24).mean()

    # --- Forecasting targets: AQI N hours in the FUTURE ---
    df["target_24h"] = df["aqi"].shift(-24)
    df["target_48h"] = df["aqi"].shift(-48)
    df["target_72h"] = df["aqi"].shift(-72)

    # Drop rows missing history (start of dataset) or future (end of dataset)
    required_cols = [
        "aqi_lag_1", "aqi_rolling_mean_24h", "pm2_5_rolling_mean_24h",
        "target_24h", "target_48h", "target_72h",
    ]
    before = len(df)
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    after = len(df)

    print(f"Training-ready dataset: {after} rows ({before - after} dropped - missing history/future)")
    return df


# Quick manual test - only runs when this file is executed directly
if __name__ == "__main__":
    df = fetch_and_prepare_training_data()
    print("\nColumns:", df.columns.tolist())
    print("\nSample rows:")
    print(df[["timestamp", "aqi", "aqi_lag_1", "aqi_change_rate",
               "aqi_rolling_mean_24h", "target_24h", "target_48h", "target_72h"]].head())