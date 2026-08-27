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
import time
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))

from hopsworks_client import get_feature_store
from config import HOPSWORKS_PROJECT_NAME, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION

MAX_RETRIES = 5
RETRY_DELAYS = [5, 10, 20, 40, 60]


def _fetch_from_hopsworks():
    """Fetches raw data from Hopsworks Feature Store with robust retry logic.

    Arrow Flight (the default fast read path) frequently times out from
    Pakistan due to ISP latency to EU-West servers. We use the slower but
    more reliable Hive/JDBC path as fallback, and catch ALL exceptions
    (not just ConnectionError) since Hopsworks wraps timeouts in its own
    FeatureStoreException.
    """
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            fs = get_feature_store(HOPSWORKS_PROJECT_NAME)
            fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

            print(f"Reading data from Feature Store (attempt {attempt + 1}/{MAX_RETRIES})...")

            # Try fast Arrow Flight first, then fallback to Hive
            try:
                df = fg.read()
                print("  ✓ Fast read succeeded.")
                return df
            except Exception as e1:
                print(f"  Fast read failed ({type(e1).__name__}), trying Hive fallback...")
                try:
                    df = fg.read(read_options={"use_hive": True})
                    print("  ✓ Hive read succeeded.")
                    return df
                except Exception as e2:
                    raise e2  # let outer retry handle it

        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                print(f"  ✗ Attempt {attempt + 1}/{MAX_RETRIES} failed: "
                      f"{type(e).__name__}. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"  ✗ All {MAX_RETRIES} attempts failed.")

    raise last_error


def fetch_engineered_data():
    """
    Reads from the Feature Store and adds lag/rolling engineered features.
    Does NOT build forecast targets or drop any rows - shared by:
      - fetch_and_prepare_training_data() below (adds targets afterwards)
      - webapp/app.py (just needs the most recent row's engineered
        features to make a live prediction - it has no "future" targets)
    """
    df = _fetch_from_hopsworks()

    # --- Clean up ---
    df = df.sort_values("unix_time").reset_index(drop=True)
    df = df.drop(columns=["nh3"])  # 100% missing for this location - no signal

    # ================================================================
    # DERIVED FEATURES — Phase 2 deferred + Accuracy Improvement Phase
    # ================================================================

    # --- AQI Lags: give the model a "memory" of recent AQI ---
    df["aqi_lag_1"]  = df["aqi"].shift(1)                # 1 hour ago
    df["aqi_lag_3"]  = df["aqi"].shift(3)                # 3 hours ago
    df["aqi_lag_6"]  = df["aqi"].shift(6)                # 6 hours ago
    df["aqi_lag_12"] = df["aqi"].shift(12)               # 12 hours ago
    df["aqi_lag_24"] = df["aqi"].shift(24)               # same hour yesterday

    # --- AQI change rate ---
    df["aqi_change_rate"] = df["aqi"] - df["aqi_lag_1"]

    # --- AQI rolling statistics (shifted by 1 to avoid data leakage) ---
    aqi_shifted = df["aqi"].shift(1)
    df["aqi_rolling_mean_24h"] = aqi_shifted.rolling(window=24).mean()
    df["aqi_rolling_std_24h"]  = aqi_shifted.rolling(window=24).std()   # volatility
    df["aqi_rolling_min_24h"]  = aqi_shifted.rolling(window=24).min()   # 24h floor
    df["aqi_rolling_max_24h"]  = aqi_shifted.rolling(window=24).max()   # 24h ceiling

    # --- AQI deviation from rolling mean (is it abnormally high/low?) ---
    df["aqi_deviation_24h"] = df["aqi"] - df["aqi_rolling_mean_24h"]

    # --- PM2.5 features (dominant pollutant in Lahore) ---
    df["pm2_5_lag_1"]  = df["pm2_5"].shift(1)
    df["pm2_5_lag_24"] = df["pm2_5"].shift(24)
    df["pm2_5_rolling_mean_24h"] = df["pm2_5"].shift(1).rolling(window=24).mean()

    # --- PM10 rolling ---
    df["pm10_rolling_mean_24h"] = df["pm10"].shift(1).rolling(window=24).mean()

    # --- PM ratio: PM2.5/PM10 indicates pollution source type ---
    df["pm_ratio"] = df["pm2_5"] / (df["pm10"] + 0.01)

    # --- Weather-derived features ---
    df["wind_speed_rolling_mean_6h"] = df["wind_speed"].shift(1).rolling(6).mean()
    df["temp_change_rate"] = df["temperature"] - df["temperature"].shift(1)
    df["wind_humidity_interaction"] = df["wind_speed"] * df["humidity"]

    # --- Cyclical time encoding (sin/cos preserves circular relationships) ---
    import numpy as np
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

    return df



def fetch_and_prepare_training_data():
    df = fetch_engineered_data()

    # --- Forecasting targets: AQI N hours in the FUTURE ---
    df["target_24h"] = df["aqi"].shift(-24)
    df["target_48h"] = df["aqi"].shift(-48)
    df["target_72h"] = df["aqi"].shift(-72)

    # Drop rows missing history (start of dataset) or future (end of dataset)
    # Must include ALL columns that use .shift() or .rolling() with large windows
    required_cols = [
        "aqi_lag_24", "aqi_rolling_mean_24h", "aqi_rolling_std_24h",
        "pm2_5_rolling_mean_24h", "pm2_5_lag_24", "pm10_rolling_mean_24h",
        "wind_speed_rolling_mean_6h", "temp_change_rate",
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