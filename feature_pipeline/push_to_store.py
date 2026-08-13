"""
feature_pipeline/push_to_store.py
------------------------------------
Takes a single feature row (dict, from compute_features.py) and
pushes it into the Hopsworks Feature Store.

WHAT IS A "FEATURE GROUP"?
Think of it like a table in a database. It has:
  - a name + version (so you can evolve it over time)
  - a primary key (uniquely identifies each row - here: city + unix_time)
  - an event_time column (tells Hopsworks WHEN this row happened,
    which is what makes time-travel / point-in-time queries possible)

get_or_create_feature_group() will:
  - create the feature group the VERY FIRST time this runs
  - simply reuse it every time after that
"""

import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))

from hopsworks_client import get_feature_store
from config import HOPSWORKS_PROJECT_NAME, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION


def push_feature_row(feature_row):
    """
    Takes a single feature dict (one row) and inserts it into the
    Hopsworks Feature Store, creating the feature group if needed.
    """
    fs = get_feature_store(HOPSWORKS_PROJECT_NAME)

    # Hopsworks expects a DataFrame, not a raw dict - wrap it in a list
    # so pandas creates a DataFrame with exactly ONE row.
    df = pd.DataFrame([feature_row])

    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly AQI, weather, and pollutant features",
        primary_key=["city", "unix_time"],
        event_time="unix_time",
        online_enabled=False,
        time_travel_format="HUDI",
    )

    fg.insert(df)
    print(f"Inserted 1 row into feature group '{FEATURE_GROUP_NAME}' (v{FEATURE_GROUP_VERSION})")


# Quick manual test - only runs when this file is executed directly
if __name__ == "__main__":
    sys.path.append(os.path.dirname(__file__))
    from fetch_data import fetch_raw_data
    from compute_features import compute_features

    raw = fetch_raw_data()
    row = compute_features(raw)
    push_feature_row(row)