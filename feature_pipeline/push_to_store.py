"""
feature_pipeline/push_to_store.py
------------------------------------
Writes feature rows into the Hopsworks Feature Store.

WHAT IS A "FEATURE GROUP"?
Think of it like a table in a database. It has:
  - a name + version (so you can evolve it over time)
  - a primary key (uniquely identifies each row - here: city + unix_time)
  - an event_time column (tells Hopsworks WHEN this row happened,
    which is what makes time-travel / point-in-time queries possible)

get_or_create_feature_group() creates it the very first time and reuses it
afterwards.

Because the primary key is (city, unix_time), re-inserting an hour that is
already stored is an UPSERT rather than a duplicate. That is what lets the
hourly pipeline write a rolling window instead of a single row, and thereby
repair any gap left by a missed run.
"""

import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))

from hopsworks_client import get_feature_store
from config import HOPSWORKS_PROJECT_NAME, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION

FEATURE_GROUP_DESCRIPTION = (
    "Hourly Lahore air quality: pollutant concentrations, weather, dispersion "
    "meteorology (boundary layer height, precipitation, dew point, radiation, "
    "wind direction/100m wind), calendar fields, and the EPA AQI"
)


def get_or_create_fg(fs):
    """The one place the feature group is defined, so the hourly pipeline and
    the historical backfill cannot drift into declaring it differently."""
    return fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description=FEATURE_GROUP_DESCRIPTION,
        primary_key=["city", "unix_time"],
        event_time="unix_time",
        online_enabled=False,
        time_travel_format="HUDI",
    )


def push_feature_rows(rows):
    """Insert a DataFrame (or list of dicts) of feature rows."""
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    if df.empty:
        print("Nothing to insert - no feature rows were produced.")
        return

    fs = get_feature_store(HOPSWORKS_PROJECT_NAME)
    fg = get_or_create_fg(fs)

    fg.insert(df)
    print(f"Inserted {len(df)} row(s) into feature group "
          f"'{FEATURE_GROUP_NAME}' (v{FEATURE_GROUP_VERSION})")


def push_feature_row(feature_row):
    """Single-row convenience wrapper, kept for the snapshot path."""
    push_feature_rows([feature_row])


# Quick manual test - only runs when this file is executed directly
if __name__ == "__main__":
    sys.path.append(os.path.dirname(__file__))
    from compute_features import build_recent_rows

    push_feature_rows(build_recent_rows(days=10))
