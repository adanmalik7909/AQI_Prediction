"""
feature_pipeline/run_pipeline.py
-----------------------------------
Main entry point for the FEATURE pipeline. GitHub Actions calls this once an
hour (.github/workflows/feature_pipeline.yml).

  1. build_recent_rows   -> a continuous 10-day window of hourly feature rows
                            from Open-Meteo, with the EPA-correct AQI
  2. push_feature_rows   -> upsert them into the Hopsworks Feature Store

WHY A WINDOW RATHER THAN THE CURRENT HOUR
The dashboard rebuilds 168h lag features from the store, so it needs an unbroken
week of rows. Writing one row per run meant any missed run left a permanent hole.
The primary key (city, unix_time) makes re-writing recent hours an upsert, so a
rolling window is idempotent and self-healing - at the cost of a slightly larger
insert once an hour.
"""

import sys

from compute_features import build_recent_rows
from push_to_store import push_feature_rows

# 10 days covers the longest lag (168h) with margin for archive lag.
WINDOW_DAYS = 10


def main():
    print(f"Step 1: Building the last {WINDOW_DAYS} days of hourly features...")
    rows = build_recent_rows(days=WINDOW_DAYS)

    if rows.empty:
        # Exit non-zero so the workflow surfaces this rather than reporting a
        # green run that quietly stored nothing.
        print("No observed hours were returned - nothing to store.")
        return 1

    print(f"  {len(rows)} rows: {rows['timestamp'].iloc[0]} -> "
          f"{rows['timestamp'].iloc[-1]}")
    print(f"  latest AQI: {rows['aqi'].iloc[-1]:.0f} "
          f"({rows['dominant_pollutant'].iloc[-1]})")

    print("Step 2: Upserting into the Hopsworks Feature Store...")
    push_feature_rows(rows)

    print("Done! Feature pipeline run completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
