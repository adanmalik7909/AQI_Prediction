"""
feature_pipeline/run_pipeline.py
-----------------------------------
Main entry point for the feature pipeline.
Runs all 3 steps in order:
  1. fetch_data       -> get raw data from OpenWeather
  2. compute_features -> turn it into a clean feature row
  3. push_to_store    -> save that row into the Hopsworks Feature Store

This is the ONE file that GitHub Actions will call every hour
(see Phase 9 - Automation).
"""

from fetch_data import fetch_raw_data
from compute_features import compute_features
from push_to_store import push_feature_row


def main():
    print("Step 1: Fetching raw data...")
    raw = fetch_raw_data()

    print("Step 2: Computing features...")
    row = compute_features(raw)

    print("Step 3: Pushing to Feature Store...")
    push_feature_row(row)

    print("Done! Pipeline run completed successfully.")


if __name__ == "__main__":
    main()