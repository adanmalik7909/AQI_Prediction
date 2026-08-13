"""
backfill/backfill_historical.py
----------------------------------
Fetches ~2 years of HOURLY historical weather + air quality data from
Open-Meteo, converts every hour into a feature row (same format as
fetch_data.py + compute_features.py produce for live data), and pushes
ALL of them into the SAME Hopsworks feature group in one bulk insert.

Run: python backfill/backfill_historical.py

NOTE: Open-Meteo's archive/reanalysis data usually has a few days of
processing delay, so we don't request all the way up to "today" -
we leave a small buffer (END_DATE_LAG_DAYS) before the end date.
"""

import sys
import os
import json
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "feature_pipeline"))

from openmeteo_client import get_historical_weather, get_historical_air_quality
from aqi_calculator import calculate_aqi_from_concentrations
from hopsworks_client import get_feature_store
from config import (
    LAT, LON, CITY_NAME,
    HOPSWORKS_PROJECT_NAME, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION,
)
from compute_features import compute_features

BACKFILL_DAYS = 730          # ~2 years
END_DATE_LAG_DAYS = 5        # leave a buffer - very recent data may not be processed yet


def build_date_range():
    end_date = datetime.now(timezone.utc) - timedelta(days=END_DATE_LAG_DAYS)
    start_date = end_date - timedelta(days=BACKFILL_DAYS)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def build_historical_rows(start_date, end_date):
    """
    Fetches historical weather + air quality, merges them by matching
    timestamp, and runs each hour through compute_features() - exactly
    like the live pipeline does for a single snapshot.
    """
    print(f"Fetching historical weather ({start_date} to {end_date})...")
    weather_data = get_historical_weather(LAT, LON, start_date, end_date)

    print(f"Fetching historical air quality ({start_date} to {end_date})...")
    air_data = get_historical_air_quality(LAT, LON, start_date, end_date)

    w_hourly = weather_data["hourly"]
    a_hourly = air_data["hourly"]

    # Index air quality data by timestamp, for fast lookup while looping weather
    air_by_time = {}
    for i, t in enumerate(a_hourly["time"]):
        air_by_time[t] = {
            "pm2_5": a_hourly["pm2_5"][i],
            "pm10": a_hourly["pm10"][i],
            "o3": a_hourly["ozone"][i],
            "co": a_hourly["carbon_monoxide"][i],
            "so2": a_hourly["sulphur_dioxide"][i],
            "no2": a_hourly["nitrogen_dioxide"][i],
            "nh3": a_hourly["ammonia"][i],
        }

    rows = []
    skipped = 0

    for i, t in enumerate(w_hourly["time"]):
        if t not in air_by_time:
            skipped += 1
            continue  # no matching air quality reading for this hour, skip it

        pollutants = air_by_time[t]
        components = {k: v for k, v in pollutants.items() if k != "nh3"}

        # Skip hours where core pollutant data is missing (can't compute AQI)
        if components.get("pm2_5") is None or components.get("pm10") is None:
            skipped += 1
            continue

        overall_aqi, dominant_pollutant, breakdown = calculate_aqi_from_concentrations(components)

        # Build the same raw_data shape that fetch_data.py produces for live data,
        # so we can reuse compute_features() unchanged.
        raw_row = {
            "city": CITY_NAME,
            "lat": LAT,
            "lon": LON,
            "weather": {
                "time": t,
                "temperature_2m": w_hourly["temperature_2m"][i],
                "relative_humidity_2m": w_hourly["relative_humidity_2m"][i],
                "surface_pressure": w_hourly["surface_pressure"][i],
                "wind_speed_10m": w_hourly["wind_speed_10m"][i],
                "cloud_cover": w_hourly["cloud_cover"][i],
            },
            "air_quality": {
                "time": t,
                "pm2_5": pollutants["pm2_5"],
                "pm10": pollutants["pm10"],
                "ozone": pollutants["o3"],
                "carbon_monoxide": pollutants["co"],
                "sulphur_dioxide": pollutants["so2"],
                "nitrogen_dioxide": pollutants["no2"],
                "ammonia": pollutants["nh3"],
            },
            "calculated_aqi": {
                "overall_aqi": overall_aqi,
                "dominant_pollutant": dominant_pollutant,
                "breakdown": breakdown,
            },
        }

        rows.append(compute_features(raw_row))

    print(f"Built {len(rows)} hourly feature rows ({skipped} hours skipped due to missing data).")
    return rows


def push_historical_rows(rows):
    """Bulk-insert all historical rows into the SAME feature group used by the live pipeline."""
    df = pd.DataFrame(rows)

    fs = get_feature_store(HOPSWORKS_PROJECT_NAME)
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly AQI, weather, and pollutant features",
        primary_key=["city", "unix_time"],
        event_time="unix_time",
        online_enabled=False,
        time_travel_format="HUDI",
    )

    print(f"Inserting {len(df)} rows into feature group '{FEATURE_GROUP_NAME}' (v{FEATURE_GROUP_VERSION})...")
    fg.insert(df)
    print("Backfill complete!")


if __name__ == "__main__":
    start_date, end_date = build_date_range()
    rows = build_historical_rows(start_date, end_date)

    if rows:
        push_historical_rows(rows)
    else:
        print("No rows to insert - check the date range and API responses.")