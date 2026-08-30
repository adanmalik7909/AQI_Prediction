"""
feature_pipeline/compute_features.py
--------------------------------------
Turns raw data into the flat "feature rows" that go into the Feature Store.

Two entry points:

  compute_features(raw)      one row from a single live snapshot
  build_recent_rows(days)    a continuous window of hourly rows

WHY A WINDOW AND NOT JUST ONE ROW
The dashboard rebuilds lag/rolling features from the store, and the longest lag
is 168h - so it needs an unbroken week of hourly rows. Writing only the current
hour meant a single failed run (or a paused schedule) left a permanent hole that
nothing ever repaired. Because the feature group's primary key is
(city, unix_time), re-writing the last few days every hour is an idempotent
upsert: it costs little, and it heals gaps automatically.

NOTE ON THE `aqi` COLUMN (important, and documented for the report)
A single snapshot can only yield the INSTANTANEOUS EPA index - there is no 24h
window to average over. The accuracy investigation established that this
instantaneous value is too noisy to forecast (see report/ACCURACY_UPGRADE.md):
its 24h-ahead autocorrelation is 0.60 versus 0.77 for the properly averaged
index. So the modelling side does not train on the snapshot value.

build_recent_rows() DOES have a series in hand, so it stores the EPA-correct
`aqi` computed via utils/aqi_daily.hourly_aqi_epa() - the same function training
and serving use.
"""

import os
import sys
from datetime import datetime, timezone

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))


def compute_features(raw_data):
    """
    Takes the dict returned by fetch_data.fetch_raw_data() and returns
    a single flat dictionary (one "row") of features + target, ready
    to be pushed to the Feature Store.
    """
    weather = raw_data["weather"]
    air = raw_data["air_quality"]
    aqi_info = raw_data["calculated_aqi"]

    # Open-Meteo gives time as an ISO string like "2026-08-07T01:00"
    # (we requested timezone=UTC, so this IS UTC time already)
    dt_obj = datetime.fromisoformat(weather["time"]).replace(tzinfo=timezone.utc)
    dt_unix = int(dt_obj.timestamp())

    feature_row = {
        # --- Identifiers / timestamp ---
        "city": raw_data["city"],
        "timestamp": dt_obj.isoformat(),
        "unix_time": dt_unix,

        # --- Time-based features ---
        "hour": dt_obj.hour,
        "day": dt_obj.day,
        "month": dt_obj.month,
        "day_of_week": dt_obj.weekday(),   # 0=Monday ... 6=Sunday
        "is_weekend": 1 if dt_obj.weekday() >= 5 else 0,

        # --- Weather features (original five) ---
        "temperature": weather.get("temperature_2m"),
        "humidity": weather.get("relative_humidity_2m"),
        "pressure": weather.get("surface_pressure"),
        "wind_speed": weather.get("wind_speed_10m"),
        "cloud_cover": weather.get("cloud_cover"),

        # --- Dispersion meteorology ---
        # These control whether emissions accumulate or disperse, and were the
        # biggest feature-side accuracy gain. blh is the "lid" on the
        # atmosphere; when it collapses in winter the same emissions concentrate
        # into a much smaller volume.
        "blh": weather.get("boundary_layer_height"),
        "precipitation": weather.get("precipitation"),
        "dew_point": weather.get("dew_point_2m"),
        "wind_dir": weather.get("wind_direction_10m"),
        "radiation": weather.get("shortwave_radiation"),
        "wind_speed_100m": weather.get("wind_speed_100m"),

        # --- Pollutant concentration features ---
        "pm2_5": air.get("pm2_5"),
        "pm10": air.get("pm10"),
        "o3": air.get("ozone"),
        "co": air.get("carbon_monoxide"),
        "so2": air.get("sulphur_dioxide"),
        "no2": air.get("nitrogen_dioxide"),
        "nh3": air.get("ammonia") if air.get("ammonia") is not None else float("nan"),
        "dust": air.get("dust"),
        "aod": air.get("aerosol_optical_depth"),

        # --- Target variable (instantaneous - see module docstring) ---
        "aqi": aqi_info["overall_aqi"],
        "dominant_pollutant": aqi_info["dominant_pollutant"],
    }

    return feature_row


# Columns a Feature Store row carries, in a fixed order so every insert has an
# identical schema regardless of which entry point produced it.
#
# nh3 is deliberately absent: it is 100% missing for Lahore in the CAMS data, so
# an all-NaN column would only give Hopsworks a type to guess at and force
# defensive null handling downstream for no signal in return.
STORE_COLUMNS = [
    "city", "timestamp", "unix_time",
    "hour", "day", "month", "day_of_week", "is_weekend",
    "temperature", "humidity", "pressure", "wind_speed", "cloud_cover",
    "blh", "precipitation", "dew_point", "wind_dir", "radiation",
    "wind_speed_100m",
    "pm2_5", "pm10", "o3", "co", "so2", "no2", "dust", "aod",
    "aqi", "dominant_pollutant",
]



# The dtype each stored column must have, pinned to what the feature group is
# actually registered with (verified against fg.features).
#
# WHY THIS EXISTS - a real failure, not defensiveness
# `is_weekend` was built with `.astype(int)`. Python's bare `int` is int32 on
# Windows and int64 on Linux, which Hopsworks maps to 'int' and 'bigint'
# respectively. The feature group was registered from a Windows machine, so
# every GitHub Actions run (Linux) was rejected:
#
#   FeatureStoreException: is_weekend (expected type: 'int', derived from
#   input: 'bigint') has the wrong type
#
# Pinning the widths makes the insert schema identical everywhere. Any int-like
# column is also a hazard in the other direction: Open-Meteo occasionally
# returns a whole number where it usually sends a decimal (or vice versa), which
# would flip float64 <-> int64 and fail the same way.
STORE_DTYPES = {
    "city": "object", "timestamp": "object", "unix_time": "int64",
    "hour": "int32", "day": "int32", "month": "int32",
    "day_of_week": "int32", "is_weekend": "int32",
    "temperature": "float64", "humidity": "int64", "pressure": "float64",
    "wind_speed": "float64", "cloud_cover": "int64",
    "blh": "float64", "precipitation": "float64", "dew_point": "float64",
    "wind_dir": "int64", "radiation": "float64", "wind_speed_100m": "float64",
    "pm2_5": "float64", "pm10": "float64", "o3": "float64", "co": "float64",
    "so2": "float64", "no2": "float64", "dust": "float64", "aod": "float64",
    "aqi": "float64", "dominant_pollutant": "object",
}


def coerce_store_dtypes(df):
    """Force every column to its registered width. Applied at the insert itself,
    so it holds for the hourly pipeline and the historical backfill alike.

    Integer columns are rounded before casting: a NaN cannot be held in an int
    column, so a row missing an integer feature is dropped rather than silently
    turned into 0, which would read as a real measurement.
    """
    import pandas as pd

    out = df.copy()
    for col, dtype in STORE_DTYPES.items():
        if col not in out.columns:
            continue
        if dtype.startswith("int"):
            numeric = pd.to_numeric(out[col], errors="coerce")
            if numeric.isna().any():
                out = out[numeric.notna()]
                numeric = numeric.loc[out.index]
            out[col] = numeric.round().astype(dtype)
        elif dtype == "object":
            out[col] = out[col].astype(str)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(dtype)
    return out


def build_recent_rows(days=10):

    """A continuous window of hourly Feature Store rows, newest last.

    Only fully observed hours are returned - forecast-weather hours have no
    pollutant readings, and storing them would put unobserved values into the
    store of record.
    """
    import pandas as pd
    from config import CITY_NAME
    from data_source import load_recent, latest_observed_index

    df = load_recent(past_days=days)
    df = df.loc[:latest_observed_index(df)].copy()

    ts = pd.to_datetime(df["timestamp"])
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

    # A row without an AQI carries no target and cannot be used for training.
    df = df[df["aqi"].notna()]
    return df[STORE_COLUMNS].reset_index(drop=True)


# Quick manual test - only runs when this file is executed directly
if __name__ == "__main__":
    import json

    sys.path.append(os.path.dirname(__file__))
    from fetch_data import fetch_raw_data

    print("--- single snapshot ---")
    print(json.dumps(compute_features(fetch_raw_data()), indent=2))

    print("\n--- recent window ---")
    rows = build_recent_rows(days=10)
    print(f"{len(rows)} rows, {rows['timestamp'].iloc[0]} -> "
          f"{rows['timestamp'].iloc[-1]}")
    print(rows[["timestamp", "aqi", "pm2_5", "blh", "wind_speed"]].tail()
          .to_string(index=False))
