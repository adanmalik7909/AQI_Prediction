"""
utils/data_source.py
---------------------
Loads the hourly weather + air-quality history that everything else is
built on, straight from Open-Meteo, and computes the EPA-correct AQI.

WHY NOT ALWAYS THE FEATURE STORE?
The Hopsworks feature group remains the project's registered store of
record (the hourly pipeline keeps writing to it). But it only holds the
five original weather columns, and reading it from Pakistan routinely
times out on Arrow Flight. The accuracy work needs:

  * the dispersion variables (boundary layer height, precipitation, dew
    point, radiation, wind direction, 100m wind) that were not in the
    original schema,
  * ~4 years of history rather than 2 (measured: 24h R2 0.55 -> 0.87 on
    the same feature set, purely from more winters to learn from),
  * a weather FORECAST for the future-weather features.

All of it comes from the same keyless Open-Meteo endpoints the project
already uses, so there is no new dependency and no new secret.

A local CSV cache keeps repeat runs fast; pass refresh=True to re-download.
"""

import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from config import LAT, LON
from openmeteo_client import (get_historical_weather, get_historical_air_quality,
                             get_recent_weather, get_recent_air_quality)
from aqi_daily import hourly_aqi_epa, dominant_pollutant

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data_cache")
CACHE_FILE = os.path.join(CACHE_DIR, "hourly_history.csv")

# Open-Meteo's air-quality archive (CAMS reanalysis) starts mid-2022.
HISTORY_START = "2022-09-01"
# The archive lags real time by a few days.
ARCHIVE_LAG_DAYS = 5
# Cache is considered stale after this many hours.
CACHE_MAX_AGE_HOURS = 12

# Open-Meteo hourly field name -> the column names used across this project
WEATHER_RENAME = {
    "time": "timestamp",
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "surface_pressure": "pressure",
    "wind_speed_10m": "wind_speed",
    "cloud_cover": "cloud_cover",
    "boundary_layer_height": "blh",
    "precipitation": "precipitation",
    "dew_point_2m": "dew_point",
    "wind_direction_10m": "wind_dir",
    "shortwave_radiation": "radiation",
    "wind_speed_100m": "wind_speed_100m",
}

AIR_RENAME = {
    "time": "timestamp",
    "pm2_5": "pm2_5",
    "pm10": "pm10",
    "ozone": "o3",
    "carbon_monoxide": "co",
    "sulphur_dioxide": "so2",
    "nitrogen_dioxide": "no2",
    "ammonia": "nh3",
    "dust": "dust",
    "aerosol_optical_depth": "aod",
}

# Without these we cannot compute the AQI at all
ESSENTIAL_COLS = ["pm2_5", "pm10", "o3", "co", "so2", "no2", "temperature", "wind_speed"]


def _to_frame(hourly, rename):
    df = pd.DataFrame(hourly).rename(columns=rename)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _merge(weather_hourly, air_hourly):
    """Join weather and air quality on the hour, then compute AQI."""
    weather = _to_frame(weather_hourly, WEATHER_RENAME)
    air = _to_frame(air_hourly, AIR_RENAME)

    df = pd.merge(weather, air, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # nh3 is 100% missing for Lahore - it carried no signal and its
    # presence forced defensive NaN handling downstream.
    df = df.drop(columns=[c for c in ["nh3"] if c in df.columns])

    df = df.dropna(subset=ESSENTIAL_COLS).reset_index(drop=True)

    # Reindex onto a strict hourly grid: lag/rolling features assume evenly
    # spaced rows, so a silent gap would quietly mislabel "24 hours ago".
    full_index = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    df = df.set_index("timestamp").reindex(full_index).rename_axis("timestamp").reset_index()

    aqi, subs = hourly_aqi_epa(df, return_breakdown=True)
    df["aqi"] = aqi
    df["dominant_pollutant"] = dominant_pollutant(subs)
    df["unix_time"] = (df["timestamp"].astype("int64") // 10**9).astype("int64")
    return df


def _cache_age_hours():
    if not os.path.exists(CACHE_FILE):
        return None
    mtime = datetime.fromtimestamp(os.path.getmtime(CACHE_FILE), tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime).total_seconds() / 3600


def load_history(refresh=False, start_date=HISTORY_START):
    """Full hourly history with AQI, for TRAINING.

    Uses the local CSV cache when it is fresh enough; otherwise downloads
    the whole range from Open-Meteo's archive.
    """
    age = _cache_age_hours()
    if not refresh and age is not None and age < CACHE_MAX_AGE_HOURS:
        df = pd.read_csv(CACHE_FILE, parse_dates=["timestamp"])
        print(f"Loaded {len(df)} hourly rows from cache "
              f"({df['timestamp'].min().date()} -> {df['timestamp'].max().date()}, "
              f"{age:.1f}h old)")
        return df

    end = (datetime.now(timezone.utc) - timedelta(days=ARCHIVE_LAG_DAYS)).strftime("%Y-%m-%d")
    print(f"Downloading hourly history {start_date} -> {end} from Open-Meteo...")

    weather = get_historical_weather(LAT, LON, start_date, end)
    air = get_historical_air_quality(LAT, LON, start_date, end)
    df = _merge(weather["hourly"], air["hourly"])

    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_csv(CACHE_FILE, index=False)
    print(f"  {len(df)} hourly rows "
          f"({df['timestamp'].min().date()} -> {df['timestamp'].max().date()}) "
          f"cached -> {CACHE_FILE}")
    return df


def latest_observed_index(df):
    """Index of the newest row that is genuinely fully observed.

    Subtle but important: hourly_aqi_epa's 24h window tolerates a few missing
    hours (min_periods=18), so `aqi` stays non-null for several hours past the
    last actual pollutant reading. Picking a prediction row by `aqi` alone
    therefore yields a row whose raw pollutant features are NaN. We require an
    observed pm2_5 as well.
    """
    valid = df["aqi"].notna() & df["pm2_5"].notna()
    if not valid.any():
        raise ValueError("No fully observed hour (AQI + pollutants) available")
    return int(valid[valid].index[-1])


def load_recent(past_days=10):
    """Recent history PLUS the weather forecast, for LIVE PREDICTION.

    The returned frame runs from `past_days` ago to ~4 days into the future.
    Rows up to 'now' have both pollutant and weather values; rows after that
    are forecast weather with empty pollutant columns, which is exactly what
    the future-weather features need. The caller predicts from the last row
    that has a valid AQI.
    """
    weather = get_recent_weather(LAT, LON, past_days=past_days)
    air = get_recent_air_quality(LAT, LON, past_days=past_days)

    wdf = _to_frame(weather["hourly"], WEATHER_RENAME)
    adf = _to_frame(air["hourly"], AIR_RENAME)

    # LEFT join on weather: weather extends further into the future than the
    # pollutant data we are willing to use, and we want to keep those hours.
    df = pd.merge(wdf, adf, on="timestamp", how="left")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.drop(columns=[c for c in ["nh3"] if c in df.columns])

    aqi, subs = hourly_aqi_epa(df, return_breakdown=True)
    df["aqi"] = aqi
    df["dominant_pollutant"] = dominant_pollutant(subs)
    df["unix_time"] = (df["timestamp"].astype("int64") // 10**9).astype("int64")
    return df


if __name__ == "__main__":
    hist = load_history()

    print("\ncolumns:", hist.columns.tolist())
    print(f"\nAQI: mean={hist['aqi'].mean():.1f} std={hist['aqi'].std():.1f} "
          f"min={hist['aqi'].min():.0f} max={hist['aqi'].max():.0f}")
    print("missing AQI rows:", int(hist["aqi"].isna().sum()))
    print("\ndominant pollutant share:")
    print(hist["dominant_pollutant"].value_counts(normalize=True).round(3).to_string())

    recent = load_recent()
    last_obs = recent["aqi"].last_valid_index()
    print(f"\nrecent frame: {len(recent)} rows, "
          f"{recent['timestamp'].min()} -> {recent['timestamp'].max()}")
    print(f"last hour with AQI: {recent.loc[last_obs, 'timestamp']} "
          f"(AQI {recent.loc[last_obs, 'aqi']:.0f})")
    print(f"forecast weather hours beyond that: {len(recent) - last_obs - 1}")
