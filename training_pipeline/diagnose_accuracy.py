"""
training_pipeline/diagnose_accuracy.py
----------------------------------------
DIAGNOSTIC script (not part of the automated pipeline).

Purpose: find out WHY the 24h R2 is only ~0.30, before changing any
model code. Instead of reading from Hopsworks (slow from Pakistan), it
rebuilds the same dataset straight from Open-Meteo's archive (keyless,
same source the feature pipeline uses), caches it to CSV, and then runs
controlled experiments:

  A. Is the target itself predictable? (autocorrelation at 24/48/72h)
  B. Does the CURRENT feature set + XGBoost reproduce R2 ~= 0.30?
  C. Does an EPA-CORRECT AQI target (24h-averaged PM2.5, 8h O3/CO -
     what the EPA formula actually requires) change the ceiling?
  D. Do longer lags / rolling windows / target transforms help?

Run:  python training_pipeline/diagnose_accuracy.py
"""

import os
import sys
import numpy as np
import pandas as pd
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from config import LAT, LON

CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "_diag_cache_lahore.csv")

WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_VARS = "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,cloud_cover"
AIR_QUALITY_VARS = "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone"

START_DATE = "2024-09-01"
END_DATE = "2026-08-25"


# ---------------------------------------------------------------- data
def _download():
    print(f"Downloading Open-Meteo archive {START_DATE} -> {END_DATE} ...")

    w = requests.get(WEATHER_ARCHIVE_URL, params={
        "latitude": LAT, "longitude": LON,
        "start_date": START_DATE, "end_date": END_DATE,
        "hourly": WEATHER_VARS, "timezone": "UTC",
    }, timeout=180).json()
    if "hourly" not in w:
        raise ValueError(f"weather error: {w}")

    a = requests.get(AIR_QUALITY_URL, params={
        "latitude": LAT, "longitude": LON,
        "start_date": START_DATE, "end_date": END_DATE,
        "hourly": AIR_QUALITY_VARS, "timezone": "UTC",
    }, timeout=180).json()
    if "hourly" not in a:
        raise ValueError(f"air quality error: {a}")

    wdf = pd.DataFrame(w["hourly"]).rename(columns={
        "time": "timestamp", "temperature_2m": "temperature",
        "relative_humidity_2m": "humidity", "surface_pressure": "pressure",
        "wind_speed_10m": "wind_speed", "cloud_cover": "cloud_cover",
    })
    adf = pd.DataFrame(a["hourly"]).rename(columns={
        "time": "timestamp", "carbon_monoxide": "co",
        "nitrogen_dioxide": "no2", "sulphur_dioxide": "so2", "ozone": "o3",
    })

    df = pd.merge(wdf, adf, on="timestamp", how="inner")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_csv(CACHE_PATH, index=False)
    print(f"  cached -> {CACHE_PATH}  ({len(df)} rows)")
    return df


def load_raw():
    if os.path.exists(CACHE_PATH):
        df = pd.read_csv(CACHE_PATH, parse_dates=["timestamp"])
        print(f"Loaded cached raw data ({len(df)} rows)")
        return df
    return _download()


# ------------------------------------------------------------ AQI calc
# Vectorised version of utils/aqi_calculator.py (SAME breakpoints), so we
# can compute AQI for 17k rows instantly and compare target definitions.
BP = {
    "pm2_5": [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
              (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
              (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)],
    "pm10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
             (255, 354, 151, 200), (355, 424, 201, 300),
             (425, 504, 301, 400), (505, 604, 401, 500)],
    "o3": [(0.000, 0.054, 0, 50), (0.055, 0.070, 51, 100), (0.071, 0.085, 101, 150),
           (0.086, 0.105, 151, 200), (0.106, 0.200, 201, 300)],
    "co": [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150),
           (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300),
           (30.5, 40.4, 301, 400), (40.5, 50.4, 401, 500)],
    "so2": [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150), (186, 304, 151, 200)],
    "no2": [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
            (361, 649, 151, 200), (650, 1249, 201, 300),
            (1250, 1649, 301, 400), (1650, 2049, 401, 500)],
}


def sub_aqi(conc, pollutant):
    """Vectorised EPA piecewise-linear sub-index. Concentrations above the
    highest breakpoint are clamped to 500 instead of returned as None."""
    conc = np.asarray(conc, dtype=float)
    out = np.full(conc.shape, np.nan)
    for lo, hi, alo, ahi in BP[pollutant]:
        m = (conc >= lo) & (conc <= hi)
        out[m] = (ahi - alo) / (hi - lo) * (conc[m] - lo) + alo
    top = BP[pollutant][-1][1]
    out[conc > top] = 500.0
    return out


def _ppm(c, mw):
    return np.asarray(c, dtype=float) * 24.45 / (mw * 1000)


def _ppb(c, mw):
    return np.asarray(c, dtype=float) * 24.45 / mw


def add_aqi_variants(df):
    """Adds BOTH target definitions so we can compare them head-to-head."""
    # --- (1) CURRENT project definition: instantaneous concentrations ---
    inst = pd.DataFrame({
        "pm2_5": sub_aqi(df["pm2_5"], "pm2_5"),
        "pm10": sub_aqi(df["pm10"], "pm10"),
        "o3": sub_aqi(_ppm(df["o3"], 48), "o3"),
        "co": sub_aqi(_ppm(df["co"], 28), "co"),
        "so2": sub_aqi(_ppb(df["so2"], 64), "so2"),
        "no2": sub_aqi(_ppb(df["no2"], 46), "no2"),
    })
    df["aqi_instant"] = inst.max(axis=1)

    # --- (2) EPA-CORRECT definition: proper averaging periods ---
    # PM2.5/PM10 -> 24h trailing mean, O3/CO -> 8h trailing mean,
    # SO2/NO2 -> 1h. This is what the EPA AQI actually means.
    epa = pd.DataFrame({
        "pm2_5": sub_aqi(df["pm2_5"].rolling(24, min_periods=18).mean(), "pm2_5"),
        "pm10": sub_aqi(df["pm10"].rolling(24, min_periods=18).mean(), "pm10"),
        "o3": sub_aqi(_ppm(df["o3"].rolling(8, min_periods=6).mean(), 48), "o3"),
        "co": sub_aqi(_ppm(df["co"].rolling(8, min_periods=6).mean(), 28), "co"),
        "so2": sub_aqi(_ppb(df["so2"], 64), "so2"),
        "no2": sub_aqi(_ppb(df["no2"], 46), "no2"),
    })
    df["aqi_epa"] = epa.max(axis=1)
    return df


if __name__ == "__main__":
    raw = load_raw()
    df = add_aqi_variants(raw)

    print("\n=== A. TARGET PROPERTIES ===")
    for col in ["aqi_instant", "aqi_epa"]:
        s = df[col].dropna()
        print(f"\n{col}:  n={len(s)}  mean={s.mean():.1f}  std={s.std():.1f}  "
              f"min={s.min():.0f}  max={s.max():.0f}")
        for lag in [1, 6, 12, 24, 48, 72]:
            print(f"   autocorr(+{lag}h) = {s.autocorr(lag):.3f}")

    print("\n=== hour-of-day profile ===")
    for col in ["aqi_instant", "aqi_epa"]:
        d = df.groupby(df["timestamp"].dt.hour)[col].mean()
        print(f"{col}: spread={d.max() - d.min():.1f} "
              f"(min {d.min():.0f} @ {d.idxmin()}h, max {d.max():.0f} @ {d.idxmax()}h)")

    print("\n=== dominant pollutant share (instant target) ===")
    inst = pd.DataFrame({
        "pm2_5": sub_aqi(df["pm2_5"], "pm2_5"),
        "pm10": sub_aqi(df["pm10"], "pm10"),
        "o3": sub_aqi(_ppm(df["o3"], 48), "o3"),
        "co": sub_aqi(_ppm(df["co"], 28), "co"),
        "so2": sub_aqi(_ppb(df["so2"], 64), "so2"),
        "no2": sub_aqi(_ppb(df["no2"], 46), "no2"),
    })
    print(inst.idxmax(axis=1).value_counts(normalize=True).round(3).to_string())
