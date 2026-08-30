"""
training_pipeline/diagnose_experiments3.py
-------------------------------------------
Round 3. What rounds 1-2 established:

  * The CURRENT target ("instantaneous" EPA formula applied to a single
    hourly concentration) has autocorr(+24h)=0.60 and is largely noise.
    Ceiling with any model ~= R2 0.24-0.37.
  * Applying the EPA averaging periods the formula actually requires
    (PM 24h mean, O3/CO 8h mean) gives autocorr(+24h)=0.73 and lifts the
    same pipeline to R2 0.55-0.66. This is a TARGET-DEFINITION fix, not a
    model fix - and it makes aqi_epa.shift(-24/-48/-72) mean exactly
    "day 1 / day 2 / day 3 AQI", which is what the dashboard claims.
  * Future WEATHER (Open-Meteo forecast API, free) adds ~+0.09 R2.

Round 3 tests the two remaining levers:
  L1  more history: archive reaches back to 2022-09 -> ~4 years, not 2
  L2  dispersion meteorology: boundary_layer_height, precipitation,
      wind_direction, dew_point, radiation, 100m wind. BLH in particular
      is the dominant physical driver of PM2.5 build-up in winter Lahore.

NOTE on why we do NOT use Open-Meteo's pollutant FORECAST as a feature:
the air-quality API's forecast comes from the SAME CAMS model that
produces our archived pollutant values, so our target is derived from it.
Feeding it back in would be circular and would produce a fake R2. Only
weather forecasts (a different model) are used.

Run:  python training_pipeline/diagnose_experiments3.py
"""

import os
import sys
import numpy as np
import pandas as pd
import requests

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from config import LAT, LON
from diagnose_accuracy import add_aqi_variants
from diagnose_experiments2 import expanding_backtest

CACHE = os.path.join(os.path.dirname(__file__), "..", "_diag_cache_long.csv")
START_DATE = "2022-09-01"
END_DATE = "2026-08-25"

WEATHER_VARS = ("temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,"
                "cloud_cover,boundary_layer_height,precipitation,wind_direction_10m,"
                "dew_point_2m,shortwave_radiation,wind_speed_100m")
AQ_VARS = "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,dust,aerosol_optical_depth"

RENAME_W = {
    "time": "timestamp", "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity", "surface_pressure": "pressure",
    "wind_speed_10m": "wind_speed", "boundary_layer_height": "blh",
    "wind_direction_10m": "wind_dir", "dew_point_2m": "dew_point",
    "shortwave_radiation": "radiation", "wind_speed_100m": "wind_speed_100m",
}
RENAME_A = {"time": "timestamp", "carbon_monoxide": "co",
            "nitrogen_dioxide": "no2", "sulphur_dioxide": "so2", "ozone": "o3",
            "aerosol_optical_depth": "aod"}


def load_long():
    if os.path.exists(CACHE):
        df = pd.read_csv(CACHE, parse_dates=["timestamp"])
        print(f"Loaded cached long dataset ({len(df)} rows, "
              f"{df['timestamp'].min().date()} -> {df['timestamp'].max().date()})")
        return df

    print(f"Downloading {START_DATE} -> {END_DATE} with extended variables ...")
    w = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": LAT, "longitude": LON, "start_date": START_DATE,
        "end_date": END_DATE, "hourly": WEATHER_VARS, "timezone": "UTC",
    }, timeout=300).json()
    a = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params={
        "latitude": LAT, "longitude": LON, "start_date": START_DATE,
        "end_date": END_DATE, "hourly": AQ_VARS, "timezone": "UTC",
    }, timeout=300).json()
    if "hourly" not in w or "hourly" not in a:
        raise ValueError(f"download failed: {w.get('reason')} / {a.get('reason')}")

    wdf = pd.DataFrame(w["hourly"]).rename(columns=RENAME_W)
    adf = pd.DataFrame(a["hourly"]).rename(columns=RENAME_A)
    df = pd.merge(wdf, adf, on="timestamp", how="inner")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.dropna(subset=["pm2_5", "temperature"]).reset_index(drop=True)
    df.to_csv(CACHE, index=False)
    print(f"  cached -> {CACHE} ({len(df)} rows)")
    return df


# ------------------------------------------------------------- features
POLLUTANTS = ["pm2_5", "pm10", "o3", "co", "no2", "so2", "dust", "aod"]
WEATHER = ["temperature", "humidity", "pressure", "wind_speed", "cloud_cover",
           "blh", "precipitation", "dew_point", "radiation", "wind_speed_100m"]


def build_features(df, use_dispersion=True):
    """Full engineered feature frame, built with pd.concat to stay fast."""
    d = df.copy()
    ts = d["timestamp"]
    new = {}

    # --- calendar / cyclical ---
    new["hour"] = ts.dt.hour
    new["day"] = ts.dt.day
    new["month"] = ts.dt.month
    new["day_of_week"] = ts.dt.dayofweek
    new["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    doy = ts.dt.dayofyear
    new["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    new["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    new["month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12)
    new["month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12)
    new["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    new["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    new["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    new["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # --- AQI memory ---
    a = d["aqi"]
    a1 = a.shift(1)
    for lag in [1, 3, 6, 12, 24, 36, 48, 72, 96, 120, 168]:
        new[f"aqi_lag_{lag}"] = a.shift(lag)
    new["aqi_change_rate"] = a - a.shift(1)
    new["aqi_change_24h"] = a - a.shift(24)
    for w in [3, 6, 12, 24, 48, 72, 168]:
        new[f"aqi_rolling_mean_{w}h"] = a1.rolling(w).mean()
    for w in [24, 48, 72]:
        new[f"aqi_rolling_std_{w}h"] = a1.rolling(w).std()
        new[f"aqi_rolling_min_{w}h"] = a1.rolling(w).min()
        new[f"aqi_rolling_max_{w}h"] = a1.rolling(w).max()
    new["aqi_ema_12h"] = a1.ewm(span=12).mean()
    new["aqi_ema_48h"] = a1.ewm(span=48).mean()
    new["aqi_deviation_24h"] = a - a1.rolling(24).mean()
    new["aqi_trend_24h"] = a1.rolling(24).mean() - a1.rolling(48).mean()
    new["aqi_trend_72h"] = a1.rolling(24).mean() - a1.rolling(72).mean()

    # --- pollutant history ---
    for col in POLLUTANTS:
        if col not in d.columns:
            continue
        s = d[col]
        for lag in [1, 6, 12, 24, 48]:
            new[f"{col}_lag_{lag}"] = s.shift(lag)
        s1 = s.shift(1)
        new[f"{col}_rolling_mean_24h"] = s1.rolling(24).mean()
        new[f"{col}_rolling_mean_72h"] = s1.rolling(72).mean()
        new[f"{col}_rolling_std_24h"] = s1.rolling(24).std()
    new["pm_ratio"] = d["pm2_5"] / (d["pm10"] + 0.01)
    new["pollution_intensity"] = (d["pm2_5"] + d["pm10"]) / 2

    # --- weather history ---
    wcols = WEATHER if use_dispersion else ["temperature", "humidity", "pressure",
                                            "wind_speed", "cloud_cover"]
    for col in wcols:
        if col not in d.columns:
            continue
        s = d[col]
        new[f"{col}_rolling_mean_24h"] = s.shift(1).rolling(24).mean()
        new[f"{col}_change_24h"] = s - s.shift(24)
    new["wind_humidity_interaction"] = d["wind_speed"] * d["humidity"]
    new["temp_humidity_interaction"] = d["temperature"] * d["humidity"]
    new["temp_change_rate"] = d["temperature"] - d["temperature"].shift(1)

    if use_dispersion:
        # physical dispersion / stagnation proxies
        new["ventilation_index"] = d["blh"] * d["wind_speed"]     # classic VI
        new["ventilation_index_24h"] = (d["blh"] * d["wind_speed"]).shift(1).rolling(24).mean()
        new["stagnation"] = 1.0 / (d["blh"] * d["wind_speed"] + 1.0)
        new["inversion_proxy"] = d["temperature"] - d["dew_point"]
        new["wind_dir_sin"] = np.sin(np.deg2rad(d["wind_dir"]))
        new["wind_dir_cos"] = np.cos(np.deg2rad(d["wind_dir"]))
        new["precip_24h"] = d["precipitation"].shift(1).rolling(24).sum()
        new["precip_72h"] = d["precipitation"].shift(1).rolling(72).sum()
        new["blh_min_24h"] = d["blh"].shift(1).rolling(24).min()

    out = pd.concat([d, pd.DataFrame(new, index=d.index)], axis=1)
    return out.copy()


def add_future_weather(d, horizon, use_dispersion=True):
    """Weather between now and t+h - in production these come from the
    Open-Meteo weather FORECAST API (free, 16 days), so it is a genuine
    production-available input, not leakage."""
    cols = WEATHER if use_dispersion else ["temperature", "humidity", "pressure",
                                           "wind_speed", "cloud_cover"]
    new = {}
    for col in cols:
        if col not in d.columns:
            continue
        fut = d[col].shift(-horizon)
        new[f"f_{col}_t"] = fut
        new[f"f_{col}_mean"] = fut.rolling(horizon, min_periods=horizon // 2).mean()
    new["f_wind_max"] = d["wind_speed"].shift(-horizon).rolling(
        horizon, min_periods=horizon // 2).max()
    if use_dispersion:
        new["f_precip_sum"] = d["precipitation"].shift(-horizon).rolling(
            horizon, min_periods=horizon // 2).sum()
        new["f_blh_min"] = d["blh"].shift(-horizon).rolling(
            horizon, min_periods=horizon // 2).min()
        new["f_vi_mean"] = (d["blh"] * d["wind_speed"]).shift(-horizon).rolling(
            horizon, min_periods=horizon // 2).mean()
    return pd.concat([d, pd.DataFrame(new, index=d.index)], axis=1).copy()


RAW_EXCLUDE = {"timestamp", "aqi_instant", "aqi_epa"}


def cols_of(d, with_future):
    out = []
    for c in d.columns:
        if c in RAW_EXCLUDE or c.startswith("t_") or c.startswith("target_"):
            continue
        if c.startswith("f_") and not with_future:
            continue
        if not pd.api.types.is_numeric_dtype(d[c]):
            continue
        out.append(c)
    return out


def evaluate(d, fcols, tcol, label, delta=False):
    dd = d.dropna(subset=fcols + [tcol, "aqi"]).reset_index(drop=True)
    if len(dd) < 2000:
        print(f"{label:<56} SKIPPED ({len(dd)} rows)")
        return None
    r = expanding_backtest(dd, fcols, tcol, delta=delta)
    print(f"{label:<56} RMSE={r['rmse']:6.2f} MAE={r['mae']:6.2f} R2={r['r2']:6.3f} "
          f"folds={r['r2_folds']} naive={r['r2_naive']:6.3f} n={len(dd)} f={len(fcols)}")
    r.update({"label": label, "n": len(dd), "features": len(fcols)})
    return r


if __name__ == "__main__":
    raw = add_aqi_variants(load_long())
    raw["aqi"] = raw["aqi_epa"]
    results = []

    print("\n--- target autocorrelation on the LONG dataset ---")
    s = raw["aqi"].dropna()
    print("  " + "  ".join(f"+{l}h={s.autocorr(l):.3f}" for l in [24, 48, 72]))

    print("\n=== SET A: no dispersion vars (old weather only) ===")
    dA = build_features(raw, use_dispersion=False)
    print("=== SET B: + dispersion meteorology (BLH, precip, dew point, ...) ===")
    dB = build_features(raw, use_dispersion=True)

    for h in [24, 48, 72]:
        for name, base, disp in [("A no-disp", dA, False), ("B disp", dB, True)]:
            d = add_future_weather(base.copy(), h, use_dispersion=disp)
            d["t_mean"] = d["aqi"].shift(-h).rolling(24, min_periods=18).mean()

            results.append(evaluate(d, cols_of(d, False), "t_mean",
                                    f"{h}h {name} history-only"))
            results.append(evaluate(d, cols_of(d, True), "t_mean",
                                    f"{h}h {name} +future-weather"))
            results.append(evaluate(d, cols_of(d, True), "t_mean",
                                    f"{h}h {name} +future-weather delta", delta=True))

    out = pd.DataFrame([r for r in results if r])
    path = os.path.join(os.path.dirname(__file__), "..", "_diag_experiments3.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved -> {path}")
