"""
training_pipeline/diagnose_experiments.py
-------------------------------------------
Step B/C/D of the accuracy diagnosis. Uses the cached raw data from
diagnose_accuracy.py and runs ABLATION experiments with the SAME
chronological split + XGBoost setup as train_models.py, so every number
here is directly comparable to the current results.csv.

Experiments:
  1. CURRENT setup (instantaneous AQI target, current feature set)
  2. Same features, EPA-CORRECT AQI target (24h-avg PM, 8h O3/CO)
  3. EPA target + EXTENDED features (48/72h lags, longer rollings, EMA)
  4. #3 + FUTURE WEATHER features (available in production from the
     Open-Meteo forecast API - not leakage, it is a real forecast input)
  5. #4 but predicting the DELTA (target - current aqi) instead of level

Run:  python training_pipeline/diagnose_experiments.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

sys.path.append(os.path.dirname(__file__))
from diagnose_accuracy import load_raw, add_aqi_variants

HORIZONS = [24, 48, 72]
TEST_FRAC = 0.2
VAL_FRAC = 0.15


# ------------------------------------------------------------- features
def base_features(df):
    """Exactly the feature set fetch_training_data.py builds today."""
    d = df.copy()
    d["hour"] = d["timestamp"].dt.hour
    d["day"] = d["timestamp"].dt.day
    d["month"] = d["timestamp"].dt.month
    d["day_of_week"] = d["timestamp"].dt.dayofweek
    d["is_weekend"] = (d["day_of_week"] >= 5).astype(int)

    a = d["aqi"]
    for lag in [1, 3, 6, 12, 24]:
        d[f"aqi_lag_{lag}"] = a.shift(lag)
    d["aqi_change_rate"] = a - d["aqi_lag_1"]

    a1 = a.shift(1)
    d["aqi_rolling_mean_24h"] = a1.rolling(24).mean()
    d["aqi_rolling_std_24h"] = a1.rolling(24).std()
    d["aqi_rolling_min_24h"] = a1.rolling(24).min()
    d["aqi_rolling_max_24h"] = a1.rolling(24).max()
    d["aqi_deviation_24h"] = a - d["aqi_rolling_mean_24h"]

    d["pm2_5_lag_1"] = d["pm2_5"].shift(1)
    d["pm2_5_lag_24"] = d["pm2_5"].shift(24)
    d["pm2_5_rolling_mean_24h"] = d["pm2_5"].shift(1).rolling(24).mean()
    d["pm10_rolling_mean_24h"] = d["pm10"].shift(1).rolling(24).mean()
    d["pm_ratio"] = d["pm2_5"] / (d["pm10"] + 0.01)

    d["wind_speed_rolling_mean_6h"] = d["wind_speed"].shift(1).rolling(6).mean()
    d["temp_change_rate"] = d["temperature"] - d["temperature"].shift(1)
    d["wind_humidity_interaction"] = d["wind_speed"] * d["humidity"]

    d["hour_sin"] = np.sin(2 * np.pi * d["hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["hour"] / 24)
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
    d["dow_sin"] = np.sin(2 * np.pi * d["day_of_week"] / 7)
    d["dow_cos"] = np.cos(2 * np.pi * d["day_of_week"] / 7)
    return d


def extended_features(d):
    """Longer memory: 48/72h lags, medium-term rollings, EMA, trends."""
    a = d["aqi"]
    for lag in [36, 48, 60, 72, 96, 120, 168]:
        d[f"aqi_lag_{lag}"] = a.shift(lag)

    a1 = a.shift(1)
    for w in [3, 6, 12, 48, 72, 168]:
        d[f"aqi_rolling_mean_{w}h"] = a1.rolling(w).mean()
    d["aqi_rolling_std_48h"] = a1.rolling(48).std()
    d["aqi_rolling_std_72h"] = a1.rolling(72).std()
    d["aqi_rolling_min_72h"] = a1.rolling(72).min()
    d["aqi_rolling_max_72h"] = a1.rolling(72).max()
    d["aqi_ema_12h"] = a1.ewm(span=12).mean()
    d["aqi_ema_48h"] = a1.ewm(span=48).mean()
    d["aqi_trend_24h"] = d["aqi_rolling_mean_24h"] - d["aqi_rolling_mean_48h"]
    d["aqi_trend_72h"] = d["aqi_rolling_mean_24h"] - d["aqi_rolling_mean_72h"]

    for col in ["pm2_5", "pm10", "o3", "co", "no2", "so2"]:
        s = d[col]
        d[f"{col}_lag_6"] = s.shift(6)
        d[f"{col}_lag_12"] = s.shift(12)
        d[f"{col}_lag_48"] = s.shift(48)
        d[f"{col}_rolling_mean_24h"] = s.shift(1).rolling(24).mean()
        d[f"{col}_rolling_mean_72h"] = s.shift(1).rolling(72).mean()
        d[f"{col}_rolling_std_24h"] = s.shift(1).rolling(24).std()

    for col in ["temperature", "humidity", "pressure", "wind_speed", "cloud_cover"]:
        s = d[col]
        d[f"{col}_rolling_mean_24h"] = s.shift(1).rolling(24).mean()
        d[f"{col}_change_24h"] = s - s.shift(24)

    d["pollution_intensity"] = (d["pm2_5"] + d["pm10"]) / 2
    d["temp_humidity_interaction"] = d["temperature"] * d["humidity"]
    d["ventilation_index"] = d["wind_speed"] * (100 - d["humidity"])
    d["day_of_year"] = d["timestamp"].dt.dayofyear
    d["doy_sin"] = np.sin(2 * np.pi * d["day_of_year"] / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * d["day_of_year"] / 365.25)
    return d


def future_weather_features(d, horizon):
    """Weather at t+horizon. In production these come from the Open-Meteo
    FORECAST API (free, 16-day horizon), so this is a legitimate input,
    not leakage - we just use the archived actuals as a stand-in during
    training/backtesting."""
    for col in ["temperature", "humidity", "pressure", "wind_speed", "cloud_cover"]:
        d[f"f_{col}_t{horizon}"] = d[col].shift(-horizon)
        # mean weather over the whole window between now and the target
        d[f"f_{col}_mean_{horizon}h"] = (
            d[col].shift(-horizon).rolling(horizon, min_periods=horizon // 2).mean()
        )
    d[f"f_wind_max_{horizon}h"] = (
        d["wind_speed"].shift(-horizon).rolling(horizon, min_periods=horizon // 2).max()
    )
    return d


# ------------------------------------------------------------ evaluation
def make_xgb(y_mean):
    return XGBRegressor(
        n_estimators=3000, max_depth=6, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.7, colsample_bylevel=0.7,
        reg_alpha=0.5, reg_lambda=3.0, min_child_weight=5, gamma=0.1,
        base_score=float(y_mean), random_state=42, n_jobs=-1,
        eval_metric="rmse", early_stopping_rounds=100,
    )


def run(d, feature_cols, target_col, delta=False, label=""):
    """Chronological 80/20 split (with an inner val split), XGBoost, report."""
    cols = feature_cols + [target_col, "aqi"]
    dd = d.dropna(subset=cols).reset_index(drop=True)
    if len(dd) < 500:
        print(f"{label:<52} SKIPPED (only {len(dd)} usable rows)")
        return None

    y_true_level = dd[target_col]
    y = (y_true_level - dd["aqi"]) if delta else y_true_level

    n = len(dd)
    split = int(n * (1 - TEST_FRAC))
    inner = int(split * (1 - VAL_FRAC))

    X = dd[feature_cols]
    X_tr, y_tr = X.iloc[:inner], y.iloc[:inner]
    X_va, y_va = X.iloc[inner:split], y.iloc[inner:split]
    X_te = X.iloc[split:]

    m = make_xgb(y_tr.mean())
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    pred = m.predict(X_te)
    if delta:
        pred = pred + dd["aqi"].iloc[split:].values

    yt = y_true_level.iloc[split:].values
    rmse = float(np.sqrt(mean_squared_error(yt, pred)))
    mae = float(mean_absolute_error(yt, pred))
    r2 = float(r2_score(yt, pred))
    naive = dd["aqi"].iloc[split:].values
    r2_naive = float(r2_score(yt, naive))

    print(f"{label:<52} RMSE={rmse:6.2f} MAE={mae:6.2f} R2={r2:6.3f} "
          f"(naive R2={r2_naive:6.3f}, n={len(dd)}, feats={len(feature_cols)})")
    return {"label": label, "rmse": rmse, "mae": mae, "r2": r2,
            "r2_naive": r2_naive, "n_features": len(feature_cols)}


EXCLUDE = {"timestamp", "aqi_instant", "aqi_epa", "day_of_year"}


def feature_list(d, extra_prefix=None):
    cols = []
    for c in d.columns:
        if c in EXCLUDE or c.startswith("target_"):
            continue
        if c.startswith("f_") and extra_prefix != "future":
            continue
        if not pd.api.types.is_numeric_dtype(d[c]):
            continue
        cols.append(c)
    return cols


if __name__ == "__main__":
    raw = add_aqi_variants(load_raw())
    results = []

    for target_def in ["aqi_instant", "aqi_epa"]:
        print(f"\n{'='*100}\nTARGET DEFINITION: {target_def}\n{'='*100}")

        d0 = raw.copy()
        d0["aqi"] = d0[target_def]
        d0 = base_features(d0)
        base_cols = feature_list(d0)

        d1 = extended_features(d0.copy())
        ext_cols = feature_list(d1)

        for h in HORIZONS:
            tcol = f"target_{h}h"

            db = d0.copy()
            db[tcol] = db["aqi"].shift(-h)
            results.append(run(db, base_cols, tcol,
                               label=f"[{target_def}] {h}h  1.current-features"))

            de = d1.copy()
            de[tcol] = de["aqi"].shift(-h)
            results.append(run(de, ext_cols, tcol,
                               label=f"[{target_def}] {h}h  2.extended-features"))

            df_ = future_weather_features(d1.copy(), h)
            df_[tcol] = df_["aqi"].shift(-h)
            fut_cols = feature_list(df_, extra_prefix="future")
            results.append(run(df_, fut_cols, tcol,
                               label=f"[{target_def}] {h}h  3.extended+future-weather"))

            results.append(run(df_, fut_cols, tcol, delta=True,
                               label=f"[{target_def}] {h}h  4.#3 predicting delta"))

    out = pd.DataFrame([r for r in results if r])
    path = os.path.join(os.path.dirname(__file__), "..", "_diag_experiments.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved -> {path}")
