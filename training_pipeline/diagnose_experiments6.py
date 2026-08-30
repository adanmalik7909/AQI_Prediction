"""
training_pipeline/diagnose_experiments6.py
-------------------------------------------
Round 6 - choosing between the two HONEST target definitions, and testing
a two-stage (per-pollutant) model.

Rounds 1-5 established:
  * instantaneous-AQI target  -> R2 ceiling 0.24-0.37   (noise)
  * "daily AQI of future day k", EPA rules inside the day -> 0.54/0.39/0.35
  * per-day future weather beats one aggregated window at day3 (0.31->0.35)
  * LightGBM ~= XGBoost; blending and log1p change nothing material

Remaining question. There are two leakage-free ways to say "AQI in 24h":

  D1  daily AQI of future day k          (round 4/5 definition)
        PM 24h mean, O3/CO max-8h, SO2/NO2 max-1h, inside day k only
  D2  the AQI value as REPORTED at hour t+24h
        = hourly_aqi_epa shifted by -h. Its PM2.5 window is exactly
          hours t+1..t+h, so it is also fully non-overlapping.

D2 is literally what the dashboard promises ("AQI 24 hours from now") and
is a smoother quantity, so it should score higher without cheating.

Also tested: TWO-STAGE. Instead of regressing AQI directly, predict each
pollutant's future window statistic (PM2.5/PM10 mean, O3/CO max-8h) and
push those through the EPA breakpoints. Because AQI is a max() of
sub-indices, a single regressor has to learn a kinked function; per
pollutant the relationships are smooth.

Run:  python training_pipeline/diagnose_experiments6.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from aqi_daily import (hourly_aqi_epa, daily_aqi_future, sub_index,
                       to_ppm, to_ppb, MIN_HOURS_8H)
from diagnose_experiments3 import load_long, build_features
from diagnose_experiments5 import add_per_day_future_weather

N_FOLDS = 3
VAL_FRAC = 0.15
HORIZONS = [24, 48, 72]


def make_xgb(y_mean):
    return XGBRegressor(
        n_estimators=3000, max_depth=6, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.7, colsample_bylevel=0.7,
        reg_alpha=0.5, reg_lambda=3.0, min_child_weight=5, gamma=0.1,
        base_score=float(y_mean), random_state=42, n_jobs=-1,
        eval_metric="rmse", early_stopping_rounds=100,
    )


def folds(n):
    block = n // (N_FOLDS + 1)
    for k in range(1, N_FOLDS + 1):
        tr_end = block * k
        te_end = block * (k + 1) if k < N_FOLDS else n
        yield int(tr_end * (1 - VAL_FRAC)), tr_end, te_end


def fit_predict(X, y, inner, tr_end, te_end):
    m = make_xgb(y.iloc[:inner].mean())
    m.fit(X.iloc[:inner], y.iloc[:inner],
          eval_set=[(X.iloc[inner:tr_end], y.iloc[inner:tr_end])], verbose=False)
    return m.predict(X.iloc[tr_end:te_end])


def report(label, yt_all, pred_all, r2s, naive):
    rmse = float(np.sqrt(mean_squared_error(yt_all, pred_all)))
    mae = float(mean_absolute_error(yt_all, pred_all))
    r2 = float(np.mean(r2s))
    print(f"{label:<46} RMSE={rmse:6.2f} MAE={mae:6.2f} R2={r2:6.3f} "
          f"folds={[round(v,3) for v in r2s]} naive={naive:6.3f}")
    return {"label": label, "rmse": rmse, "mae": mae, "r2": r2,
            "r2_naive": naive, "r2_folds": [round(v, 3) for v in r2s]}


def direct(dd, fcols, tcol, label):
    """One regressor straight onto the AQI target."""
    X, lvl = dd[fcols], dd[tcol]
    r2s, yts, prs, nvs = [], [], [], []
    for inner, tr_end, te_end in folds(len(dd)):
        pred = fit_predict(X, lvl, inner, tr_end, te_end)
        yt = lvl.iloc[tr_end:te_end].values
        r2s.append(r2_score(yt, pred))
        nvs.append(r2_score(yt, dd["aqi"].iloc[tr_end:te_end].values))
        yts.append(yt); prs.append(pred)
    return report(label, np.concatenate(yts), np.concatenate(prs),
                  r2s, float(np.mean(nvs)))


def two_stage(dd, fcols, sub_targets, tcol, label):
    """Predict each pollutant's future window stat, then apply EPA breakpoints."""
    X, lvl = dd[fcols], dd[tcol]
    r2s, yts, prs, nvs = [], [], [], []
    for inner, tr_end, te_end in folds(len(dd)):
        parts = {}
        for name, (col, pollutant, unit) in sub_targets.items():
            raw_pred = fit_predict(X, dd[col], inner, tr_end, te_end)
            raw_pred = np.clip(raw_pred, 0, None)
            if unit == "ppm":
                conc = to_ppm(raw_pred, pollutant)
            elif unit == "ppb":
                conc = to_ppb(raw_pred, pollutant)
            else:
                conc = raw_pred
            parts[name] = sub_index(conc, pollutant)
        pred = np.nanmax(np.vstack(list(parts.values())), axis=0)

        yt = lvl.iloc[tr_end:te_end].values
        ok = ~np.isnan(pred)
        r2s.append(r2_score(yt[ok], pred[ok]))
        nvs.append(r2_score(yt, dd["aqi"].iloc[tr_end:te_end].values))
        yts.append(yt[ok]); prs.append(pred[ok])
    return report(label, np.concatenate(yts), np.concatenate(prs),
                  r2s, float(np.mean(nvs)))


EXCLUDE_PREFIX = ("target_", "t_", "sub_")


def feature_cols(d):
    skip = {"timestamp", "aqi_instant", "aqi_epa"}
    return [c for c in d.columns
            if c not in skip and not c.startswith(EXCLUDE_PREFIX)
            and pd.api.types.is_numeric_dtype(d[c])]


if __name__ == "__main__":
    raw = load_long()
    raw["aqi"] = hourly_aqi_epa(raw)

    d = build_features(raw, use_dispersion=True)
    d = add_per_day_future_weather(d)

    # --- D1: daily AQI of future day k (rounds 4/5) ---
    for c, s in daily_aqi_future(raw).items():
        d[c] = s

    # --- D2: the reported AQI value at hour t+h ---
    for h in HORIZONS:
        d[f"target_at_{h}h"] = raw["aqi"].shift(-h)

    # --- per-pollutant window stats, for the two-stage model (D2 style) ---
    o3_8h = raw["o3"].rolling(8, min_periods=MIN_HOURS_8H).mean()
    co_8h = raw["co"].rolling(8, min_periods=MIN_HOURS_8H).mean()
    for h in HORIZONS:
        d[f"sub_pm2_5_{h}"] = raw["pm2_5"].rolling(24, min_periods=18).mean().shift(-h)
        d[f"sub_pm10_{h}"] = raw["pm10"].rolling(24, min_periods=18).mean().shift(-h)
        d[f"sub_o3_{h}"] = o3_8h.shift(-h)
        d[f"sub_co_{h}"] = co_8h.shift(-h)
        d[f"sub_so2_{h}"] = raw["so2"].shift(-h)
        d[f"sub_no2_{h}"] = raw["no2"].shift(-h)

    fcols = feature_cols(d)
    print(f"\nFeatures: {len(fcols)}")

    results = []
    for k, h in [(1, 24), (2, 48), (3, 72)]:
        d1col, d2col = f"target_day{k}", f"target_at_{h}h"
        subs = {
            "pm2_5": (f"sub_pm2_5_{h}", "pm2_5", "ugm3"),
            "pm10": (f"sub_pm10_{h}", "pm10", "ugm3"),
            "o3": (f"sub_o3_{h}", "o3", "ppm"),
            "co": (f"sub_co_{h}", "co", "ppm"),
            "so2": (f"sub_so2_{h}", "so2", "ppb"),
            "no2": (f"sub_no2_{h}", "no2", "ppb"),
        }
        need = fcols + [d1col, d2col, "aqi"] + [v[0] for v in subs.values()]
        dd = d.dropna(subset=need).reset_index(drop=True)
        print(f"\n--- horizon {h}h (day{k}), usable rows: {len(dd)} ---")

        results.append(direct(dd, fcols, d1col, f"{h}h D1 daily-AQI  direct"))
        results.append(direct(dd, fcols, d2col, f"{h}h D2 AQI-at-t+h direct"))
        results.append(two_stage(dd, fcols, subs, d2col,
                                 f"{h}h D2 AQI-at-t+h two-stage"))

    out = pd.DataFrame(results)
    path = os.path.join(os.path.dirname(__file__), "..", "_diag_experiments6.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved -> {path}")
