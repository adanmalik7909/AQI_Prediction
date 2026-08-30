"""
training_pipeline/diagnose_experiments2.py
--------------------------------------------
Round 2 of the diagnosis. Round 1 showed:
  * EPA-correct AQI target (24h-avg PM2.5, 8h O3/CO) doubles achievable R2
    (24h: 0.24 -> 0.55) because the instantaneous target is mostly noise.
  * Future weather (from Open-Meteo forecast API) adds ~+0.09 R2.
  * Predicting the delta helps at 48/72h.

Round 2 asks the remaining question: what exactly should the target be?
A "3-day AQI forecast" on a dashboard means the AQI level for day+1/2/3,
not the instantaneous value at one specific hour. So we compare:

  T1  point-in-time AQI at t+h                (current design)
  T2  mean AQI over the window ending at t+h  (day-level average)
  T3  max  AQI over that window               (worst-case / alert use)

and validate with an EXPANDING-WINDOW (3-fold) backtest instead of a
single 80/20 split, so a mild test season can't flatter/punish the score.

Run:  python training_pipeline/diagnose_experiments2.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

sys.path.append(os.path.dirname(__file__))
from diagnose_accuracy import load_raw, add_aqi_variants
from diagnose_experiments import (base_features, extended_features,
                                  future_weather_features, feature_list)

HORIZONS = [24, 48, 72]
N_FOLDS = 3
VAL_FRAC = 0.15


def make_xgb(y_mean):
    return XGBRegressor(
        n_estimators=3000, max_depth=6, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.7, colsample_bylevel=0.7,
        reg_alpha=0.5, reg_lambda=3.0, min_child_weight=5, gamma=0.1,
        base_score=float(y_mean), random_state=42, n_jobs=-1,
        eval_metric="rmse", early_stopping_rounds=100,
    )


def expanding_backtest(dd, feature_cols, target_col, delta=False):
    """3 expanding-window folds: train on the first k blocks, test the next.
    Returns the mean metrics across folds + the per-fold R2 list."""
    n = len(dd)
    block = n // (N_FOLDS + 1)
    X = dd[feature_cols]
    lvl = dd[target_col]
    base = dd["aqi"]
    y = (lvl - base) if delta else lvl

    fold_r2, fold_rmse, fold_mae, fold_naive = [], [], [], []
    for k in range(1, N_FOLDS + 1):
        tr_end = block * k
        te_end = block * (k + 1) if k < N_FOLDS else n
        inner = int(tr_end * (1 - VAL_FRAC))

        m = make_xgb(y.iloc[:inner].mean())
        m.fit(X.iloc[:inner], y.iloc[:inner],
              eval_set=[(X.iloc[inner:tr_end], y.iloc[inner:tr_end])], verbose=False)

        pred = m.predict(X.iloc[tr_end:te_end])
        if delta:
            pred = pred + base.iloc[tr_end:te_end].values
        yt = lvl.iloc[tr_end:te_end].values

        fold_rmse.append(np.sqrt(mean_squared_error(yt, pred)))
        fold_mae.append(mean_absolute_error(yt, pred))
        fold_r2.append(r2_score(yt, pred))
        fold_naive.append(r2_score(yt, base.iloc[tr_end:te_end].values))

    return {
        "rmse": float(np.mean(fold_rmse)), "mae": float(np.mean(fold_mae)),
        "r2": float(np.mean(fold_r2)), "r2_naive": float(np.mean(fold_naive)),
        "r2_folds": [round(v, 3) for v in fold_r2],
    }


def evaluate(d, feature_cols, target_col, label, delta=False):
    dd = d.dropna(subset=feature_cols + [target_col, "aqi"]).reset_index(drop=True)
    if len(dd) < 1000:
        print(f"{label:<58} SKIPPED ({len(dd)} rows)")
        return None
    r = expanding_backtest(dd, feature_cols, target_col, delta=delta)
    print(f"{label:<58} RMSE={r['rmse']:6.2f} MAE={r['mae']:6.2f} "
          f"R2={r['r2']:6.3f} folds={r['r2_folds']} naive={r['r2_naive']:6.3f}")
    r.update({"label": label, "n": len(dd), "features": len(feature_cols)})
    return r


if __name__ == "__main__":
    raw = add_aqi_variants(load_raw())
    results = []

    d0 = raw.copy()
    d0["aqi"] = d0["aqi_epa"]          # round 1 winner: EPA-correct target
    d0 = base_features(d0)
    d1 = extended_features(d0)

    for h in HORIZONS:
        d = future_weather_features(d1.copy(), h)
        fcols = feature_list(d, extra_prefix="future")

        fut = d["aqi"].shift(-h)
        # T1: point-in-time value at t+h
        d["t_point"] = fut
        # T2: mean AQI over the h-hour window ENDING at t+h (day-level level)
        d["t_mean"] = d["aqi"].shift(-h).rolling(24, min_periods=18).mean()
        # T3: worst AQI within that window (alert / health-warning use case)
        d["t_max"] = d["aqi"].shift(-h).rolling(24, min_periods=18).max()

        for tcol, name in [("t_point", "T1 point-in-time"),
                           ("t_mean", "T2 24h-window mean"),
                           ("t_max", "T3 24h-window max")]:
            results.append(evaluate(d, fcols, tcol, f"{h}h  {name}  level"))
            results.append(evaluate(d, fcols, tcol, f"{h}h  {name}  delta", delta=True))

    out = pd.DataFrame([r for r in results if r])
    path = os.path.join(os.path.dirname(__file__), "..", "_diag_experiments2.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved -> {path}")
