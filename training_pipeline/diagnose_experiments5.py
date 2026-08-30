"""
training_pipeline/diagnose_experiments5.py
--------------------------------------------
Round 5 - final tuning of the HONEST target before we change the pipeline.

Round 4 settled the target definition: non-overlapping future daily AQI
(day1 = t+1..t+24, day2 = t+25..t+48, day3 = t+49..t+72), EPA averaging
applied INSIDE each day. No overlap with observed data, so the score is
real. Baseline from round 4 (XGBoost, per-horizon future weather):

    day1 R2 0.551 | day2 R2 0.382 | day3 R2 0.312

Round 4 gave the model future weather aggregated over t+1..t+h, which for
day3 blurs all three days together. Round 5 fixes that and tests the
remaining levers:

  L1  PER-DAY future weather windows (day1/day2/day3 separately, all
      given to every horizon - a 3-day forecast is available anyway)
  L2  LightGBM vs XGBoost vs their average
  L3  log1p target transform (AQI is right-skewed)

Run:  python training_pipeline/diagnose_experiments5.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

sys.path.append(os.path.dirname(__file__))
from diagnose_accuracy import add_aqi_variants
from diagnose_experiments3 import load_long, build_features, WEATHER
from diagnose_experiments4 import daily_aqi_targets

N_FOLDS = 3
VAL_FRAC = 0.15


def add_per_day_future_weather(d):
    """Future weather summarised SEPARATELY for each forecast day.

    In production these come from the Open-Meteo weather forecast API
    (free, keyless, 16-day horizon) - a genuinely available input, not
    leakage. Here the archived actuals stand in for the forecast.
    """
    new = {}
    for k in (1, 2, 3):
        end = 24 * k
        for col in WEATHER:
            if col not in d.columns:
                continue
            s = d[col].shift(-end).rolling(24, min_periods=20)
            new[f"f{k}_{col}_mean"] = s.mean()
        new[f"f{k}_wind_max"] = d["wind_speed"].shift(-end).rolling(24, min_periods=20).max()
        new[f"f{k}_wind_min"] = d["wind_speed"].shift(-end).rolling(24, min_periods=20).min()
        new[f"f{k}_precip_sum"] = d["precipitation"].shift(-end).rolling(24, min_periods=20).sum()
        new[f"f{k}_blh_min"] = d["blh"].shift(-end).rolling(24, min_periods=20).min()
        new[f"f{k}_blh_max"] = d["blh"].shift(-end).rolling(24, min_periods=20).max()
        vi = d["blh"] * d["wind_speed"]
        new[f"f{k}_vi_mean"] = vi.shift(-end).rolling(24, min_periods=20).mean()
        new[f"f{k}_vi_min"] = vi.shift(-end).rolling(24, min_periods=20).min()
        # change relative to the last observed 24h - "is it getting windier?"
        new[f"f{k}_wind_delta"] = (d["wind_speed"].shift(-end).rolling(24, min_periods=20).mean()
                                   - d["wind_speed"].shift(1).rolling(24).mean())
        new[f"f{k}_blh_delta"] = (d["blh"].shift(-end).rolling(24, min_periods=20).mean()
                                  - d["blh"].shift(1).rolling(24).mean())
    return pd.concat([d, pd.DataFrame(new, index=d.index)], axis=1).copy()


EXCLUDE = {"timestamp", "aqi_instant", "aqi_epa"}


def feature_cols(d):
    return [c for c in d.columns
            if c not in EXCLUDE and not c.startswith("t_day")
            and pd.api.types.is_numeric_dtype(d[c])]


def make_xgb(y_mean):
    return XGBRegressor(
        n_estimators=3000, max_depth=6, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.7, colsample_bylevel=0.7,
        reg_alpha=0.5, reg_lambda=3.0, min_child_weight=5, gamma=0.1,
        base_score=float(y_mean), random_state=42, n_jobs=-1,
        eval_metric="rmse", early_stopping_rounds=100,
    )


def make_lgb():
    return LGBMRegressor(
        n_estimators=4000, learning_rate=0.02, num_leaves=63,
        min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=3.0,
        random_state=42, n_jobs=-1, verbose=-1,
    )


def backtest(dd, fcols, tcol, model="xgb", log_target=False):
    """Expanding-window backtest; returns mean metrics + per-fold R2."""
    import lightgbm as lgb_mod

    n = len(dd)
    block = n // (N_FOLDS + 1)
    X = dd[fcols]
    lvl = dd[tcol]
    y = np.log1p(lvl) if log_target else lvl

    r2s, rmses, maes, naives = [], [], [], []
    for k in range(1, N_FOLDS + 1):
        tr_end = block * k
        te_end = block * (k + 1) if k < N_FOLDS else n
        inner = int(tr_end * (1 - VAL_FRAC))

        Xtr, ytr = X.iloc[:inner], y.iloc[:inner]
        Xva, yva = X.iloc[inner:tr_end], y.iloc[inner:tr_end]
        Xte = X.iloc[tr_end:te_end]

        preds = []
        if model in ("xgb", "blend"):
            m = make_xgb(ytr.mean())
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            preds.append(m.predict(Xte))
        if model in ("lgb", "blend"):
            m = make_lgb()
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  callbacks=[lgb_mod.early_stopping(150, verbose=False)])
            preds.append(m.predict(Xte))

        pred = np.mean(preds, axis=0)
        if log_target:
            pred = np.expm1(pred)

        yt = lvl.iloc[tr_end:te_end].values
        rmses.append(np.sqrt(mean_squared_error(yt, pred)))
        maes.append(mean_absolute_error(yt, pred))
        r2s.append(r2_score(yt, pred))
        naives.append(r2_score(yt, dd["aqi"].iloc[tr_end:te_end].values))

    return {"rmse": float(np.mean(rmses)), "mae": float(np.mean(maes)),
            "r2": float(np.mean(r2s)), "r2_naive": float(np.mean(naives)),
            "r2_folds": [round(v, 3) for v in r2s]}


def evaluate(d, fcols, tcol, label, **kw):
    dd = d.dropna(subset=fcols + [tcol, "aqi"]).reset_index(drop=True)
    if len(dd) < 2000:
        print(f"{label:<48} SKIPPED ({len(dd)} rows)")
        return None
    r = backtest(dd, fcols, tcol, **kw)
    print(f"{label:<48} RMSE={r['rmse']:6.2f} MAE={r['mae']:6.2f} R2={r['r2']:6.3f} "
          f"folds={r['r2_folds']} n={len(dd)} f={len(fcols)}")
    r.update({"label": label, "n": len(dd), "features": len(fcols)})
    return r


if __name__ == "__main__":
    raw = add_aqi_variants(load_long())
    raw["aqi"] = raw["aqi_epa"]
    tg = daily_aqi_targets(raw)

    d = build_features(raw, use_dispersion=True)
    d = add_per_day_future_weather(d)
    for c in tg.columns:
        d[c] = tg[c]
    fcols = feature_cols(d)
    print(f"\nFeatures: {len(fcols)}   rows: {len(d)}")

    results = []
    for k in (1, 2, 3):
        tcol = f"t_day{k}"
        results.append(evaluate(d, fcols, tcol, f"day{k} xgb", model="xgb"))
        results.append(evaluate(d, fcols, tcol, f"day{k} lgb", model="lgb"))
        results.append(evaluate(d, fcols, tcol, f"day{k} blend", model="blend"))
        results.append(evaluate(d, fcols, tcol, f"day{k} blend log1p",
                                model="blend", log_target=True))

    out = pd.DataFrame([r for r in results if r])
    path = os.path.join(os.path.dirname(__file__), "..", "_diag_experiments5.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved -> {path}")
