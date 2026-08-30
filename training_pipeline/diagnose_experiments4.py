"""
training_pipeline/diagnose_experiments4.py
--------------------------------------------
Round 4 - the LEAKAGE CHECK, and the final target definition.

Round 3 reached R2 0.87 at 24h using target = mean of aqi_epa over
hours t+1..t+24. That number is partly inflated: aqi_epa(t+1) is itself a
24h TRAILING average, so it re-uses pollution from t-22..t+1, i.e. data
we already observed. It is not leakage of the future, but it makes the
task easier than a genuine "day ahead" forecast.

The clean, defensible definition (and the one air-quality agencies
actually publish) is the DAILY AQI computed on a non-overlapping future
day, using EPA averaging rules applied INSIDE that day only:

  day1 = hours t+1  .. t+24
  day2 = hours t+25 .. t+48
  day3 = hours t+49 .. t+72

  PM2.5 / PM10 -> mean of the day
  O3 / CO      -> max 8-hour mean within the day
  SO2 / NO2    -> max 1-hour value within the day
  daily AQI    -> max of those sub-indices

Zero overlap with observed data, so whatever R2 we get here is real.

Run:  python training_pipeline/diagnose_experiments4.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from diagnose_accuracy import add_aqi_variants, sub_aqi, _ppm, _ppb
from diagnose_experiments2 import expanding_backtest
from diagnose_experiments3 import (load_long, build_features, add_future_weather,
                                   cols_of)


def daily_aqi_targets(df, days=(1, 2, 3)):
    """Non-overlapping future daily AQI, EPA averaging applied within the day."""
    out = {}
    # 8-hour trailing means for the O3/CO sub-indices
    o3_8h = df["o3"].rolling(8, min_periods=6).mean()
    co_8h = df["co"].rolling(8, min_periods=6).mean()

    for k in days:
        end = 24 * k          # last hour of day k, relative to t

        def fwd(series, how="mean"):
            s = series.shift(-end)
            r = s.rolling(24, min_periods=20)
            return getattr(r, how)()

        subs = pd.DataFrame({
            "pm2_5": sub_aqi(fwd(df["pm2_5"], "mean"), "pm2_5"),
            "pm10": sub_aqi(fwd(df["pm10"], "mean"), "pm10"),
            "o3": sub_aqi(_ppm(fwd(o3_8h, "max"), 48), "o3"),
            "co": sub_aqi(_ppm(fwd(co_8h, "max"), 28), "co"),
            "so2": sub_aqi(_ppb(fwd(df["so2"], "max"), 64), "so2"),
            "no2": sub_aqi(_ppb(fwd(df["no2"], "max"), 46), "no2"),
        })
        out[f"t_day{k}"] = subs.max(axis=1)
    return pd.DataFrame(out, index=df.index)


def evaluate(d, fcols, tcol, label, delta=False):
    dd = d.dropna(subset=fcols + [tcol, "aqi"]).reset_index(drop=True)
    if len(dd) < 2000:
        print(f"{label:<52} SKIPPED ({len(dd)} rows)")
        return None
    r = expanding_backtest(dd, fcols, tcol, delta=delta)
    print(f"{label:<52} RMSE={r['rmse']:6.2f} MAE={r['mae']:6.2f} R2={r['r2']:6.3f} "
          f"folds={r['r2_folds']} naive={r['r2_naive']:6.3f} n={len(dd)} f={len(fcols)}")
    r.update({"label": label, "n": len(dd), "features": len(fcols)})
    return r


if __name__ == "__main__":
    raw = add_aqi_variants(load_long())
    raw["aqi"] = raw["aqi_epa"]

    tg = daily_aqi_targets(raw)
    print("\n--- daily target stats ---")
    for c in tg.columns:
        s = tg[c].dropna()
        print(f"{c}: n={len(s)} mean={s.mean():.1f} std={s.std():.1f} "
              f"min={s.min():.0f} max={s.max():.0f}")
    print("\ncorr(day1, current aqi) = "
          f"{tg['t_day1'].corr(raw['aqi']):.3f}")
    print(f"corr(day2, current aqi) = {tg['t_day2'].corr(raw['aqi']):.3f}")
    print(f"corr(day3, current aqi) = {tg['t_day3'].corr(raw['aqi']):.3f}")

    feats = build_features(raw, use_dispersion=True)
    results = []

    for k, h in [(1, 24), (2, 48), (3, 72)]:
        d = add_future_weather(feats.copy(), h, use_dispersion=True)
        tcol = f"t_day{k}"
        d[tcol] = tg[tcol]

        results.append(evaluate(d, cols_of(d, False), tcol,
                                f"day{k} history-only"))
        results.append(evaluate(d, cols_of(d, True), tcol,
                                f"day{k} +future-weather"))
        results.append(evaluate(d, cols_of(d, True), tcol,
                                f"day{k} +future-weather delta", delta=True))

    out = pd.DataFrame([r for r in results if r])
    path = os.path.join(os.path.dirname(__file__), "..", "_diag_experiments4.csv")
    out.to_csv(path, index=False)
    print(f"\nSaved -> {path}")
