"""
utils/aqi_daily.py
--------------------
Vectorised, EPA-CORRECT AQI computation over an hourly time series.

WHY THIS EXISTS
utils/aqi_calculator.py applies the EPA breakpoint formula to a SINGLE
instantaneous hourly concentration. That is what the project did until
now, and it is the root cause of the low R2 the supervisor flagged:

  * The EPA AQI is defined on AVERAGING PERIODS, not instantaneous
    readings - 24h mean for PM2.5/PM10, max 8h mean for O3/CO, max 1h
    for SO2/NO2. Applying it hour-by-hour produces a spiky quantity
    whose 24h-ahead autocorrelation is only 0.60 - i.e. mostly noise
    that NO model can predict (measured R2 ceiling: 0.24-0.37).
  * With the correct averaging the same quantity has autocorrelation
    0.77 at +24h and is genuinely forecastable.

So this module provides:
  hourly_aqi_epa()   - "AQI as of hour t" using trailing EPA windows.
                       Used as the current-conditions value and as the
                       basis for all lag/rolling features.
  daily_aqi_future() - the FORECAST TARGETS: the official daily AQI for
                       each of the next 3 days, computed from that day's
                       hours ONLY (non-overlapping with observed data, so
                       no leakage and no easy-mode overlap).

Both are vectorised (whole-column pandas), so 35k hourly rows compute in
milliseconds. Breakpoints are identical to utils/aqi_calculator.py.
"""

import numpy as np
import pandas as pd

# (conc_low, conc_high, aqi_low, aqi_high) - same tables as aqi_calculator.py
BREAKPOINTS = {
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

# Molecular weights for the ug/m3 -> ppm/ppb conversions the EPA tables need
MOL_WEIGHT = {"o3": 48, "co": 28, "so2": 64, "no2": 46}

# Minimum valid hours required inside a window before we trust its average
MIN_HOURS_24H = 18
MIN_HOURS_8H = 6


def sub_index(conc, pollutant):
    """Vectorised EPA piecewise-linear sub-index for one pollutant.

    Concentrations above the highest breakpoint are clamped to 500 rather
    than dropped - aqi_calculator.py returns None there, which silently
    removed the worst smog hours from the signal.
    """
    conc = np.asarray(conc, dtype=float)
    out = np.full(conc.shape, np.nan)
    for lo, hi, alo, ahi in BREAKPOINTS[pollutant]:
        mask = (conc >= lo) & (conc <= hi)
        out[mask] = (ahi - alo) / (hi - lo) * (conc[mask] - lo) + alo
    out[conc > BREAKPOINTS[pollutant][-1][1]] = 500.0
    return out


def to_ppm(conc_ugm3, pollutant):
    """ug/m3 -> ppm at 25C, 1 atm."""
    return np.asarray(conc_ugm3, dtype=float) * 24.45 / (MOL_WEIGHT[pollutant] * 1000)


def to_ppb(conc_ugm3, pollutant):
    """ug/m3 -> ppb at 25C, 1 atm."""
    return np.asarray(conc_ugm3, dtype=float) * 24.45 / MOL_WEIGHT[pollutant]


def hourly_aqi_epa(df, return_breakdown=False):
    """AQI 'as of' each hour, using TRAILING EPA averaging windows.

    df must contain the hourly columns pm2_5, pm10, o3, co, so2, no2
    (all in ug/m3), sorted ascending by time.

    Returns a Series of AQI values, and optionally the per-pollutant
    sub-index DataFrame so the dominant pollutant can be derived.
    """
    pm2_5_24h = df["pm2_5"].rolling(24, min_periods=MIN_HOURS_24H).mean()
    pm10_24h = df["pm10"].rolling(24, min_periods=MIN_HOURS_24H).mean()
    o3_8h = df["o3"].rolling(8, min_periods=MIN_HOURS_8H).mean()
    co_8h = df["co"].rolling(8, min_periods=MIN_HOURS_8H).mean()

    subs = pd.DataFrame({
        "pm2_5": sub_index(pm2_5_24h, "pm2_5"),
        "pm10": sub_index(pm10_24h, "pm10"),
        "o3": sub_index(to_ppm(o3_8h, "o3"), "o3"),
        "co": sub_index(to_ppm(co_8h, "co"), "co"),
        "so2": sub_index(to_ppb(df["so2"], "so2"), "so2"),
        "no2": sub_index(to_ppb(df["no2"], "no2"), "no2"),
    }, index=df.index)

    aqi = subs.max(axis=1)

    # PM2.5 is the dominant pollutant in Lahore ~87% of hours, and it is the
    # only sub-index needing a full 24h window. Without this guard, the first
    # rows of a series (and any row whose PM history is incomplete) would
    # report an AQI derived from SO2/NO2 alone - a plausible-looking but far
    # too low number. Better to return NaN and let the caller drop the row.
    aqi = aqi.where(subs["pm2_5"].notna())

    if return_breakdown:
        return aqi, subs
    return aqi



def daily_aqi_future(df, days=(1, 2, 3)):
    """Official daily AQI for each of the next `days` days.

    Day k covers the 24 hours ending at t + 24*k, and EPA averaging is
    applied INSIDE that day only:
        PM2.5 / PM10 -> mean of the day
        O3 / CO      -> max 8-hour mean within the day
        SO2 / NO2    -> max 1-hour value within the day
        daily AQI    -> max of the six sub-indices

    Each day's window is disjoint from the observed past, so these targets
    contain neither observed pollution nor future leakage.

    Returns a DataFrame with columns target_day1 / target_day2 / target_day3.
    """
    o3_8h = df["o3"].rolling(8, min_periods=MIN_HOURS_8H).mean()
    co_8h = df["co"].rolling(8, min_periods=MIN_HOURS_8H).mean()

    out = {}
    for k in days:
        end = 24 * k

        def forward(series, how):
            # min_periods=24: a DAILY value must cover a complete day. Any
            # tolerance here would let the final rows of the series produce a
            # target from a partial future window, which both weakens the label
            # and makes the last rows look predictable when they are not.
            window = series.shift(-end).rolling(24, min_periods=24)
            return getattr(window, how)()


        subs = pd.DataFrame({
            "pm2_5": sub_index(forward(df["pm2_5"], "mean"), "pm2_5"),
            "pm10": sub_index(forward(df["pm10"], "mean"), "pm10"),
            "o3": sub_index(to_ppm(forward(o3_8h, "max"), "o3"), "o3"),
            "co": sub_index(to_ppm(forward(co_8h, "max"), "co"), "co"),
            "so2": sub_index(to_ppb(forward(df["so2"], "max"), "so2"), "so2"),
            "no2": sub_index(to_ppb(forward(df["no2"], "max"), "no2"), "no2"),
        }, index=df.index)
        out[f"target_day{k}"] = subs.max(axis=1)

    return pd.DataFrame(out, index=df.index)


def dominant_pollutant(subs):
    """Given the sub-index DataFrame from hourly_aqi_epa(return_breakdown=True),
    returns the pollutant responsible for the overall AQI at each hour.

    Hours where no sub-index could be computed (start of the series, or the
    forecast-weather rows the webapp appends) return None rather than raising.
    """
    valid = subs.notna().any(axis=1)
    result = pd.Series(None, index=subs.index, dtype="object")
    if valid.any():
        result.loc[valid] = subs.loc[valid].idxmax(axis=1)
    return result
