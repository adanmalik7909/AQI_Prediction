"""
utils/feature_engineering.py
-----------------------------
THE single source of truth for feature engineering.

Before this module existed, the lag/rolling features were written out twice
- once in training_pipeline/fetch_training_data.py and again by hand in
webapp/app.py. Any change had to be mirrored in both places or the served
features would silently drift from the trained ones. Now both import from
here, so train-serve consistency is structural rather than a promise.

Three groups of features are built:

  1. HISTORY features - lags, rolling stats, EMAs, trends of AQI and of
     each pollutant. Everything uses .shift(1) before .rolling() so a row
     never sees its own hour inside a "past" average.

  2. WEATHER + DISPERSION features - including the physics that actually
     controls smog: boundary layer height (the lid on the atmosphere),
     ventilation index (BLH x wind = how much air is available to dilute
     into), stagnation, an inversion proxy, and wet deposition (rain).

  3. FUTURE WEATHER features - per forecast day (day1/day2/day3), the
     weather the pollution is heading into. During training these come
     from shifting the archived weather columns; in production they come
     from the Open-Meteo weather forecast API. Same columns, same names.
     Measured contribution: +0.08 to +0.11 R2 at the 48/72h horizons.

NOTE: pollutant forecasts are deliberately NOT used as features - see the
explanation in utils/openmeteo_client.py.
"""

import numpy as np
import pandas as pd

# Pollutant columns we build history features for. dust / aod come from
# Open-Meteo's air quality API and are useful for Lahore specifically
# (dust storms and crop-burning haze).
POLLUTANT_COLS = ["pm2_5", "pm10", "o3", "co", "no2", "so2", "dust", "aod"]

# Weather columns. The first five are the project's original set; the rest
# are the dispersion-meteorology variables added during the accuracy work.
WEATHER_COLS = [
    "temperature", "humidity", "pressure", "wind_speed", "cloud_cover",
    "blh", "precipitation", "dew_point", "radiation", "wind_speed_100m",
]

AQI_LAGS = [1, 3, 6, 12, 24, 36, 48, 72, 96, 120, 168]
AQI_ROLL_WINDOWS = [3, 6, 12, 24, 48, 72, 168]
AQI_SPREAD_WINDOWS = [24, 48, 72]
POLLUTANT_LAGS = [1, 6, 12, 24, 48]

FORECAST_DAYS = [1, 2, 3]
# A forecast day needs at least this many of its 24 hours present before we
# trust its summary statistic.
MIN_FORECAST_HOURS = 20


def _cyclical(values, period):
    """sin/cos encoding so hour 23 sits next to hour 0, not 23 units away."""
    radians = 2 * np.pi * values / period
    return np.sin(radians), np.cos(radians)


def add_calendar_features(df, out):
    """Time-of-day / seasonality. Lahore's AQI swings from ~207 in January
    to ~126 in August, so day-of-year matters as much as hour-of-day."""
    ts = df["timestamp"]
    out["hour"] = ts.dt.hour
    out["day"] = ts.dt.day
    out["month"] = ts.dt.month
    out["day_of_week"] = ts.dt.dayofweek
    out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    out["hour_sin"], out["hour_cos"] = _cyclical(ts.dt.hour, 24)
    out["month_sin"], out["month_cos"] = _cyclical(ts.dt.month, 12)
    out["dow_sin"], out["dow_cos"] = _cyclical(ts.dt.dayofweek, 7)
    out["doy_sin"], out["doy_cos"] = _cyclical(ts.dt.dayofyear, 365.25)
    return out


def add_aqi_history(df, out):
    """The model's memory of where AQI has been.

    Lags reach back a full week (168h): at the 48/72h horizons the model
    needs to know what happened 2-3 days ago, and the same-hour-last-week
    value carries the weekly traffic cycle.
    """
    aqi = df["aqi"]
    past = aqi.shift(1)  # shift first => rolling windows exclude the current hour

    for lag in AQI_LAGS:
        out[f"aqi_lag_{lag}"] = aqi.shift(lag)

    out["aqi_change_rate"] = aqi - aqi.shift(1)
    out["aqi_change_24h"] = aqi - aqi.shift(24)

    for window in AQI_ROLL_WINDOWS:
        out[f"aqi_rolling_mean_{window}h"] = past.rolling(window).mean()

    for window in AQI_SPREAD_WINDOWS:
        rolling = past.rolling(window)
        out[f"aqi_rolling_std_{window}h"] = rolling.std()
        out[f"aqi_rolling_min_{window}h"] = rolling.min()
        out[f"aqi_rolling_max_{window}h"] = rolling.max()

    # Exponential means weight recent hours more heavily than a flat average
    out["aqi_ema_12h"] = past.ewm(span=12).mean()
    out["aqi_ema_48h"] = past.ewm(span=48).mean()

    # Is the current hour unusual, and is the trend rising or falling?
    mean_24h = past.rolling(24).mean()
    out["aqi_deviation_24h"] = aqi - mean_24h
    out["aqi_trend_24h"] = mean_24h - past.rolling(48).mean()
    out["aqi_trend_72h"] = mean_24h - past.rolling(72).mean()
    return out


def add_pollutant_history(df, out):
    """Same treatment for the raw pollutants that feed the AQI formula."""
    for col in POLLUTANT_COLS:
        if col not in df.columns:
            continue
        series = df[col]
        past = series.shift(1)
        for lag in POLLUTANT_LAGS:
            out[f"{col}_lag_{lag}"] = series.shift(lag)
        out[f"{col}_rolling_mean_24h"] = past.rolling(24).mean()
        out[f"{col}_rolling_mean_72h"] = past.rolling(72).mean()
        out[f"{col}_rolling_std_24h"] = past.rolling(24).std()

    # PM2.5/PM10 ratio separates fine combustion haze from coarse dust
    out["pm_ratio"] = df["pm2_5"] / (df["pm10"] + 0.01)
    out["pollution_intensity"] = (df["pm2_5"] + df["pm10"]) / 2
    return out


def add_weather_features(df, out):
    """Current weather, its recent history, and the dispersion physics."""
    for col in WEATHER_COLS:
        if col not in df.columns:
            continue
        series = df[col]
        out[f"{col}_rolling_mean_24h"] = series.shift(1).rolling(24).mean()
        out[f"{col}_change_24h"] = series - series.shift(24)

    out["wind_humidity_interaction"] = df["wind_speed"] * df["humidity"]
    out["temp_humidity_interaction"] = df["temperature"] * df["humidity"]
    out["temp_change_rate"] = df["temperature"] - df["temperature"].shift(1)

    if "blh" in df.columns:
        # Ventilation index = mixing depth x wind speed. This is the standard
        # meteorological measure of how much clean air is available to dilute
        # emissions into; when it collapses, AQI climbs even with constant
        # emissions. Its inverse is a stagnation score.
        ventilation = df["blh"] * df["wind_speed"]
        out["ventilation_index"] = ventilation
        out["ventilation_index_24h"] = ventilation.shift(1).rolling(24).mean()
        out["stagnation"] = 1.0 / (ventilation + 1.0)
        out["blh_min_24h"] = df["blh"].shift(1).rolling(24).min()

    if "dew_point" in df.columns:
        # Temperature minus dew point: a small spread means saturated, stable
        # air near the surface - the classic winter inversion in Lahore.
        out["inversion_proxy"] = df["temperature"] - df["dew_point"]

    if "wind_dir" in df.columns:
        # Direction is circular: 359 degrees and 1 degree are neighbours.
        radians = np.deg2rad(df["wind_dir"])
        out["wind_dir_sin"] = np.sin(radians)
        out["wind_dir_cos"] = np.cos(radians)

    if "precipitation" in df.columns:
        # Wet deposition - rain physically scrubs particulates out of the air
        past_precip = df["precipitation"].shift(1)
        out["precip_24h"] = past_precip.rolling(24).sum()
        out["precip_72h"] = past_precip.rolling(72).sum()
    return out


def add_future_weather_features(df, out, forecast_days=FORECAST_DAYS):
    """Per-day summaries of the weather AHEAD of each row.

    For day k we summarise the 24 hours ending at t + 24k. During training
    these are produced by shifting the archived weather columns backwards;
    in production build_live_features() fills the identical column names
    from the Open-Meteo weather forecast. Nothing here touches pollutant
    data, so no target information leaks in.
    """
    min_hours = MIN_FORECAST_HOURS

    for k in forecast_days:
        end = 24 * k

        def forward(series, how):
            window = series.shift(-end).rolling(24, min_periods=min_hours)
            return getattr(window, how)()

        for col in WEATHER_COLS:
            if col not in df.columns:
                continue
            out[f"f{k}_{col}_mean"] = forward(df[col], "mean")

        out[f"f{k}_wind_max"] = forward(df["wind_speed"], "max")
        out[f"f{k}_wind_min"] = forward(df["wind_speed"], "min")

        # How different is the coming day from the last observed day?
        # A jump in wind or mixing depth is what actually clears the air.
        out[f"f{k}_wind_delta"] = (forward(df["wind_speed"], "mean")
                                   - df["wind_speed"].shift(1).rolling(24).mean())

        if "precipitation" in df.columns:
            out[f"f{k}_precip_sum"] = forward(df["precipitation"], "sum")

        if "blh" in df.columns:
            out[f"f{k}_blh_min"] = forward(df["blh"], "min")
            out[f"f{k}_blh_max"] = forward(df["blh"], "max")
            out[f"f{k}_blh_delta"] = (forward(df["blh"], "mean")
                                      - df["blh"].shift(1).rolling(24).mean())
            ventilation = df["blh"] * df["wind_speed"]
            out[f"f{k}_vi_mean"] = forward(ventilation, "mean")
            out[f"f{k}_vi_min"] = forward(ventilation, "min")
    return out


def build_features(df, include_future_weather=True):
    """Build the full feature frame from an hourly dataframe.

    df must be sorted ascending by time and contain at minimum:
      timestamp, aqi, pm2_5, pm10, o3, co, no2, so2, and the weather columns.

    Returns a NEW dataframe: the original columns plus every engineered
    feature. Rows near the start (insufficient history) and, when future
    weather is included, near the end will contain NaNs - callers decide
    whether to drop them.
    """
    features = {}
    add_calendar_features(df, features)
    add_aqi_history(df, features)
    add_pollutant_history(df, features)
    add_weather_features(df, features)
    if include_future_weather:
        add_future_weather_features(df, features)

    # Drop any input column this module recomputes. The Feature Store rows
    # already carry hour/day/month/day_of_week/is_weekend, and concatenating
    # without this produced DUPLICATE column names - after which df[col]
    # returns a DataFrame instead of a Series and every model's predict()
    # fails deep inside its own input validation. The freshly computed value
    # wins because it is derived from `timestamp` here and cannot disagree
    # with the rest of the feature set.
    base = df.drop(columns=[c for c in features if c in df.columns])

    # One concat instead of hundreds of inserts - avoids pandas fragmentation
    return pd.concat([base, pd.DataFrame(features, index=df.index)],
                     axis=1).copy()



# Columns that must never be fed to a model: identifiers, the raw targets,
# and the timestamp itself (its information is already in the calendar
# features; leaving it in would let trees memorise specific dates).
NON_FEATURE_COLS = {"city", "timestamp", "unix_time", "dominant_pollutant"}


def get_feature_columns(df, target_cols=()):
    """Every numeric column that is safe to train on."""
    excluded = NON_FEATURE_COLS | set(target_cols)
    return [
        col for col in df.columns
        if col not in excluded
        and not col.startswith("target_")
        and pd.api.types.is_numeric_dtype(df[col])
    ]
