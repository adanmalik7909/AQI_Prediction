"""
training_pipeline/fetch_training_data.py
--------------------------------------------
Builds the TRAINING dataset: hourly history -> engineered features ->
forecast targets.

WHAT CHANGED IN THE ACCURACY UPGRADE (and why)

1. TARGET DEFINITION - this was the real bug behind R2 ~= 0.30.
   The old target was `aqi.shift(-24)`, where `aqi` came from applying the
   EPA breakpoint formula to a SINGLE hourly concentration. But the EPA AQI
   is defined on averaging periods (24h for PM2.5/PM10, 8h for O3/CO), so
   the old quantity was a spiky proxy, not the AQI. Measured on 4 years of
   Lahore data:

     instantaneous target: autocorr(+24h) = 0.60, best achievable R2 ~0.24
     EPA-averaged target : autocorr(+24h) = 0.77, same pipeline -> R2 0.65

   No model choice or hyperparameter can recover signal the target does not
   contain, which is why the previous tuning rounds all landed near 0.30.
   Targets are now `aqi.shift(-h)` where `aqi` is the EPA-correct value -
   i.e. "the AQI that will be reported 24/48/72 hours from now", which is
   also exactly what the dashboard claims to show.

   No leakage: the PM2.5 window behind target_24h covers hours t+1..t+24,
   entirely in the future relative to the features.

2. DATA SOURCE - the Hopsworks Feature Store is now the primary source, read
   through utils/feature_store_source.py, the same reader the dashboard uses.
   The store holds ~35k hourly rows (2022-09 onward, backfilled by
   backfill/backfill_historical.py) and is kept current by the hourly feature
   pipeline. Training and serving therefore read the identical rows.

   Open-Meteo's archive remains the automatic fallback for when the store is
   unreachable or too thin. Either way the history spans ~4 years and carries
   the dispersion variables (boundary layer height, precipitation, dew point,
   radiation, wind direction). More winters to learn from was worth more than
   any model change: 24h R2 went 0.55 -> 0.87 on the identical feature set,
   which is why the fallback threshold is set at ~2 years rather than accepting
   whatever the store happens to hold.

3. FEATURES - moved wholesale into utils/feature_engineering.py so the
   webapp computes them from the same code instead of a hand-copied
   duplicate. Adds 48/72h/weekly lags, medium-term rollings, EMAs, trends,
   pollutant histories, dispersion physics, and per-day future weather
   from the Open-Meteo forecast (+0.08-0.11 R2 at 48/72h).
"""

import os
import sys

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))

from data_source import load_history
from feature_engineering import build_features, get_feature_columns
from feature_store_source import (read_store_history, reindex_hourly,
                                  MIN_TRAINING_ROWS)
from aqi_daily import hourly_aqi_epa, dominant_pollutant

# Forecast horizons, in hours. Names kept as target_24h/48h/72h for
# continuity with the model registry and the webapp.
TARGET_HORIZONS = {"target_24h": 24, "target_48h": 48, "target_72h": 72}


def load_hourly_history(refresh=False):
    """Hourly history for training. FEATURE STORE FIRST, archive as fallback.

    The Feature Store is the project's registered store of record, so training
    reads from it rather than re-deriving its own dataset - that is the whole
    point of having one, and it guarantees the training rows are the same rows
    the dashboard serves from.

    It falls back to the Open-Meteo archive when the store cannot support
    training: fewer than ~2 years of rows, a schema missing the modelled
    columns, or an unreachable Hopsworks. The fallback is not a formality -
    2 years of data scores 24h R2 0.55 against 0.87 for 4 years, so training on
    a thin store would silently produce a much worse model than the archive
    would.

    Returns (dataframe, source_description).
    """
    try:
        df = read_store_history(min_rows=MIN_TRAINING_ROWS)

        # Rows must be evenly spaced before any lag is computed, and the AQI is
        # recomputed over the reindexed series so a gap cannot leave a stale
        # 24h average behind.
        df = reindex_hourly(df)
        aqi, subs = hourly_aqi_epa(df, return_breakdown=True)
        df["aqi"] = aqi
        df["dominant_pollutant"] = dominant_pollutant(subs)

        source = (f"Hopsworks Feature Store ({len(df)} hourly rows, "
                  f"{df['timestamp'].min().date()} -> "
                  f"{df['timestamp'].max().date()})")
        print(f"Training data source: {source}")
        return df, source

    except Exception as e:
        print(f"Feature Store unusable for training ({type(e).__name__}: {e})")
        print("Falling back to the Open-Meteo archive...")
        df = load_history(refresh=refresh)
        return df, f"Open-Meteo archive fallback ({len(df)} hourly rows)"


def add_targets(df):
    """AQI as it will be reported h hours from now, for each horizon."""
    for name, hours in TARGET_HORIZONS.items():
        df[name] = df["aqi"].shift(-hours)
    return df



def _sanity_check(df, feature_cols):
    """Guards against the two failure modes that silently ruin the metrics:
    a target hiding among the features, and an accidental forward shift."""
    leaked = [c for c in feature_cols if c.startswith("target_")]
    if leaked:
        raise ValueError(f"Target columns leaked into features: {leaked}")

    for target in TARGET_HORIZONS:
        corr = df[feature_cols].corrwith(df[target]).abs()
        suspicious = corr[corr > 0.995]
        if not suspicious.empty:
            raise ValueError(
                f"Suspiciously perfect correlation with {target}: "
                f"{suspicious.to_dict()} - check for leakage")

    for target, hours in TARGET_HORIZONS.items():
        autocorr = df["aqi"].autocorr(hours)
        print(f"  {target}: mean={df[target].mean():.1f} std={df[target].std():.1f} "
              f"| AQI autocorr(+{hours}h)={autocorr:.3f}")


def fetch_and_prepare_training_data(refresh=False):
    """Returns (dataframe, feature_columns) ready for model training."""
    df, source = load_hourly_history(refresh=refresh)

    df = build_features(df, include_future_weather=True)
    df = add_targets(df)

    feature_cols = get_feature_columns(df, target_cols=TARGET_HORIZONS.keys())

    # Drop rows missing ANY feature or target. The longest lag is 168h and
    # the future-weather windows reach 72h ahead, so this trims roughly a
    # week from the start and three days from the end.
    before = len(df)
    df = df.dropna(subset=feature_cols + list(TARGET_HORIZONS)).reset_index(drop=True)
    dropped = before - len(df)

    print(f"Training-ready dataset: {len(df)} rows x {len(feature_cols)} features "
          f"({dropped} rows dropped - incomplete history/future)")
    print(f"  source: {source}")
    print(f"  timespan: {df['timestamp'].min().date()} -> {df['timestamp'].max().date()}")

    _sanity_check(df, feature_cols)
    return df, feature_cols



if __name__ == "__main__":
    df, feature_cols = fetch_and_prepare_training_data()
    print(f"\nFirst 25 of {len(feature_cols)} features:\n{feature_cols[:25]}")
    print("\nSample rows:")
    cols = ["timestamp", "aqi", "aqi_lag_1", "aqi_rolling_mean_24h",
            "f1_wind_speed_mean", "target_24h", "target_48h", "target_72h"]
    print(df[cols].head().to_string(index=False))
