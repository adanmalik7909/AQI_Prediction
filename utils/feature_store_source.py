"""
utils/feature_store_source.py
------------------------------
Reads hourly rows out of the Hopsworks Feature Store, for BOTH the training
pipeline and the dashboard.

WHY THIS MODULE EXISTS
The webapp and the training pipeline each need "give me the observed history
from the store, and tell me if it isn't usable". Written twice, the two copies
would inevitably disagree about what "usable" means - and a store that silently
satisfies one but not the other is exactly the sort of train-serve gap this
project has already been bitten by. One implementation, one definition.

CONTRACT
read_store_history() either returns a frame that can drive build_features(), or
raises with the reason. It never returns something partially usable: a caller
that gets a frame back can rely on the schema and on hourly continuity. The
reason text is surfaced in the dashboard badge and in the training log, so a
fallback is always attributable.
"""

import pandas as pd

from config import HOPSWORKS_PROJECT_NAME, FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION
from hopsworks_client import get_feature_store

# Raw columns build_features() needs before it can produce the modelled feature
# set. This list is what decides whether a feature-group version can serve the
# models at all - v4 predated the dispersion variables and cannot.
REQUIRED_RAW_COLS = [
    "timestamp", "pm2_5", "pm10", "o3", "co", "no2", "so2",
    "temperature", "humidity", "pressure", "wind_speed", "cloud_cover",
    "blh", "precipitation", "dew_point", "radiation", "wind_speed_100m",
]

# A 168h lag plus a 24h rolling window needs a genuinely continuous week; below
# this the newest rows would have NaN features and could not be predicted from.
MIN_SERVING_ROWS = 24 * 8

# Training needs enough history to span multiple winter smog seasons. Measured:
# 2 years of data gave 24h R2 0.55, 4 years gave 0.87 on the identical feature
# set. Accepting a thin store would quietly hand back a much worse model, so the
# bar is set at roughly two years and the caller falls back to the archive below
# it.
MIN_TRAINING_ROWS = 24 * 365 * 2


def read_store_history(min_rows=MIN_SERVING_ROWS, days=None):
    """Hourly rows from the Feature Store, sorted ascending, deduplicated.

    Args:
        min_rows: reject the store below this many rows (see the two constants).
        days: only read the last N days. None reads everything, which is what
            training wants; the dashboard passes a small window because reading
            four years to predict one hour would be absurd.

    Raises:
        Exception with an explanatory message whenever the store cannot drive a
        prediction - connection failure, wrong schema, or too few rows.
    """
    fs = get_feature_store(HOPSWORKS_PROJECT_NAME)
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME,
                              version=FEATURE_GROUP_VERSION)
    if fg is None:
        raise ValueError(f"feature group {FEATURE_GROUP_NAME} "
                         f"v{FEATURE_GROUP_VERSION} does not exist")

    df = _read(fg, days)

    if df is None or df.empty:
        window = f"the last {days} days" if days else "any period"
        raise ValueError(f"feature group holds no rows for {window}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df = (df.dropna(subset=["timestamp"]).sort_values("timestamp")
            .drop_duplicates("timestamp").reset_index(drop=True))

    missing = [c for c in REQUIRED_RAW_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"feature group v{FEATURE_GROUP_VERSION} schema is missing "
            f"{len(missing)} modelled columns ({', '.join(missing[:4])}...)")

    if len(df) < min_rows:
        raise ValueError(f"only {len(df)} hourly rows in the store - need "
                         f"{min_rows}+ (run backfill/backfill_historical.py)")

    return df


def _read(fg, days):
    """Pushdown-filtered read where supported, pandas-side filter otherwise."""
    if days is None:
        return fg.read()

    cutoff = int((pd.Timestamp.utcnow() - pd.Timedelta(days=days)).timestamp())
    try:
        return fg.filter(fg.unix_time >= cutoff).read()
    except Exception:
        # Some Hopsworks versions reject pushdown filters on HUDI groups.
        df = fg.read()
        return df[df["unix_time"] >= cutoff]


def reindex_hourly(df, fill_small_gaps=False, max_fill_hours=2):
    """Put rows onto a strict hourly grid.

    Lag and rolling features assume evenly spaced rows, so a missing hour would
    quietly mislabel "24 hours ago" as 23 or 25. The hourly pipeline upserts a
    rolling window specifically to avoid gaps, but a store that was paused for a
    while can still contain them, and silently trusting the spacing is how a
    feature set drifts from its own definition.

    Args:
        fill_small_gaps: if True, forward-fill isolated gaps of up to
            ``max_fill_hours`` consecutive missing hours.  Use this ONLY on the
            live inference path — training data must see the true NaNs so model
            evaluation is honest.
        max_fill_hours: maximum consecutive NaN hours to fill (default 2).
            Gaps larger than this are left as NaN so the caller can detect them
            and fall back to a row with a complete window.
    """
    full = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="h")
    out = (df.set_index("timestamp").reindex(full)
             .rename_axis("timestamp").reset_index())

    if fill_small_gaps:
        # Identify which rows were inserted (all-NaN except timestamp).
        was_missing = out.drop(columns=["timestamp"]).isna().all(axis=1)
        n_missing = int(was_missing.sum())

        if n_missing > 0:
            # Group consecutive missing rows and only fill groups that are
            # at most max_fill_hours long.
            gap_id = (~was_missing).cumsum()
            gap_sizes = was_missing.groupby(gap_id).transform("sum")
            fillable = was_missing & (gap_sizes <= max_fill_hours)
            n_filled = int(fillable.sum())
            n_unfilled = n_missing - n_filled

            if n_filled > 0:
                # Forward-fill ONLY the small-gap rows, on numeric columns.
                numeric_cols = out.select_dtypes(include="number").columns
                filled_values = out[numeric_cols].ffill()
                out.loc[fillable, numeric_cols] = filled_values.loc[fillable, numeric_cols]
                # Also forward-fill string/object columns (e.g. dominant_pollutant)
                obj_cols = out.select_dtypes(include="object").columns
                if len(obj_cols):
                    filled_obj = out[obj_cols].ffill()
                    out.loc[fillable, obj_cols] = filled_obj.loc[fillable, obj_cols]

                gap_timestamps = out.loc[fillable, "timestamp"].tolist()
                print(f"  [gap-fill] Forward-filled {n_filled} missing hour(s) "
                      f"(gaps <= {max_fill_hours}h): "
                      f"{[str(t) for t in gap_timestamps[:5]]}"
                      f"{'...' if n_filled > 5 else ''}")

            if n_unfilled > 0:
                unfilled_ts = out.loc[was_missing & ~fillable, "timestamp"]
                print(f"  [gap-fill] Left {n_unfilled} hour(s) unfilled "
                      f"(gaps > {max_fill_hours}h) starting at "
                      f"{unfilled_ts.iloc[0]}")

    return out
