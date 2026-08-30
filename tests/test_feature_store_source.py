"""
tests/test_feature_store_source.py
------------------------------------
The Hopsworks Feature Store is the project's primary source for BOTH training
and serving. These tests assert the structural properties that keep it that way,
without needing network access:

  1. Training and serving read it through ONE shared reader.
  2. The reader rejects a schema that cannot build the modelled features.
  3. Hourly gaps are closed before lag features are computed.
  4. The insert dtypes do not depend on the host OS (a real CI failure).

Run:  python -m pytest tests/test_feature_store_source.py -v
"""

import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "utils"))


def test_store_dtypes_are_platform_independent():
    """The insert schema must not depend on the host OS.

    This is a regression test for a real CI failure: `is_weekend` was built with
    `.astype(int)`, which is int32 on Windows and int64 on Linux. Hopsworks maps
    those to 'int' and 'bigint', so a feature group registered from Windows
    rejected every GitHub Actions run:

        is_weekend (expected type: 'int', derived from input: 'bigint')

    Bare `int`/`long` widths are therefore banned in the stored columns.
    """
    sys.path.insert(0, os.path.join(ROOT, "feature_pipeline"))
    from compute_features import STORE_COLUMNS, STORE_DTYPES

    assert set(STORE_COLUMNS) == set(STORE_DTYPES), \
        "every stored column needs an explicitly pinned dtype"

    for col, dtype in STORE_DTYPES.items():
        assert dtype in ("object", "int32", "int64", "float64"), \
            f"{col} has an unpinned or ambiguous dtype: {dtype}"


def test_coerce_store_dtypes_pins_widths_regardless_of_input():
    """Whatever width the caller happens to produce, the insert gets the pinned
    one. The bug was invisible locally precisely because Windows produced the
    correct width by accident."""
    sys.path.insert(0, os.path.join(ROOT, "feature_pipeline"))
    from compute_features import STORE_DTYPES, coerce_store_dtypes

    # Deliberately the WRONG widths: int64 where int32 is required, and a float
    # where an int is required, mimicking Linux and a whole-number API response.
    row = pd.DataFrame({
        "city": ["Lahore"], "timestamp": ["2026-01-01T00:00:00+00:00"],
        "unix_time": [1767225600], "hour": pd.Series([0], dtype="int64"),
        "day": pd.Series([1], dtype="int64"),
        "month": pd.Series([1], dtype="int64"),
        "day_of_week": pd.Series([2], dtype="int64"),
        "is_weekend": pd.Series([0], dtype="int64"),
        "temperature": [12.0], "humidity": [71.0], "pressure": [1001.2],
        "wind_speed": [4.1], "cloud_cover": [30.0], "blh": [220.0],
        "precipitation": [0.0], "dew_point": [7.1], "wind_dir": [180.0],
        "radiation": [0.0], "wind_speed_100m": [8.2], "pm2_5": [85.0],
        "pm10": [120.0], "o3": [40.0], "co": [600.0], "so2": [5.0],
        "no2": [20.0], "dust": [3.0], "aod": [0.4], "aqi": [153.0],
        "dominant_pollutant": ["pm2_5"],
    })

    out = coerce_store_dtypes(row)

    for col, expected in STORE_DTYPES.items():
        assert str(out[col].dtype) == expected, \
            f"{col} is {out[col].dtype}, must be {expected}"


def test_coerce_drops_rows_missing_an_integer_feature():
    """An int column cannot hold NaN. Filling with 0 would read as a real
    measurement (0% humidity, midnight), so the row is dropped instead."""
    sys.path.insert(0, os.path.join(ROOT, "feature_pipeline"))
    from compute_features import coerce_store_dtypes

    two = pd.DataFrame({
        "is_weekend": [0, 1],
        "humidity": [71.0, float("nan")],
        "pm2_5": [85.0, 90.0],
    })

    out = coerce_store_dtypes(two)

    assert len(out) == 1, "row with a missing integer feature was not dropped"
    assert out["humidity"].iloc[0] == 71
    assert str(out["humidity"].dtype) == "int64"


def test_training_and_serving_share_one_feature_store_reader():
    """Both sides must read the store through utils/feature_store_source.

    A second, hand-rolled reader is how the schema check and the minimum-row
    check drift apart - and a store that satisfies serving but not training (or
    vice versa) is precisely the train-serve gap these tests exist to catch.
    """
    import inspect

    import feature_store_source

    train_src = open(os.path.join(ROOT, "training_pipeline",
                                  "fetch_training_data.py"),
                     encoding="utf-8").read()
    app_src = open(os.path.join(ROOT, "webapp", "app.py"),
                   encoding="utf-8").read()

    assert "read_store_history" in train_src, \
        "training does not read the Feature Store"
    assert "read_store_history" in app_src, \
        "serving does not read the Feature Store"

    # get_feature_group belongs to the shared reader only. Finding it inline in
    # either caller means a duplicate reader has crept back in.
    for name, src in (("fetch_training_data.py", train_src),
                      ("app.py", app_src)):
        assert "get_feature_group" not in src, \
            f"{name} reads the feature group directly instead of via " \
            f"feature_store_source"

    # Training must demand far more history than serving: 8 days is enough to
    # predict from, but training on it would produce a much worse model.
    assert (feature_store_source.MIN_TRAINING_ROWS
            > feature_store_source.MIN_SERVING_ROWS * 10)


def test_feature_store_reader_rejects_an_incomplete_schema():
    """The v4 feature group lacked the dispersion columns. Serving from it would
    have failed later and less clearly, so the reader has to reject it up front
    rather than hand back a frame that cannot build features."""
    from feature_store_source import REQUIRED_RAW_COLS

    for col in ["blh", "precipitation", "dew_point", "radiation",
                "wind_speed_100m"]:
        assert col in REQUIRED_RAW_COLS, \
            f"{col} is a modelled feature but not required of the store"


def test_feature_store_reindex_closes_hourly_gaps():
    """Lag features assume even hourly spacing: a missing hour would relabel
    '24 hours ago' as 23 or 25 without any error being raised."""
    from feature_store_source import reindex_hourly

    gappy = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00",
                                     "2026-01-01 05:00"]),
        "pm2_5": [10.0, 11.0, 12.0],
    })

    out = reindex_hourly(gappy)

    assert len(out) == 6, "gap not filled onto a strict hourly grid"
    assert out["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all()
    # The missing hours become NaN rather than being silently interpolated -
    # fabricating pollutant values would be worse than dropping the row later.
    assert out["pm2_5"].isna().sum() == 3
