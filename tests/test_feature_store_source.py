"""
tests/test_feature_store_source.py
------------------------------------
The Hopsworks Feature Store is the project's primary source for BOTH training
and serving. These tests assert the structural properties that keep it that way,
without needing network access:

  1. Training and serving read it through ONE shared reader.
  2. The reader rejects a schema that cannot build the modelled features.
  3. Hourly gaps are closed before lag features are computed.

Run:  python -m pytest tests/test_feature_store_source.py -v
"""

import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "utils"))


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
