"""
tests/test_serving_consistency.py
-----------------------------------
The test that matters most for this project: does the LIVE feature row the
webapp builds match, exactly, what the models were TRAINED on?

This is the failure mode that produces confidently wrong forecasts without
raising anything. Historically the webapp re-implemented the lag/rolling
features by hand, so any change on the training side silently drifted. Both
sides now import utils/feature_engineering, and these tests assert that.

Checks:
  1. Every saved model's feature_columns.pkl is fully covered by the live
     feature row, with no NaNs.
  2. Predictions from the live row are finite and inside the AQI range.
  3. The AQI helper reproduces utils/aqi_calculator.py's breakpoint maths.
  4. Targets are genuinely shifted (no accidental leakage into features).

Run:  python -m pytest tests/test_serving_consistency.py -v
      (or just: python tests/test_serving_consistency.py)
"""

import os
import sys
import json

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "utils"))

from aqi_daily import sub_index, hourly_aqi_epa, daily_aqi_future, to_ppm, to_ppb
from aqi_calculator import calculate_aqi_from_concentrations
from feature_engineering import build_features, get_feature_columns

MODELS_DIR = os.path.join(ROOT, "trained_models")
HORIZONS = ["target_24h", "target_48h", "target_72h"]


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def live_row():
    """The exact feature row the webapp would predict from.

    Skipped rather than failed when Open-Meteo is unreachable - a network
    outage is not a defect in this code.
    """
    from data_source import load_recent, latest_observed_index
    try:
        recent = load_recent(past_days=10)
    except Exception as e:
        pytest.skip(f"Open-Meteo unreachable ({type(e).__name__})")
    featured = build_features(recent, include_future_weather=True)
    return featured.loc[latest_observed_index(featured)]




def _saved_feature_lists():
    lists = {}
    for horizon in HORIZONS:
        path = os.path.join(MODELS_DIR, horizon, "feature_columns.pkl")
        if os.path.exists(path):
            lists[horizon] = joblib.load(path)
    return lists


# ------------------------------------------------------------------ tests

def test_aqi_helper_matches_original_calculator():
    """The vectorised sub-index must agree with the project's existing
    per-row EPA implementation, so switching to it changes the averaging
    window only - never the breakpoint maths."""
    components = {"pm2_5": 85.77, "pm10": 94.21, "o3": 86.72,
                  "co": 667.55, "so2": 5.33, "no2": 6.5}

    expected, _, _ = calculate_aqi_from_concentrations(components)

    vectorised = max(
        sub_index([components["pm2_5"]], "pm2_5")[0],
        sub_index([components["pm10"]], "pm10")[0],
        sub_index(to_ppm([components["o3"]], "o3"), "o3")[0],
        sub_index(to_ppm([components["co"]], "co"), "co")[0],
        sub_index(to_ppb([components["so2"]], "so2"), "so2")[0],
        sub_index(to_ppb([components["no2"]], "no2"), "no2")[0],
    )
    # The original rounds each sub-index; allow that 1-unit difference.
    assert abs(vectorised - expected) <= 1.0


def test_sub_index_clamps_instead_of_dropping():
    """aqi_calculator returns None above the last breakpoint, which silently
    discarded the worst smog hours. The new helper clamps to 500."""
    assert sub_index([9999.0], "pm2_5")[0] == 500.0


def test_daily_targets_are_strictly_in_the_future():
    """target_day1 must not be computable from observed rows - i.e. it has
    to be built from hours AFTER t, so the tail of the series is NaN."""
    n = 400
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "pm2_5": rng.uniform(20, 200, n),
        "pm10": rng.uniform(40, 300, n),
        "o3": rng.uniform(10, 120, n),
        "co": rng.uniform(200, 2000, n),
        "so2": rng.uniform(1, 40, n),
        "no2": rng.uniform(5, 80, n),
    })
    targets = daily_aqi_future(df)
    # The final 24 hours cannot know day 1, so they must be missing.
    assert targets["target_day1"].tail(24).isna().all()
    assert targets["target_day3"].tail(72).isna().all()


def test_hourly_aqi_uses_trailing_windows():
    """A constant series must yield a constant AQI once the 24h window is
    full, and must be NaN before enough history exists."""
    n = 60
    df = pd.DataFrame({
        "pm2_5": [50.0] * n, "pm10": [80.0] * n, "o3": [40.0] * n,
        "co": [500.0] * n, "so2": [5.0] * n, "no2": [10.0] * n,
    })
    aqi = hourly_aqi_epa(df)
    assert aqi.iloc[:5].isna().all(), "AQI should be NaN before the window fills"
    assert aqi.iloc[30:].notna().all()
    assert aqi.iloc[30:].std() == pytest.approx(0.0, abs=1e-9)


def test_no_future_leakage_in_history_features():
    """History features must depend only on past values. We verify this by
    perturbing the FINAL row and confirming no earlier row's history feature
    changes."""
    n = 300
    rng = np.random.default_rng(1)
    base = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "aqi": rng.uniform(80, 300, n),
        "pm2_5": rng.uniform(20, 200, n), "pm10": rng.uniform(40, 300, n),
        "o3": rng.uniform(10, 120, n), "co": rng.uniform(200, 2000, n),
        "so2": rng.uniform(1, 40, n), "no2": rng.uniform(5, 80, n),
        "temperature": rng.uniform(5, 45, n), "humidity": rng.uniform(10, 95, n),
        "pressure": rng.uniform(980, 1020, n), "wind_speed": rng.uniform(0, 20, n),
        "cloud_cover": rng.uniform(0, 100, n),
    })

    tampered = base.copy()
    tampered.loc[n - 1, "aqi"] *= 3.0
    tampered.loc[n - 1, "pm2_5"] *= 3.0

    a = build_features(base, include_future_weather=False)
    b = build_features(tampered, include_future_weather=False)

    history_cols = [c for c in a.columns
                    if c.startswith(("aqi_lag", "aqi_rolling", "aqi_ema",
                                     "pm2_5_lag", "pm2_5_rolling"))]
    assert history_cols, "expected history features to exist"

    # Compare everything except the tampered row itself.
    pd.testing.assert_frame_equal(
        a.loc[: n - 2, history_cols], b.loc[: n - 2, history_cols],
        check_dtype=False,
    )


@pytest.mark.needs_trained_models
def test_saved_feature_lists_exist():
    lists = _saved_feature_lists()
    assert lists, ("No feature_columns.pkl found - run "
                   "training_pipeline/train_models.py first")
    for horizon, cols in lists.items():
        assert len(cols) > 50, f"{horizon}: suspiciously few features ({len(cols)})"
        assert not any(c.startswith("target_") for c in cols), \
            f"{horizon}: a target column leaked into the feature list"


@pytest.mark.needs_trained_models
@pytest.mark.parametrize("horizon", HORIZONS)
def test_live_row_covers_trained_features(horizon, live_row):

    """THE key train-serve check: every column the model expects must be
    present in the live row, and none of them may be NaN."""
    path = os.path.join(MODELS_DIR, horizon, "feature_columns.pkl")
    if not os.path.exists(path):
        pytest.skip(f"{horizon} not trained yet")
    trained_cols = joblib.load(path)

    missing = [c for c in trained_cols if c not in live_row.index]
    assert not missing, f"{horizon}: live row missing {len(missing)}: {missing[:8]}"

    values = live_row[trained_cols].astype(float)
    nans = values.index[values.isna()].tolist()
    assert not nans, f"{horizon}: NaN in live features: {nans[:8]}"


@pytest.mark.needs_trained_models
@pytest.mark.parametrize("horizon", HORIZONS)
def test_local_models_predict_sane_values(horizon, live_row):
    """Load each locally saved model and predict from the live row. Values
    must be finite and within the AQI scale."""
    horizon_dir = os.path.join(MODELS_DIR, horizon)
    fcols_path = os.path.join(horizon_dir, "feature_columns.pkl")
    results_path = os.path.join(horizon_dir, "results.csv")
    if not (os.path.exists(fcols_path) and os.path.exists(results_path)):
        pytest.skip(f"{horizon} not trained yet")

    trained_cols = joblib.load(fcols_path)
    scaler = joblib.load(os.path.join(horizon_dir, "scaler.pkl"))
    X = live_row[trained_cols].to_frame().T.astype(float)

    checked = 0
    for filename, scaled in [("xgboost.pkl", False), ("lightgbm.pkl", False),
                             ("random_forest.pkl", False), ("ridge.pkl", True)]:
        model_path = os.path.join(horizon_dir, filename)
        if not os.path.exists(model_path):
            continue
        model = joblib.load(model_path)
        pred = model.predict(scaler.transform(X) if scaled else X)[0]
        assert np.isfinite(pred), f"{horizon}/{filename}: non-finite prediction"
        assert -50 < pred < 700, f"{horizon}/{filename}: implausible AQI {pred}"
        checked += 1

    assert checked > 0, f"{horizon}: no saved models found to check"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--no-header", "-x"]))
