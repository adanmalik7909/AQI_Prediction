"""
tests/test_end_to_end_prediction.py
-------------------------------------
End-to-end smoke test WITHOUT Hopsworks: loads each horizon's registry bundle
straight from disk exactly the way webapp/app.py loads it from the registry,
builds the live feature row, and predicts.

This catches the class of bug that unit tests miss - the pieces each work, but
the wiring between them is wrong. Specifically it would have caught the scaling
bug where the webapp fed scaled input to tree models trained on raw values.

Run:  python -m pytest tests/test_end_to_end_prediction.py -v
"""

import os
import sys
import json

import joblib
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "utils"))

from data_source import load_recent, latest_observed_index
from feature_engineering import build_features

MODELS_DIR = os.path.join(ROOT, "trained_models")
HORIZONS = ["target_24h", "target_48h", "target_72h"]

# Mirrors webapp/app.py: only these two were trained on scaled input.
SCALED_INPUT_MODELS = ("Ridge Regression", "Neural Network (TF)")


@pytest.fixture(scope="module")
def live_row():
    """Latest fully-observed feature row, built the way the webapp builds it.

    Skips (rather than fails) if Open-Meteo is unreachable: a network outage is
    not a defect in this code, and an ERROR here would be misread as one.
    """
    try:
        recent = load_recent(past_days=10)
    except Exception as e:
        pytest.skip(f"Open-Meteo unreachable ({type(e).__name__}) - "
                    f"cannot build live features")
    featured = build_features(recent, include_future_weather=True)
    return featured.loc[latest_observed_index(featured)]



def load_bundle(horizon):
    """Same contract as the webapp's registry download, read from disk."""
    d = os.path.join(MODELS_DIR, horizon, "registry_bundle")
    if not os.path.exists(os.path.join(d, "metadata.json")):
        return None

    with open(os.path.join(d, "metadata.json")) as f:
        meta = json.load(f)

    bundle = {
        "meta": meta,
        "scaler": joblib.load(os.path.join(d, "scaler.pkl")),
        "features": joblib.load(os.path.join(d, "feature_columns.pkl")),
    }

    if meta["flavor"] == "keras":
        from tensorflow import keras
        bundle["model"] = keras.models.load_model(
            os.path.join(d, meta["model_file"]), compile=False)
    else:
        bundle["model"] = joblib.load(os.path.join(d, meta["model_file"]))
    return bundle


def predict(bundle, row):
    X = row[bundle["features"]].to_frame().T.astype(float)
    assert not X.isna().any().any(), "live features contain NaNs"

    if bundle["meta"]["model_name"] in SCALED_INPUT_MODELS:
        X_in = bundle["scaler"].transform(X)
    else:
        X_in = X

    if bundle["meta"]["flavor"] == "keras":
        return float(bundle["model"].predict(X_in, verbose=0).flatten()[0])
    return float(bundle["model"].predict(X_in)[0])


@pytest.mark.parametrize("horizon", HORIZONS)
def test_bundle_is_complete(horizon):
    bundle = load_bundle(horizon)
    if bundle is None:
        pytest.skip(f"{horizon} bundle not built yet")

    meta = bundle["meta"]
    for key in ["model_name", "model_file", "flavor", "rmse", "mae", "r2"]:
        assert key in meta, f"{horizon}: metadata missing '{key}'"

    assert len(bundle["features"]) > 50
    # The scaler must have been fitted on the same width as the feature list,
    # otherwise Ridge/NN would silently receive misaligned columns.
    assert bundle["scaler"].n_features_in_ == len(bundle["features"])


@pytest.mark.parametrize("horizon", HORIZONS)
def test_end_to_end_prediction_is_plausible(horizon, live_row):
    bundle = load_bundle(horizon)
    if bundle is None:
        pytest.skip(f"{horizon} bundle not built yet")

    pred = predict(bundle, live_row)
    current = float(live_row["aqi"])

    assert np.isfinite(pred), "prediction is not finite"
    assert 0 <= pred <= 500, f"AQI outside the defined scale: {pred}"

    # A 3-day forecast that departs from current conditions by more than the
    # entire scale would indicate a feature-alignment problem, not weather.
    assert abs(pred - current) < 300, \
        f"{horizon}: implausible jump (now={current:.0f}, pred={pred:.0f})"


def test_forecasts_are_ordered_sensibly(live_row):
    """Not a strict requirement - AQI genuinely can rise then fall - but all
    three horizons collapsing to the same value, or diverging wildly, points at
    a wiring problem."""
    preds = {}
    for horizon in HORIZONS:
        bundle = load_bundle(horizon)
        if bundle is None:
            pytest.skip("bundles not built yet")
        preds[horizon] = predict(bundle, live_row)

    values = list(preds.values())
    assert len(set(round(v, 6) for v in values)) > 1, \
        f"all horizons predicted identical values: {preds}"
    assert max(values) - min(values) < 250, f"horizons diverge wildly: {preds}"


def test_scaling_choice_actually_matters(live_row):
    """Guards the bug that was found: if a tree model is fed scaled input it
    still returns a number, just a wrong one. Assert the two paths differ, so
    the distinction can never be dismissed as cosmetic."""
    bundle = load_bundle("target_24h")
    if bundle is None or bundle["meta"]["model_name"] in SCALED_INPUT_MODELS:
        pytest.skip("24h winner is a scaled-input model")

    X = live_row[bundle["features"]].to_frame().T.astype(float)
    correct = float(bundle["model"].predict(X)[0])
    wrong = float(bundle["model"].predict(bundle["scaler"].transform(X))[0])

    assert abs(correct - wrong) > 1.0, (
        "Scaled and raw input produced nearly the same prediction - the "
        "regression test for the scaling bug is no longer meaningful")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--no-header"]))
