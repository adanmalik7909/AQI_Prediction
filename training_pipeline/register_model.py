"""
training_pipeline/register_model.py
--------------------------------------
Registers the SELECTED model for EACH forecast horizon into the Hopsworks
Model Registry - so the web app can load models without needing this
training code or any local files.

WHICH MODEL WINS
train_models.py marks its choice with a `selected` flag, based on mean
BACKTEST R2 across the expanding-window folds - not on the held-out tail
score. Picking by the tail score would mean selecting on the same rows we
then report as the headline metric, which quietly overstates performance.
This script honours that flag and falls back to lowest tail RMSE only if
the flag is absent (e.g. an older results.csv).

Reads the artifacts train_models.py already saved locally:
  trained_models/<horizon>/results.csv           -> which model was selected
  trained_models/<horizon>/<model file>          -> the model itself
  trained_models/<horizon>/scaler.pkl            -> the feature scaler
  trained_models/<horizon>/feature_columns.pkl   -> exact feature order

The feature list travels WITH the model. Without it, serving code has to
infer the column order, and a silent mismatch produces confident nonsense
rather than an error.
"""

import os
import sys
import json
import shutil

import joblib
import pandas as pd


sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from hopsworks_client import get_model_registry
from config import HOPSWORKS_PROJECT_NAME

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "trained_models")
TARGET_HORIZONS = ["target_24h", "target_48h", "target_72h"]

# Model display name -> (saved filename, loader "flavor"). The flavor tells
# the webapp whether to use joblib.load() or keras.models.load_model().
MODEL_FILE_MAP = {
    "Ridge Regression": ("ridge.pkl", "sklearn"),
    "Random Forest": ("random_forest.pkl", "sklearn"),
    "XGBoost (tuned)": ("xgboost.pkl", "sklearn"),
    "LightGBM": ("lightgbm.pkl", "sklearn"),
    "Neural Network (TF)": ("neural_network.keras", "keras"),
}

NAIVE_MODEL_NAME = "Naive Persistence"
BUNDLE_EXTRA_FILES = ["scaler.pkl", "feature_columns.pkl"]


def pick_winner(results_df):
    """The model train_models.py selected, or lowest-RMSE as a fallback."""
    candidates = results_df[results_df["model"] != NAIVE_MODEL_NAME]

    if "selected" in candidates.columns:
        flagged = candidates[candidates["selected"].astype(str).str.lower() == "true"]
        if not flagged.empty:
            return flagged.iloc[0]

    return candidates.sort_values("rmse").iloc[0]


def prepare_model_bundle(horizon):
    """Copies the winning model + scaler + feature list + metadata.json into
    a clean 'bundle' folder. That bundle is what gets uploaded."""
    horizon_dir = os.path.join(MODELS_DIR, horizon)
    results_df = pd.read_csv(os.path.join(horizon_dir, "results.csv"))

    winner = pick_winner(results_df)
    winner_name = winner["model"]
    filename, flavor = MODEL_FILE_MAP[winner_name]

    bundle_dir = os.path.join(horizon_dir, "registry_bundle")
    if os.path.exists(bundle_dir):
        shutil.rmtree(bundle_dir)   # never ship a stale model from a past run
    os.makedirs(bundle_dir)

    shutil.copy(os.path.join(horizon_dir, filename),
                os.path.join(bundle_dir, filename))
    for extra in BUNDLE_EXTRA_FILES:
        shutil.copy(os.path.join(horizon_dir, extra),
                    os.path.join(bundle_dir, extra))

    # Random Forests pickle to ~80 MB uncompressed, which made the dashboard's
    # first load spend minutes downloading from the registry. Re-dumping with
    # compression typically cuts that by 5-10x and changes nothing about the
    # model itself. Keras files are already compressed archives, so skip those.
    if flavor != "keras":
        bundled = os.path.join(bundle_dir, filename)
        joblib.dump(joblib.load(bundled), bundled, compress=3)


    naive_row = results_df[results_df["model"] == NAIVE_MODEL_NAME]

    metadata = {
        "horizon": horizon,
        "model_name": winner_name,
        "model_file": filename,
        "flavor": flavor,
        "rmse": float(winner["rmse"]),
        "mae": float(winner["mae"]),
        "r2": float(winner["r2"]),
    }

    # Backtest score + the naive baseline travel with the model so the
    # dashboard can show honest context instead of a bare R2.
    if "backtest_r2" in winner:
        metadata["backtest_r2"] = float(winner["backtest_r2"])
    if "backtest_r2_folds" in winner:
        metadata["backtest_r2_folds"] = str(winner["backtest_r2_folds"])
    if not naive_row.empty:
        metadata["naive_rmse"] = float(naive_row.iloc[0]["rmse"])
        metadata["naive_r2"] = float(naive_row.iloc[0]["r2"])

    with open(os.path.join(bundle_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return bundle_dir, metadata


def register_horizon_model(mr, horizon):
    bundle_dir, metadata = prepare_model_bundle(horizon)

    registry_name = f"aqi_model_{horizon}"
    metrics = {"rmse": metadata["rmse"], "mae": metadata["mae"], "r2": metadata["r2"]}
    if "backtest_r2" in metadata:
        metrics["backtest_r2"] = metadata["backtest_r2"]

    model = mr.python.create_model(
        name=registry_name,
        metrics=metrics,
        description=(f"Best model for {horizon} AQI forecast: "
                     f"{metadata['model_name']}"),
    )
    model.save(bundle_dir)

    naive = metadata.get("naive_rmse")
    beat = f", {(1 - metadata['rmse'] / naive) * 100:.1f}% below naive" if naive else ""
    print(f"Registered '{registry_name}' (winner: {metadata['model_name']}, "
          f"RMSE={metadata['rmse']:.2f}, R2={metadata['r2']:.3f}{beat}) "
          f"-> version {model.version}")


def main():
    mr = get_model_registry(HOPSWORKS_PROJECT_NAME)

    for horizon in TARGET_HORIZONS:
        register_horizon_model(mr, horizon)

    print("\nAll horizon models registered successfully!")


if __name__ == "__main__":
    main()
