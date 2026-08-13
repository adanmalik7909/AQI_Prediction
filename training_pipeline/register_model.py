"""
training_pipeline/register_model.py
--------------------------------------
Registers the WINNING model (lowest RMSE, excluding the Naive Baseline
which isn't a real deployable model) for EACH forecast horizon into the
Hopsworks Model Registry - so the web app can load models later without
needing this training code or any local files at all.

Reads the artifacts train_models.py already saved locally:
  trained_models/<horizon>/results.csv   -> which model won
  trained_models/<horizon>/<model file>  -> the actual winning model
  trained_models/<horizon>/scaler.pkl    -> the feature scaler

Only the WINNING model per horizon gets uploaded (not all 4 candidates) -
that's the one that will actually be used in production.
"""

import os
import sys
import json
import shutil
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from hopsworks_client import get_model_registry
from config import HOPSWORKS_PROJECT_NAME

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "trained_models")
TARGET_HORIZONS = ["target_24h", "target_48h", "target_72h"]

# Maps the model name (as printed/saved by train_models.py) -> (its saved
# filename, and a simple "flavor" tag so the web app later knows HOW to
# load it: joblib.load() for sklearn-style models, keras.models.load_model()
# for the neural network).
MODEL_FILE_MAP = {
    "Ridge Regression": ("ridge.pkl", "sklearn"),
    "Random Forest": ("random_forest.pkl", "sklearn"),
    "XGBoost (tuned)": ("xgboost.pkl", "sklearn"),
    "Neural Network (TF)": ("neural_network.keras", "keras"),
}


def prepare_model_bundle(horizon):
    """
    Reads results.csv to find the winning model for this horizon, then
    copies JUST that model's file + the scaler + a small metadata.json
    into a clean 'bundle' folder. This bundle is what gets uploaded.
    """
    horizon_dir = os.path.join(MODELS_DIR, horizon)
    results_path = os.path.join(horizon_dir, "results.csv")

    results_df = pd.read_csv(results_path)
    results_df = results_df[results_df["model"] != "Naive Baseline"]  # not deployable
    winner = results_df.sort_values("rmse").iloc[0]

    winner_name = winner["model"]
    filename, flavor = MODEL_FILE_MAP[winner_name]

    bundle_dir = os.path.join(horizon_dir, "registry_bundle")
    os.makedirs(bundle_dir, exist_ok=True)

    shutil.copy(os.path.join(horizon_dir, filename), os.path.join(bundle_dir, filename))
    shutil.copy(os.path.join(horizon_dir, "scaler.pkl"), os.path.join(bundle_dir, "scaler.pkl"))

    metadata = {
        "horizon": horizon,
        "model_name": winner_name,
        "model_file": filename,
        "flavor": flavor,
        "rmse": float(winner["rmse"]),
        "mae": float(winner["mae"]),
        "r2": float(winner["r2"]),
    }
    with open(os.path.join(bundle_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return bundle_dir, metadata


def register_horizon_model(mr, horizon):
    bundle_dir, metadata = prepare_model_bundle(horizon)

    registry_name = f"aqi_model_{horizon}"
    metrics = {"rmse": metadata["rmse"], "mae": metadata["mae"], "r2": metadata["r2"]}

    model = mr.python.create_model(
        name=registry_name,
        metrics=metrics,
        description=f"Best model for {horizon} AQI forecast: {metadata['model_name']}",
    )
    model.save(bundle_dir)

    print(f"Registered '{registry_name}' (winner: {metadata['model_name']}, "
          f"RMSE={metadata['rmse']:.2f}) -> version {model.version}")


def main():
    mr = get_model_registry(HOPSWORKS_PROJECT_NAME)

    for horizon in TARGET_HORIZONS:
        register_horizon_model(mr, horizon)

    print("\nAll horizon models registered successfully!")


if __name__ == "__main__":
    main()