"""
training_pipeline/explain_shap.py
------------------------------------
Generates SHAP (SHapley Additive exPlanations) explanations for the
WINNING model of each forecast horizon - answers the question:
"which features actually drove this model's predictions?"

For each horizon:
  1. Loads the winning model + scaler (saved by train_models.py /
     register_model.py) from trained_models/<horizon>/registry_bundle
  2. Rebuilds the SAME test set used during training (same chronological
     split), so explanations are computed on genuinely unseen data
  3. Computes SHAP values and saves:
       - a bar chart of average feature importance (PNG, for the report)
       - a beeswarm plot showing direction of effect (PNG, for the report)
       - a printed ranked list of the top features

NOTE: We use shap.Explainer(), SHAP's unified API - it automatically
picks the fast, exact TreeExplainer for tree models (Random Forest,
XGBoost) and the appropriate explainer for linear models (Ridge). For
the Neural Network, we pass its predict function directly (slower, but
model-agnostic and always works).

This is a separate, on-demand script - NOT part of the daily automated
training pipeline (run_training.py), since SHAP output is meant for
human review (the report / dashboard), not for the automation itself.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

sys.path.append(os.path.dirname(__file__))
from fetch_training_data import fetch_and_prepare_training_data
from train_models import get_feature_columns, chronological_split, TARGET_HORIZONS, TEST_SIZE_FRACTION

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "trained_models")
BACKGROUND_SAMPLE_SIZE = 100   # rows used as SHAP's "background" reference
EXPLAIN_SAMPLE_SIZE = 200      # rows to actually compute SHAP values for (keeps it fast)


def load_bundle(horizon):
    """Loads the winning model + scaler + metadata for a horizon."""
    bundle_dir = os.path.join(MODELS_DIR, horizon, "registry_bundle")

    with open(os.path.join(bundle_dir, "metadata.json")) as f:
        metadata = json.load(f)

    scaler = joblib.load(os.path.join(bundle_dir, "scaler.pkl"))

    if metadata["flavor"] == "sklearn":
        model = joblib.load(os.path.join(bundle_dir, metadata["model_file"]))
    else:  # keras
        from tensorflow import keras
        model = keras.models.load_model(os.path.join(bundle_dir, metadata["model_file"]))

    return model, scaler, metadata


def explain_horizon(df, feature_cols, target_col, horizon_name):
    print(f"\n{'='*65}\nSHAP EXPLANATION: {horizon_name}\n{'='*65}")

    model, scaler, metadata = load_bundle(horizon_name)
    print(f"Explaining the winning model: {metadata['model_name']}")

    _, test_df = chronological_split(df, TEST_SIZE_FRACTION)
    X_test = test_df[feature_cols]
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)

    # Modest samples - SHAP can be slow, especially on non-tree models
    background = X_test_scaled.sample(n=min(BACKGROUND_SAMPLE_SIZE, len(X_test_scaled)), random_state=42)
    explain_sample = X_test_scaled.sample(n=min(EXPLAIN_SAMPLE_SIZE, len(X_test_scaled)), random_state=42)

    if metadata["flavor"] == "sklearn":
        if isinstance(model, (XGBRegressor, RandomForestRegressor)):
            # Tree-based models: TreeExplainer is fast, exact, and purpose-built for these
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, Ridge):
            # Linear models: LinearExplainer is exact and fast for these
            explainer = shap.LinearExplainer(model, background)
        else:
            explainer = shap.Explainer(model, background)
        shap_values_array = explainer(explain_sample).values
    else:
        # Neural Network: DeepExplainer is purpose-built for TF/Keras models
        # and uses backpropagation internally - MUCH faster than the generic
        # black-box PermutationExplainer (seconds instead of tens of minutes).
        try:
            explainer = shap.DeepExplainer(model, background.values)
            raw_shap_values = explainer.shap_values(explain_sample.values)
            if isinstance(raw_shap_values, list):
                raw_shap_values = raw_shap_values[0]
            shap_values_array = np.array(raw_shap_values).reshape(len(explain_sample), -1)
        except Exception as e:
            # Fallback: DeepExplainer can occasionally be incompatible with a
            # given TF version - fall back to a much SMALLER permutation-based
            # sample so it still finishes in reasonable time.
            print(f"DeepExplainer failed ({type(e).__name__}), falling back to a smaller/slower explainer...")
            explain_sample = explain_sample.sample(n=min(30, len(explain_sample)), random_state=42)
            predict_fn = lambda x: model.predict(x, verbose=0).flatten()
            explainer = shap.Explainer(predict_fn, background)
            shap_values_array = explainer(explain_sample).values

    # --- Ranked feature importance table ---
    mean_abs_shap = np.abs(shap_values_array).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print("\nTop 10 most important features:")
    print(importance_df.head(10).to_string(index=False))

    # --- Save a summary bar plot (for the report) ---
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_array, explain_sample, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance - {horizon_name} ({metadata['model_name']})")
    plt.tight_layout()
    bar_path = os.path.join(MODELS_DIR, horizon_name, "shap_summary.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"Saved SHAP summary bar plot -> {bar_path}")

    # --- Save a beeswarm plot (shows DIRECTION of effect, not just magnitude) ---
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_array, explain_sample, show=False)
    plt.title(f"SHAP Detail (Beeswarm) - {horizon_name} ({metadata['model_name']})")
    plt.tight_layout()
    beeswarm_path = os.path.join(MODELS_DIR, horizon_name, "shap_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=150)
    plt.close()
    print(f"Saved SHAP beeswarm plot -> {beeswarm_path}")

    return importance_df


def main():
    df = fetch_and_prepare_training_data()
    feature_cols = get_feature_columns(df)

    all_importances = {}
    for target_col in TARGET_HORIZONS:
        importance_df = explain_horizon(df, feature_cols, target_col, target_col)
        all_importances[target_col] = importance_df

    print("\n\n" + "=" * 65)
    print("SUMMARY - Top 5 features per horizon")
    print("=" * 65)
    for horizon, importance_df in all_importances.items():
        top5 = ", ".join(importance_df.head(5)["feature"].tolist())
        print(f"{horizon}: {top5}")


if __name__ == "__main__":
    main()