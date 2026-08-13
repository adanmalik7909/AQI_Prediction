"""
training_pipeline/train_models.py
------------------------------------
Trains and compares 4 models - from simple statistical to deep
learning - for EACH of the 3 forecast horizons (24h, 48h, 72h):

  1. Ridge Regression   - simple linear baseline
  2. Random Forest      - tree ensemble (bagging)
  3. XGBoost             - tree ensemble (boosting) - usually strongest
                           on tabular data, great SHAP support later
  4. Neural Network      - TensorFlow/Keras (deep learning)

IMPORTANT: because this is TIME-SERIES data, we do NOT use a random
train/test split (that would leak nearby future rows into training and
give falsely optimistic scores). Instead we split CHRONOLOGICALLY:
train on the older ~80% of the timeline, test on the newest ~20% -
simulating how the model would actually perform on real future data.
"""

import os
import sys
import pandas as pd
import joblib

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from tensorflow import keras

sys.path.append(os.path.dirname(__file__))
from fetch_training_data import fetch_and_prepare_training_data
from evaluate import evaluate_predictions

TARGET_HORIZONS = ["target_24h", "target_48h", "target_72h"]
TEST_SIZE_FRACTION = 0.2
VAL_SIZE_FRACTION = 0.15   # fraction of the TRAIN set held out for early stopping (XGBoost + NN)
MODELS_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "trained_models")


def get_feature_columns(df):
    """
    All columns EXCEPT identifiers, the 3 target columns, and any
    text/object columns (e.g. dominant_pollutant) - kept simple for
    this first version.
    """
    exclude = {"city", "timestamp", "unix_time"} | set(TARGET_HORIZONS)
    return [c for c in df.columns if c not in exclude and df[c].dtype != "object"]


def chronological_split(df, test_fraction):
    """Split by TIME, not randomly - train on older data, test on newer data."""
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def build_neural_network(input_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_and_evaluate_horizon(df, feature_cols, target_col):
    print(f"\n{'='*65}\nHORIZON: {target_col}\n{'='*65}")

    # Outer split: train_full (80%) vs test (20%) - chronological
    train_full_df, test_df = chronological_split(df, TEST_SIZE_FRACTION)
    # Inner split: carve a VALIDATION set out of train_full, for early stopping
    # (XGBoost and the Neural Network use this to know when to stop training -
    #  this is the actual fix for the overfitting we saw last run)
    train_df, val_df = chronological_split(train_full_df, VAL_SIZE_FRACTION)

    print(f"Train rows: {len(train_df)}  |  Val rows: {len(val_df)}  |  Test rows: {len(test_df)}")

    X_train_full, y_train_full = train_full_df[feature_cols], train_full_df[target_col]
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    # Scale features - needed for Ridge + Neural Network, harmless for tree models
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    results = []
    trained_models = {}

    # --- 0. Naive Baseline: "AQI won't change from right now" ---
    # This tells us whether our ML models are actually adding value,
    # or just re-discovering that AQI is highly autocorrelated.
    naive_preds = test_df["aqi"].values
    results.append(evaluate_predictions(y_test, naive_preds, "Naive Baseline"))

    # --- 1. Ridge Regression (no early stopping needed -> use full train set) ---
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_full_scaled, y_train_full)
    preds = ridge.predict(X_test_scaled)
    results.append(evaluate_predictions(y_test, preds, "Ridge Regression"))
    trained_models["ridge"] = ridge

    # --- 2. Random Forest (no early stopping needed -> use full train set) ---
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train_full, y_train_full)
    preds = rf.predict(X_test)
    results.append(evaluate_predictions(y_test, preds, "Random Forest"))
    trained_models["random_forest"] = rf

    # --- 3. XGBoost (TUNED: shallower trees, regularization, early stopping) ---
    # Fixes from last run: max_depth 6->4 (less prone to memorizing noise),
    # added subsample/colsample (randomness = less overfitting), added L1/L2
    # regularization, and early_stopping_rounds so it stops the moment
    # validation performance stops improving instead of training the full
    # n_estimators regardless.
    xgb = XGBRegressor(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        base_score=float(y_train.mean()),  # explicit float - avoids a known XGBoost 3.x / SHAP bug
        random_state=42,
        eval_metric="rmse",
        early_stopping_rounds=30,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = xgb.predict(X_test)
    results.append(evaluate_predictions(y_test, preds, "XGBoost (tuned)"))
    trained_models["xgboost"] = xgb
    print(f"  (XGBoost stopped at {xgb.best_iteration} trees, out of 1000 max)")

    # --- 4. Neural Network (TensorFlow) - now uses a proper held-out validation set ---
    nn = build_neural_network(X_train_scaled.shape[1])
    nn.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        batch_size=64,
        verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
    )
    preds = nn.predict(X_test_scaled, verbose=0).flatten()
    results.append(evaluate_predictions(y_test, preds, "Neural Network (TF)"))
    trained_models["neural_network"] = nn

    results_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    print(f"\n--- Comparison for {target_col} (sorted by RMSE, lower = better) ---")
    print(results_df.to_string(index=False))

    return results_df, trained_models, scaler


def save_models(horizon_name, trained_models, scaler):
    horizon_dir = os.path.join(MODELS_OUTPUT_DIR, horizon_name)
    os.makedirs(horizon_dir, exist_ok=True)

    joblib.dump(scaler, os.path.join(horizon_dir, "scaler.pkl"))
    joblib.dump(trained_models["ridge"], os.path.join(horizon_dir, "ridge.pkl"))
    joblib.dump(trained_models["random_forest"], os.path.join(horizon_dir, "random_forest.pkl"))
    joblib.dump(trained_models["xgboost"], os.path.join(horizon_dir, "xgboost.pkl"))
    trained_models["neural_network"].save(os.path.join(horizon_dir, "neural_network.keras"))

    print(f"Saved all 4 models for {horizon_name} -> {horizon_dir}")


def main():
    df = fetch_and_prepare_training_data()
    feature_cols = get_feature_columns(df)
    print(f"\nUsing {len(feature_cols)} features:\n{feature_cols}")

    all_results = {}

    for target_col in TARGET_HORIZONS:
        results_df, trained_models, scaler = train_and_evaluate_horizon(df, feature_cols, target_col)
        all_results[target_col] = results_df
        save_models(target_col, trained_models, scaler)

        # Save the comparison table too - register_model.py reads this
        # later to know which model won, without needing to retrain anything.
        results_path = os.path.join(MODELS_OUTPUT_DIR, target_col, "results.csv")
        results_df.to_csv(results_path, index=False)

    print("\n\n" + "=" * 65)
    print("FINAL SUMMARY - best model per horizon")
    print("=" * 65)
    for horizon, results_df in all_results.items():
        best = results_df.iloc[0]
        print(f"{horizon:>12}: BEST = {best['model']:<20} "
              f"(RMSE={best['rmse']:.2f}, MAE={best['mae']:.2f}, R2={best['r2']:.3f})")


if __name__ == "__main__":
    main()