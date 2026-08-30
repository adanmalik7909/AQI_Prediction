"""
training_pipeline/train_models.py
------------------------------------
Trains and compares 5 models for EACH forecast horizon (24h, 48h, 72h):

  0. Naive Persistence   - "AQI in 24h = AQI now". The baseline the
                           supervisor asked for; anything that cannot beat
                           it is worthless regardless of its R2.
  1. Ridge Regression    - linear reference point
  2. Random Forest       - tree ensemble (bagging)
  3. XGBoost             - tree ensemble (boosting), strongest here
  4. LightGBM            - leaf-wise boosting, close second
  5. Neural Network      - TensorFlow/Keras

VALIDATION - why the numbers here are trustworthy

The old script used a single chronological 80/20 split. With strong AQI
seasonality that measures one specific season: the previous test set was
spring/summer (test mean 138 vs train mean 160), so the score partly
reflected which months happened to land last. This version uses an
EXPANDING-WINDOW backtest - train on the first block, test the next, then
grow the training window and repeat. Every fold still trains only on the
past, but the reported score averages over three different test seasons
and the per-fold spread is printed so seasonal variation is visible.

The final saved model is then refit on ALL data up to a held-out tail, so
production gets a model that has seen the most recent months.
"""

import os
import sys

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from tensorflow import keras

sys.path.append(os.path.dirname(__file__))
from fetch_training_data import fetch_and_prepare_training_data, TARGET_HORIZONS
from evaluate import evaluate_predictions

HORIZON_NAMES = list(TARGET_HORIZONS)

N_FOLDS = 3            # expanding-window backtest folds
VAL_SIZE_FRACTION = 0.15   # tail of each training block held out for early stopping
FINAL_TEST_FRACTION = 0.2  # tail reserved for the headline metrics
MODELS_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "trained_models")


def chronological_split(df, test_fraction):
    """Split by TIME, not randomly - train on older data, test on newer."""
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def expanding_folds(n_rows, n_folds=N_FOLDS):
    """Yield (train_end, test_end) index pairs for an expanding backtest.

    With n_folds=3 the timeline is cut into 4 blocks:
      fold 1: train [0:1]  test [1:2]
      fold 2: train [0:2]  test [2:3]
      fold 3: train [0:3]  test [3:end]
    """
    block = n_rows // (n_folds + 1)
    for k in range(1, n_folds + 1):
        train_end = block * k
        test_end = block * (k + 1) if k < n_folds else n_rows
        yield train_end, test_end


def build_neural_network(input_dim):
    """Wider than the previous version (256->128->64->32) because the feature
    count grew from 42 to ~220."""
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="huber",       # more robust to smog-event outliers than MSE
        metrics=["mae"],
    )
    return model


def make_xgboost(y_mean):
    return XGBRegressor(
        n_estimators=3000,
        max_depth=6,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        reg_alpha=0.5,
        reg_lambda=3.0,
        min_child_weight=5,
        gamma=0.1,
        base_score=float(y_mean),  # explicit float - avoids an XGBoost 3.x/SHAP bug
        random_state=42,
        n_jobs=-1,
        eval_metric="rmse",
        early_stopping_rounds=100,
    )


def make_lightgbm():
    return LGBMRegressor(
        n_estimators=4000,
        learning_rate=0.02,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=3.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def make_random_forest():
    return RandomForestRegressor(
        n_estimators=300, max_depth=20,
        min_samples_leaf=5, min_samples_split=10,
        max_features=0.3, random_state=42, n_jobs=-1,
    )


# ---------------------------------------------------------------------
#  Fitting a single candidate on one (train, val) pair
# ---------------------------------------------------------------------

def fit_candidate(name, X_train, y_train, X_val, y_val, scaler=None):
    """Fit one model. Boosters and the NN use the validation slice for early
    stopping; Ridge/RF have no such mechanism and just see the training rows.

    Returns a callable predict(X_raw) so the caller never has to remember
    which models need scaled input.
    """
    if name == "ridge":
        model = Ridge(alpha=1.0)
        model.fit(scaler.transform(X_train), y_train)
        return model, lambda X: model.predict(scaler.transform(X))

    if name == "random_forest":
        model = make_random_forest()
        model.fit(X_train, y_train)
        return model, model.predict

    if name == "xgboost":
        model = make_xgboost(y_train.mean())
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return model, model.predict

    if name == "lightgbm":
        model = make_lightgbm()
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
        return model, model.predict

    if name == "neural_network":
        model = build_neural_network(X_train.shape[1])
        model.fit(
            scaler.transform(X_train), y_train,
            validation_data=(scaler.transform(X_val), y_val),
            epochs=300, batch_size=64, verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=25, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-5),
            ],
        )
        return model, lambda X: model.predict(scaler.transform(X), verbose=0).flatten()

    raise ValueError(f"Unknown model: {name}")


CANDIDATES = ["ridge", "random_forest", "xgboost", "lightgbm", "neural_network"]

DISPLAY_NAMES = {
    "naive": "Naive Persistence",
    "ridge": "Ridge Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost (tuned)",
    "lightgbm": "LightGBM",
    "neural_network": "Neural Network (TF)",
}


# ---------------------------------------------------------------------
#  Backtest: how well does each model type generalise across seasons?
# ---------------------------------------------------------------------

def backtest_horizon(df, feature_cols, target_col):
    """Expanding-window backtest of every candidate for one horizon.

    Returns {model_key: {"r2": mean, "r2_folds": [...], "rmse": mean, ...}}.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    scores = {name: {"r2": [], "rmse": [], "mae": []}
              for name in ["naive"] + CANDIDATES}

    for fold_num, (train_end, test_end) in enumerate(expanding_folds(len(df)), start=1):
        train_block = df.iloc[:train_end]
        test_block = df.iloc[train_end:test_end]

        inner_train, inner_val = chronological_split(train_block, VAL_SIZE_FRACTION)

        X_tr, y_tr = inner_train[feature_cols], inner_train[target_col]
        X_va, y_va = inner_val[feature_cols], inner_val[target_col]
        X_te, y_te = test_block[feature_cols], test_block[target_col]

        scaler = StandardScaler().fit(X_tr)

        # Naive persistence: the forecast IS the current AQI
        preds_by_model = {"naive": test_block["aqi"].values}

        for name in CANDIDATES:
            _, predict = fit_candidate(name, X_tr, y_tr, X_va, y_va, scaler=scaler)
            preds_by_model[name] = predict(X_te)

        print(f"  fold {fold_num}: train={len(train_block):>6} test={len(test_block):>6} "
              f"({test_block['timestamp'].min().date()} -> "
              f"{test_block['timestamp'].max().date()})")

        for name, preds in preds_by_model.items():
            scores[name]["r2"].append(r2_score(y_te, preds))
            scores[name]["rmse"].append(np.sqrt(mean_squared_error(y_te, preds)))
            scores[name]["mae"].append(mean_absolute_error(y_te, preds))

    summary = {}
    for name, vals in scores.items():
        summary[name] = {
            "r2": float(np.mean(vals["r2"])),
            "rmse": float(np.mean(vals["rmse"])),
            "mae": float(np.mean(vals["mae"])),
            "r2_folds": [round(v, 3) for v in vals["r2"]],
        }
    return summary


# ---------------------------------------------------------------------
#  Final fit: headline metrics on the most recent tail + saved artifacts
# ---------------------------------------------------------------------

def train_and_evaluate_horizon(df, feature_cols, target_col):
    """Backtest every candidate, then refit them all on the long train span
    and score on the held-out recent tail.

    The saved model is the one with the best BACKTEST R2, not the best tail
    score - picking on the tail would be selecting on the same rows we then
    report, which quietly overstates the result.
    """
    print(f"\n{'='*70}\nHORIZON: {target_col}\n{'='*70}")

    print("Expanding-window backtest:")
    backtest = backtest_horizon(df, feature_cols, target_col)

    print("\n  Backtest results (mean over folds):")
    for name in ["naive"] + CANDIDATES:
        s = backtest[name]
        print(f"    {DISPLAY_NAMES[name]:<22} R2={s['r2']:6.3f}  folds={s['r2_folds']}"
              f"  RMSE={s['rmse']:6.2f}")

    ranked = sorted(CANDIDATES, key=lambda n: backtest[n]["r2"], reverse=True)
    best_key = ranked[0]
    print(f"\n  Best by backtest R2: {DISPLAY_NAMES[best_key]}")

    # --- Final fit on everything except the most recent tail ---
    train_full, test_tail = chronological_split(df, FINAL_TEST_FRACTION)
    inner_train, inner_val = chronological_split(train_full, VAL_SIZE_FRACTION)

    X_tr, y_tr = inner_train[feature_cols], inner_train[target_col]
    X_va, y_va = inner_val[feature_cols], inner_val[target_col]
    X_te, y_te = test_tail[feature_cols], test_tail[target_col]

    scaler = StandardScaler().fit(X_tr)

    print(f"\n  Final fit: train={len(inner_train)} val={len(inner_val)} "
          f"test={len(test_tail)} "
          f"({test_tail['timestamp'].min().date()} -> "
          f"{test_tail['timestamp'].max().date()})")

    results = [evaluate_predictions(y_te, test_tail["aqi"].values,
                                    DISPLAY_NAMES["naive"])]
    trained_models = {}

    for name in CANDIDATES:
        model, predict = fit_candidate(name, X_tr, y_tr, X_va, y_va, scaler=scaler)
        row = evaluate_predictions(y_te, predict(X_te), DISPLAY_NAMES[name])
        row["backtest_r2"] = round(backtest[name]["r2"], 4)
        row["backtest_r2_folds"] = str(backtest[name]["r2_folds"])
        row["selected"] = (name == best_key)
        results.append(row)
        trained_models[name] = model

    results_df = pd.DataFrame(results)
    print(f"\n--- {target_col}: held-out tail performance ---")
    print(results_df[["model", "rmse", "mae", "r2"]].to_string(index=False))

    return results_df, trained_models, scaler, feature_cols, best_key


MODEL_FILENAMES = {
    "ridge": "ridge.pkl",
    "random_forest": "random_forest.pkl",
    "xgboost": "xgboost.pkl",
    "lightgbm": "lightgbm.pkl",
    "neural_network": "neural_network.keras",
}


def save_models(horizon_name, trained_models, scaler, feature_cols):
    horizon_dir = os.path.join(MODELS_OUTPUT_DIR, horizon_name)
    os.makedirs(horizon_dir, exist_ok=True)

    joblib.dump(scaler, os.path.join(horizon_dir, "scaler.pkl"))

    # The exact feature order the models were trained on. Saving it means the
    # webapp can validate rather than assume - a silent column reorder would
    # otherwise produce confident nonsense.
    joblib.dump(list(feature_cols), os.path.join(horizon_dir, "feature_columns.pkl"))

    for key, model in trained_models.items():
        path = os.path.join(horizon_dir, MODEL_FILENAMES[key])
        if key == "neural_network":
            model.save(path)
        else:
            joblib.dump(model, path)

    print(f"  Saved {len(trained_models)} models + scaler + feature list -> {horizon_dir}")


def main():
    df, feature_cols = fetch_and_prepare_training_data()

    all_results = {}
    for target_col in HORIZON_NAMES:
        results_df, trained_models, scaler, fcols, best_key = \
            train_and_evaluate_horizon(df, feature_cols, target_col)

        all_results[target_col] = results_df
        save_models(target_col, trained_models, scaler, fcols)

        results_path = os.path.join(MODELS_OUTPUT_DIR, target_col, "results.csv")
        results_df.to_csv(results_path, index=False)

    print("\n\n" + "=" * 70)
    print("FINAL SUMMARY - selected model per horizon")
    print("=" * 70)
    for horizon, results_df in all_results.items():
        chosen = results_df[results_df.get("selected") == True]
        best = chosen.iloc[0] if not chosen.empty else results_df.iloc[0]
        naive = results_df[results_df["model"] == DISPLAY_NAMES["naive"]].iloc[0]
        print(f"{horizon:>12}: {best['model']:<22} "
              f"RMSE={best['rmse']:6.2f} MAE={best['mae']:6.2f} R2={best['r2']:.3f}  "
              f"| backtest R2={best['backtest_r2']:.3f} {best['backtest_r2_folds']}")
        print(f"{'':>12}  vs naive persistence: RMSE={naive['rmse']:6.2f} "
              f"R2={naive['r2']:.3f}  ->  "
              f"{(1 - best['rmse'] / naive['rmse']) * 100:.1f}% lower error")


if __name__ == "__main__":
    main()
