"""
training_pipeline/evaluate.py
--------------------------------
Reusable evaluation function: computes RMSE, MAE, and R2 for any
model's predictions, and prints them in a consistent format.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_predictions(y_true, y_pred, model_name=""):
    """
    Computes RMSE, MAE, and R2 for a set of predictions and returns
    them as a dict (so results can be collected into a comparison table).
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"{model_name:<25} RMSE: {rmse:6.2f} | MAE: {mae:6.2f} | R2: {r2:.3f}")

    return {"model": model_name, "rmse": rmse, "mae": mae, "r2": r2}