"""
webapp/app.py
----------------
Streamlit dashboard for the AQI Predictor.

What it does:
  1. Connects to Hopsworks (Feature Store + Model Registry)
  2. Reads recent feature history and engineers the same lag/rolling
     features the training pipeline uses (see fetch_training_data.py)
  3. Downloads the LATEST version of each horizon's winning model
     (24h / 48h / 72h) from the Model Registry
  4. Computes a 3-day AQI forecast
  5. Displays: current AQI, 3-day forecast, hazardous alert, and a
     SHAP explanation for each prediction

Run: streamlit run webapp/app.py
"""

import sys
import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training_pipeline"))

from hopsworks_client import get_feature_store, get_model_registry
from config import HOPSWORKS_PROJECT_NAME, CITY_NAME
from fetch_training_data import fetch_engineered_data
from train_models import get_feature_columns, TARGET_HORIZONS

HORIZON_LABELS = {
    "target_24h": "Tomorrow (24h)",
    "target_48h": "In 2 Days (48h)",
    "target_72h": "In 3 Days (72h)",
}

AQI_CATEGORIES = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#c8c800"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]

HAZARD_THRESHOLD = 150  # US EPA "Unhealthy" and above


def get_aqi_category(aqi_value):
    for lo, hi, label, color in AQI_CATEGORIES:
        if lo <= aqi_value <= hi:
            return label, color
    return "Hazardous", "#7e0023"


# ---------------------------------------------------------------------
# Cached connections / data / models - Streamlit re-runs the whole
# script on every interaction, so caching avoids re-downloading
# everything each time the user touches the page.
# ---------------------------------------------------------------------

@st.cache_resource(show_spinner="Connecting to Hopsworks...")
def get_connections():
    fs = get_feature_store(HOPSWORKS_PROJECT_NAME)
    mr = get_model_registry(HOPSWORKS_PROJECT_NAME)
    return fs, mr


@st.cache_data(ttl=1800, show_spinner="Fetching latest feature data...")
def get_latest_row():
    df = fetch_engineered_data()
    latest = df.iloc[-1]
    feature_cols = get_feature_columns(df)
    return latest, feature_cols


@st.cache_resource(show_spinner="Loading models from Model Registry...")
def load_models():
    _, mr = get_connections()
    models = {}
    for horizon in TARGET_HORIZONS:
        registry_name = f"aqi_model_{horizon}"
        all_versions = mr.get_models(registry_name)
        latest_meta = max(all_versions, key=lambda m: m.version)
        download_dir = latest_meta.download()

        with open(os.path.join(download_dir, "metadata.json")) as f:
            metadata = json.load(f)
        scaler = joblib.load(os.path.join(download_dir, "scaler.pkl"))

        if metadata["flavor"] == "sklearn":
            model = joblib.load(os.path.join(download_dir, metadata["model_file"]))
        else:
            from tensorflow import keras
            model = keras.models.load_model(os.path.join(download_dir, metadata["model_file"]))

        models[horizon] = {
            "model": model,
            "scaler": scaler,
            "metadata": metadata,
            "version": latest_meta.version,
        }
    return models


def predict_all_horizons(latest_row, feature_cols, models):
    X = latest_row[feature_cols].to_frame().T
    predictions = {}
    for horizon, bundle in models.items():
        X_scaled = bundle["scaler"].transform(X)
        model = bundle["model"]
        if bundle["metadata"]["flavor"] == "sklearn":
            pred = model.predict(X_scaled)[0]
        else:
            pred = model.predict(X_scaled, verbose=0).flatten()[0]
        predictions[horizon] = float(pred)
    return predictions


def get_shap_values(bundle, X_scaled_df):
    """Returns per-feature SHAP values for a single prediction, or None
    if this model type doesn't have a fast explainer available."""
    model = bundle["model"]
    if bundle["metadata"]["flavor"] != "sklearn":
        return None  # Neural Network - skip for a fast live dashboard

    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    if isinstance(model, (XGBRegressor, RandomForestRegressor)):
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, Ridge):
        explainer = shap.LinearExplainer(model, X_scaled_df)
    else:
        return None

    return explainer(X_scaled_df).values[0]


# ---------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------

st.set_page_config(page_title=f"{CITY_NAME} AQI Predictor", page_icon="🌫️", layout="wide")

st.title(f"🌫️ {CITY_NAME} AQI Predictor")
st.caption("A self-trained ML pipeline: Open-Meteo -> Hopsworks Feature Store -> XGBoost/TensorFlow -> Hopsworks Model Registry")

latest_row, feature_cols = get_latest_row()
models = load_models()
predictions = predict_all_horizons(latest_row, feature_cols, models)

current_aqi = latest_row["aqi"]
current_category, current_color = get_aqi_category(current_aqi)
last_updated = pd.to_datetime(int(latest_row["unix_time"]), unit="s")

# --- Hazardous alert banner ---
hazardous_horizons = [h for h, v in predictions.items() if v >= HAZARD_THRESHOLD]
if hazardous_horizons:
    labels = ", ".join(HORIZON_LABELS[h] for h in hazardous_horizons)
    st.error(f"⚠️ **Hazardous AQI expected**: {labels}. Consider limiting outdoor activity.")

# --- Current AQI ---
st.subheader("Current Conditions")
col1, col2, col3 = st.columns(3)
col1.metric("Current AQI", f"{current_aqi:.0f}", current_category)
col2.metric("Dominant Pollutant", str(latest_row.get("dominant_pollutant", "N/A")).upper())
col3.metric("Last Updated", last_updated.strftime("%b %d, %H:%M UTC"))

st.divider()

# --- 3-day forecast ---
st.subheader("3-Day Forecast")
forecast_cols = st.columns(3)
for i, horizon in enumerate(TARGET_HORIZONS):
    pred_val = predictions[horizon]
    category, color = get_aqi_category(pred_val)
    with forecast_cols[i]:
        st.metric(HORIZON_LABELS[horizon], f"{pred_val:.0f}", category)
        st.caption(f"Model: {models[horizon]['metadata']['model_name']} (v{models[horizon]['version']})")

# --- Forecast chart ---
chart_df = pd.DataFrame({
    "Time": ["Now"] + [HORIZON_LABELS[h] for h in TARGET_HORIZONS],
    "AQI": [current_aqi] + [predictions[h] for h in TARGET_HORIZONS],
})
st.line_chart(chart_df.set_index("Time"), height=300)

st.divider()

# --- SHAP explanations ---
st.subheader("Why these predictions? (SHAP Feature Importance)")
tabs = st.tabs([HORIZON_LABELS[h] for h in TARGET_HORIZONS])

for tab, horizon in zip(tabs, TARGET_HORIZONS):
    with tab:
        bundle = models[horizon]
        X = latest_row[feature_cols].to_frame().T
        X_scaled = pd.DataFrame(bundle["scaler"].transform(X), columns=feature_cols)

        shap_vals = get_shap_values(bundle, X_scaled)

        if shap_vals is not None:
            importance_df = pd.DataFrame({
                "feature": feature_cols,
                "impact": shap_vals,
            })
            importance_df["abs_impact"] = importance_df["impact"].abs()
            importance_df = importance_df.sort_values("abs_impact", ascending=False).head(10)
            st.bar_chart(importance_df.set_index("feature")["impact"], height=350)
            st.caption("Positive = pushes AQI higher for this prediction. Negative = pushes it lower.")
        else:
            st.info(f"SHAP explanation not available for {bundle['metadata']['model_name']} in this dashboard.")

st.divider()
st.caption(f"AQI scale: 0-50 Good · 51-100 Moderate · 101-150 Unhealthy for Sensitive Groups · "
           f"151-200 Unhealthy · 201-300 Very Unhealthy · 301-500 Hazardous")