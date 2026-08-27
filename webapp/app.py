"""
webapp/app.py
----------------
Professional Streamlit dashboard for the Lahore AQI Predictor.

Design Philosophy:
  - Clean white/beige background, orange accents, dark text
  - Plotly charts for gauge, trends, SHAP
  - NO raw HTML card wrappers (they conflict with Streamlit's DOM)
  - All styling via single CSS block targeting Streamlit's own elements
  - Minimal vertical spacing, no empty grey boxes

Run: streamlit run webapp/app.py
"""

import sys
import os
import json
import shutil
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import shap

# --- Deployment support: Streamlit Cloud uses a "Secrets" manager (st.secrets)
# instead of a .env file. Promote any matching secrets to environment
# variables BEFORE importing hopsworks_client (which reads them at import
# time). Locally, st.secrets is simply empty/unavailable and this is a
# no-op - .env + python-dotenv still handles it as before.
try:
    for key in ["HOPSWORKS_API_KEY"]:
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass  # no secrets.toml locally - that's expected, .env handles it instead

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "training_pipeline"))

from hopsworks_client import get_feature_store, get_model_registry
from config import HOPSWORKS_PROJECT_NAME, CITY_NAME, LAT, LON
from train_models import get_feature_columns, TARGET_HORIZONS

MODEL_CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_cache")

# =====================================================================
#  KERAS COMPATIBILITY PATCHES (run once at import time)
# =====================================================================
# Models saved with Keras 3.12+ serialize config keys (quantization_config,
# input_axes, output_axes) that some builds of the SAME Keras version
# reject during deserialization. We monkey-patch at module level so it
# runs exactly once and cannot cause infinite recursion.
try:
    from tensorflow import keras as _keras

    # Patch 1: Layer.__init__ — strip quantization_config
    _orig_layer_init = _keras.layers.Layer.__init__
    def _compat_layer_init(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        return _orig_layer_init(self, *args, **kwargs)
    _keras.layers.Layer.__init__ = _compat_layer_init

    # Patch 2: Initializer.from_config — strip input_axes / output_axes
    _orig_init_from_config = _keras.initializers.Initializer.from_config.__func__
    @classmethod
    def _compat_init_from_config(cls, config):
        config = dict(config)  # avoid mutating the original
        config.pop("input_axes", None)
        config.pop("output_axes", None)
        return cls(**config)
    _keras.initializers.Initializer.from_config = _compat_init_from_config
except Exception:
    pass  # TensorFlow not installed or patch not needed

# =====================================================================
#  CONSTANTS
# =====================================================================

HORIZON_LABELS = {
    "target_24h": "24 Hours",
    "target_48h": "48 Hours",
    "target_72h": "72 Hours",
}

AQI_CATEGORIES = [
    (0, 50, "Good", "#2E7D32"),
    (51, 100, "Moderate", "#F9A825"),
    (101, 150, "Unhealthy for Sensitive Groups", "#E65100"),
    (151, 200, "Unhealthy", "#C62828"),
    (201, 300, "Very Unhealthy", "#6A1B9A"),
    (301, 500, "Hazardous", "#4A0010"),
]

HAZARD_THRESHOLD = 150

FEATURE_LABELS = {
    "aqi": "Current AQI",
    "aqi_lag_1": "1h Ago AQI",
    "aqi_lag_3": "3h Ago AQI",
    "aqi_lag_6": "6h Ago AQI",
    "aqi_lag_12": "12h Ago AQI",
    "aqi_lag_24": "24h Ago AQI",
    "aqi_change_rate": "AQI Change Rate",
    "aqi_rolling_mean_24h": "24h Avg AQI",
    "aqi_rolling_std_24h": "24h AQI Volatility",
    "aqi_rolling_min_24h": "24h Min AQI",
    "aqi_rolling_max_24h": "24h Max AQI",
    "aqi_deviation_24h": "AQI Dev from 24h Avg",
    "pm2_5": "PM2.5",
    "pm2_5_lag_1": "1h Ago PM2.5",
    "pm2_5_lag_24": "24h Ago PM2.5",
    "pm2_5_rolling_mean_24h": "24h Avg PM2.5",
    "pm10": "PM10",
    "pm10_rolling_mean_24h": "24h Avg PM10",
    "pm_ratio": "PM2.5 / PM10 Ratio",
    "temperature": "Temperature",
    "temp_change_rate": "Temp Change Rate",
    "humidity": "Humidity",
    "pressure": "Atm. Pressure",
    "wind_speed": "Wind Speed",
    "wind_speed_rolling_mean_6h": "6h Avg Wind Speed",
    "wind_humidity_interaction": "Wind × Humidity",
    "cloud_cover": "Cloud Cover",
    "o3": "Ozone (O₃)",
    "co": "Carbon Monoxide",
    "so2": "Sulphur Dioxide",
    "no2": "Nitrogen Dioxide",
    "month": "Month",
    "hour": "Hour",
    "day": "Day",
    "day_of_week": "Day of Week",
    "is_weekend": "Weekend",
    "hour_sin": "Hour (Cyclic Sin)",
    "hour_cos": "Hour (Cyclic Cos)",
    "month_sin": "Month (Cyclic Sin)",
    "month_cos": "Month (Cyclic Cos)",
    "dow_sin": "Day of Week (Cyclic Sin)",
    "dow_cos": "Day of Week (Cyclic Cos)",
}


def get_aqi_category(v):
    for lo, hi, label, color in AQI_CATEGORIES:
        if lo <= v <= hi:
            return label, color
    return "Hazardous", "#4A0010"


# =====================================================================
#  SINGLE CSS BLOCK — targets Streamlit's own DOM, no raw HTML wrappers
# =====================================================================

st.set_page_config(
    page_title=f"{CITY_NAME} Air Quality Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base ─────────────────────────────────── */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    background: linear-gradient(180deg, #FDFBF7 0%, #F5F0E8 100%) !important;
}
#MainMenu, header, footer { visibility: hidden; }

/* ── Remove excess padding ────────────────── */
.block-container { padding-top: 2rem !important; padding-bottom: 1rem !important; max-width: 1200px; }
[data-testid="stVerticalBlock"] > div { gap: 0.4rem !important; }

/* ── Smooth load animation ────────────────── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.block-container { animation: fadeIn 0.5s ease-out; }

/* ── Plotly charts — remove extra padding ─── */
.stPlotlyChart > div { border-radius: 12px !important; }
iframe[title="streamlit_pydeck"] { border-radius: 12px !important; }

/* ── Tab styling ──────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.6);
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
    border: 1px solid rgba(0,0,0,0.06);
}
.stTabs [aria-selected="true"] {
    background: #E8732A !important;
    color: white !important;
    border-color: #E8732A !important;
}

/* ── Metric styling override ──────────────── */
[data-testid="stMetricValue"] {
    font-weight: 700 !important;
    font-size: 28px !important;
}
[data-testid="stMetricDelta"] {
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)


# =====================================================================
#  PLOTLY HELPERS
# =====================================================================

def make_gauge(aqi_val, cat_label, cat_color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_val,
        number={"font": {"size": 56, "color": "#2C2C2C", "family": "Inter"}},
        title={"text": cat_label, "font": {"size": 15, "color": cat_color, "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 500], "tickwidth": 0.5, "tickcolor": "#e0e0e0",
                     "tickfont": {"size": 9, "color": "#bbb"}},
            "bar": {"color": cat_color, "thickness": 0.25},
            "bgcolor": "#fafafa",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50],    "color": "rgba(46,125,50,0.10)"},
                {"range": [51, 100],  "color": "rgba(249,168,37,0.10)"},
                {"range": [101, 150], "color": "rgba(230,81,0,0.10)"},
                {"range": [151, 200], "color": "rgba(198,40,40,0.10)"},
                {"range": [201, 300], "color": "rgba(106,27,154,0.10)"},
                {"range": [301, 500], "color": "rgba(74,0,16,0.10)"},
            ],
            "threshold": {"line": {"color": "#333", "width": 2.5}, "thickness": 0.75, "value": aqi_val},
        },
    ))
    fig.update_layout(
        height=240, margin=dict(l=24, r=24, t=24, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"family": "Inter"},
    )
    return fig


def make_trend(current_aqi, preds):
    labels = ["Now", "24h", "48h", "72h"]
    vals = [current_aqi] + [preds[h] for h in TARGET_HORIZONS]
    colors = [get_aqi_category(v)[1] for v in vals]

    fig = go.Figure()

    # Zone backgrounds
    max_v = max(vals) + 25
    for lo, hi, name, _ in AQI_CATEGORIES:
        if lo < max_v:
            fig.add_hrect(y0=lo, y1=min(hi, max_v), fillcolor=get_aqi_category(lo + 1)[1],
                          opacity=0.04, line_width=0)

    fig.add_trace(go.Scatter(
        x=labels, y=vals, mode="lines+markers+text",
        text=[f"{v:.0f}" for v in vals], textposition="top center",
        textfont={"size": 14, "color": "#333", "family": "Inter"},
        line={"color": "#E8732A", "width": 3, "shape": "spline"},
        marker={"size": 14, "color": colors, "line": {"width": 2.5, "color": "#fff"}},
        fill="tozeroy", fillcolor="rgba(232,115,42,0.05)",
        hovertemplate="<b>%{x}</b><br>AQI: %{y:.0f}<extra></extra>",
    ))

    fig.update_layout(
        height=280, margin=dict(l=16, r=16, t=8, b=36),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"showgrid": False, "tickfont": {"size": 13, "color": "#888", "family": "Inter"},
               "fixedrange": True},
        yaxis={"showgrid": True, "gridcolor": "rgba(0,0,0,0.04)", "fixedrange": True,
               "tickfont": {"size": 10, "color": "#bbb"}, "title": ""},
        showlegend=False, font={"family": "Inter"},
    )
    return fig


def make_shap_bars(feat_names, shap_vals):
    df = pd.DataFrame({"f": feat_names, "v": shap_vals})
    df["abs"] = df["v"].abs()
    df = df.nlargest(10, "abs").sort_values("abs", ascending=True)
    df["label"] = df["f"].map(lambda x: FEATURE_LABELS.get(x, x.replace("_", " ").title()))

    fig = go.Figure(go.Bar(
        x=df["v"], y=df["label"], orientation="h",
        marker={"color": ["#E8732A" if v > 0 else "#3D85C6" for v in df["v"]],
                "cornerradius": 4},
        hovertemplate="<b>%{y}</b><br>Impact: %{x:+.2f}<extra></extra>",
    ))
    fig.update_layout(
        height=320, margin=dict(l=8, r=16, t=4, b=32),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"showgrid": True, "gridcolor": "rgba(0,0,0,0.04)",
               "zeroline": True, "zerolinecolor": "rgba(0,0,0,0.08)", "fixedrange": True,
               "tickfont": {"size": 10, "color": "#bbb"},
               "title": {"text": "← Lowers AQI  ·  Raises AQI →",
                          "font": {"size": 10, "color": "#ccc"}}},
        yaxis={"tickfont": {"size": 12, "color": "#555", "family": "Inter"}, "fixedrange": True},
        showlegend=False, font={"family": "Inter"},
    )
    return fig


# =====================================================================
#  DATA LOADING (optimized for speed — webapp only needs recent data)
# =====================================================================

@st.cache_resource(show_spinner=False)
def _connect():
    fs = get_feature_store(HOPSWORKS_PROJECT_NAME)
    mr = get_model_registry(HOPSWORKS_PROJECT_NAME)
    return fs, mr


@st.cache_data(ttl=1800, show_spinner=False)
def _load_data():
    """
    Optimized data loader for the webapp — fetches only recent rows
    from Hopsworks instead of the entire 2-year history.

    We need ~50 rows to compute all lag/rolling features:
      - Max lag: 24h (aqi_lag_24)
      - Max rolling window: 24h (aqi_rolling_mean_24h etc.)
      - Total: 24 + 24 + some buffer = 50 rows is safe
    """
    import time as _time
    import numpy as np
    from config import FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION

    fs, _ = _connect()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)

    # Fetch ONLY the most recent 50 rows using a time filter
    # This is MUCH faster than fg.read() which downloads ALL data
    cutoff_time = int(_time.time()) - (50 * 3600)  # 50 hours ago

    try:
        query = fg.filter(fg["unix_time"] > cutoff_time)
        df = query.read()
        if len(df) == 0:
            raise ValueError("No rows found in recent window - pipeline may be paused")
    except Exception:
        # Fallback: if filter fails OR the recent window is empty
        # (e.g. the hourly pipeline was paused), read all history instead.
        try:
            df = fg.read()
        except Exception:
            df = fg.read(read_options={"use_hive": True})

    # --- Clean up & sort ---
    df = df.sort_values("unix_time").reset_index(drop=True)
    if "nh3" in df.columns:
        df = df.drop(columns=["nh3"])

    # --- Compute ALL the same derived features as fetch_engineered_data ---
    df["aqi_lag_1"]  = df["aqi"].shift(1)
    df["aqi_lag_3"]  = df["aqi"].shift(3)
    df["aqi_lag_6"]  = df["aqi"].shift(6)
    df["aqi_lag_12"] = df["aqi"].shift(12)
    df["aqi_lag_24"] = df["aqi"].shift(24)
    df["aqi_change_rate"] = df["aqi"] - df["aqi_lag_1"]

    aqi_shifted = df["aqi"].shift(1)
    df["aqi_rolling_mean_24h"] = aqi_shifted.rolling(window=24).mean()
    df["aqi_rolling_std_24h"]  = aqi_shifted.rolling(window=24).std()
    df["aqi_rolling_min_24h"]  = aqi_shifted.rolling(window=24).min()
    df["aqi_rolling_max_24h"]  = aqi_shifted.rolling(window=24).max()
    df["aqi_deviation_24h"] = df["aqi"] - df["aqi_rolling_mean_24h"]

    df["pm2_5_lag_1"]  = df["pm2_5"].shift(1)
    df["pm2_5_lag_24"] = df["pm2_5"].shift(24)
    df["pm2_5_rolling_mean_24h"] = df["pm2_5"].shift(1).rolling(window=24).mean()
    df["pm10_rolling_mean_24h"] = df["pm10"].shift(1).rolling(window=24).mean()
    df["pm_ratio"] = df["pm2_5"] / (df["pm10"] + 0.01)

    df["wind_speed_rolling_mean_6h"] = df["wind_speed"].shift(1).rolling(6).mean()
    df["temp_change_rate"] = df["temperature"] - df["temperature"].shift(1)
    df["wind_humidity_interaction"] = df["wind_speed"] * df["humidity"]

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

    latest = df.iloc[-1]
    feature_cols = get_feature_columns(df)
    return latest, feature_cols


@st.cache_resource(show_spinner=False)
def _load_models():
    _, mr = _connect()
    models = {}
    for h in TARGET_HORIZONS:
        name = f"aqi_model_{h}"
        meta = max(mr.get_models(name), key=lambda m: m.version)

        # Check local cache first — only download from Hopsworks if this
        # exact version isn't already saved on disk from a previous run.
        local_dir = os.path.join(MODEL_CACHE_DIR, h, f"v{meta.version}")
        metadata_path = os.path.join(local_dir, "metadata.json")

        if os.path.exists(metadata_path):
            d = local_dir
        else:
            fresh = meta.download()
            os.makedirs(os.path.dirname(local_dir), exist_ok=True)
            if os.path.exists(local_dir):
                shutil.rmtree(local_dir)
            shutil.copytree(fresh, local_dir)
            d = local_dir

        with open(os.path.join(d, "metadata.json")) as f:
            md = json.load(f)
        sc = joblib.load(os.path.join(d, "scaler.pkl"))
        if md["flavor"] == "sklearn":
            m = joblib.load(os.path.join(d, md["model_file"]))
        else:
            from tensorflow import keras
            m = keras.models.load_model(os.path.join(d, md["model_file"]), compile=False)
        models[h] = {"model": m, "scaler": sc, "meta": md, "ver": meta.version}
    return models

def _predict(row, fcols, models):
    X = row[fcols].to_frame().T
    out = {}
    for h, b in models.items():
        Xs = b["scaler"].transform(X)
        if b["meta"]["flavor"] == "sklearn":
            out[h] = float(b["model"].predict(Xs)[0])
        else:
            out[h] = float(b["model"].predict(Xs, verbose=0).flatten()[0])
    return out

def _shap(bundle, X_df):
    m = bundle["model"]
    if bundle["meta"]["flavor"] != "sklearn":
        return None
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    if isinstance(m, (XGBRegressor, RandomForestRegressor)):
        ex = shap.TreeExplainer(m)
    elif isinstance(m, Ridge):
        ex = shap.LinearExplainer(m, X_df)
    else:
        return None
    return ex(X_df).values[0]


# =====================================================================
#  LOADING SCREEN
# =====================================================================

placeholder = st.empty()
with placeholder.container():
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;padding:100px 0;opacity:0.6;">
        <div style="font-size:48px;margin-bottom:16px;">🌿</div>
        <div style="font-size:16px;color:#888;font-weight:500;">Loading Lahore AQI Dashboard...</div>
        <div style="font-size:12px;color:#bbb;margin-top:6px;">Connecting to Hopsworks · Fetching data · Loading models</div>
    </div>
    """, unsafe_allow_html=True)

latest_row, feature_cols = _load_data()
models = _load_models()
preds = _predict(latest_row, feature_cols, models)
placeholder.empty()


# =====================================================================
#  DERIVED
# =====================================================================

cur_aqi = float(latest_row["aqi"])
cur_cat, cur_color = get_aqi_category(cur_aqi)
updated = pd.to_datetime(int(latest_row["unix_time"]), unit="s")


# =====================================================================
#  HEADER
# =====================================================================

st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:2px;">
    <div>
        <span style="font-size:30px;font-weight:800;color:#2C2C2C;">🌿 {CITY_NAME} Air Quality</span>
        <div style="font-size:12px;color:#aaa;margin-top:2px;">
            Real-time AQI monitoring & 3-day ML forecast &nbsp;·&nbsp;
            Updated {updated.strftime("%b %d, %Y · %I:%M %p UTC")}
        </div>
    </div>
    <div style="font-size:11px;color:#E8732A;font-weight:500;background:rgba(232,115,42,0.08);
                padding:5px 14px;border-radius:20px;">⚡ ML-Powered</div>
</div>
""", unsafe_allow_html=True)

# Alert
haz = [h for h, v in preds.items() if v >= HAZARD_THRESHOLD]
if haz or cur_aqi >= HAZARD_THRESHOLD:
    parts = []
    if cur_aqi >= HAZARD_THRESHOLD:
        parts.append(f"Current AQI is **{cur_aqi:.0f}** ({cur_cat})")
    if haz:
        parts.append("Unhealthy levels expected in **" + ", ".join(HORIZON_LABELS[h] for h in haz) + "**")
    st.warning(" · ".join(parts) + ". Consider limiting outdoor activity. ⚠️")

st.markdown("---")


# =====================================================================
#  SECTION 1: Current Conditions
# =====================================================================

st.markdown("#### 📊 Current Conditions")
st.caption("Live data from Open-Meteo API via Hopsworks Feature Store")

c1, c2, c3 = st.columns([2.2, 1.3, 1.5])

with c1:
    st.plotly_chart(make_gauge(cur_aqi, cur_cat, cur_color),
                    use_container_width=True, config={"displayModeBar": False})

with c2:
    st.markdown("**🌤️ Weather**")
    temp = latest_row.get("temperature", 0)
    hum = latest_row.get("humidity", 0)
    wind = latest_row.get("wind_speed", 0)
    cloud = latest_row.get("cloud_cover", 0)
    pres = latest_row.get("pressure", 0)
    poll = str(latest_row.get("dominant_pollutant", "N/A")).upper()

    st.metric("🌡️ Temperature", f"{temp:.1f}°C")
    st.metric("💧 Humidity", f"{hum:.0f}%")
    st.metric("🌬️ Wind Speed", f"{wind:.1f} km/h")
    st.metric("☁️ Cloud Cover", f"{cloud:.0f}%")
    st.metric("📊 Pressure", f"{pres:.0f} hPa")
    st.metric("🎯 Pollutant", poll)

with c3:
    st.markdown(f"**📍 {CITY_NAME}, Pakistan**")
    st.map(pd.DataFrame({"lat": [LAT], "lon": [LON]}), zoom=10, use_container_width=True)
    st.caption(f"{LAT:.4f}°N, {LON:.4f}°E")

st.markdown("---")


# =====================================================================
#  SECTION 2: 3-Day Forecast
# =====================================================================

st.markdown("#### 🔮 3-Day AQI Forecast")
st.caption("Machine learning predictions powered by XGBoost · Updated with each pipeline run")

fc1, fc2, fc3 = st.columns(3)

for col, h in zip([fc1, fc2, fc3], TARGET_HORIZONS):
    v = preds[h]
    cat, color = get_aqi_category(v)
    with col:
        st.metric(
            label=HORIZON_LABELS[h],
            value=f"{v:.0f}",
            delta=cat,
            delta_color="off",
        )
        st.caption(f"🤖 {models[h]['meta']['model_name']} · v{models[h]['ver']}")

st.markdown("")
st.markdown("**📈 Forecast Trend** — Current AQI → 72-hour prediction")
st.plotly_chart(make_trend(cur_aqi, preds), use_container_width=True, config={"displayModeBar": False})

st.markdown("---")


# =====================================================================
#  SECTION 3: SHAP Explainability
# =====================================================================

st.markdown("#### 🧠 Why These Predictions?")
st.caption("SHAP (SHapley Additive exPlanations) shows which features drive each forecast")

tabs = st.tabs([f"  {HORIZON_LABELS[h]}  " for h in TARGET_HORIZONS])

for tab, h in zip(tabs, TARGET_HORIZONS):
    with tab:
        bundle = models[h]
        X = latest_row[feature_cols].to_frame().T
        X_sc = pd.DataFrame(bundle["scaler"].transform(X), columns=feature_cols)
        sv = _shap(bundle, X_sc)

        if sv is not None:
            left, right = st.columns([3, 1.2])

            with left:
                st.plotly_chart(make_shap_bars(feature_cols, sv),
                                use_container_width=True, config={"displayModeBar": False})

            with right:
                imp = pd.DataFrame({"f": feature_cols, "v": sv})
                imp["label"] = imp["f"].map(lambda x: FEATURE_LABELS.get(x, x.replace("_", " ").title()))
                top_p = imp.nlargest(1, "v").iloc[0]
                top_n = imp.nsmallest(1, "v").iloc[0]

                st.markdown("**💡 Key Insight**")
                st.markdown("")
                st.markdown(f"📈 **Risk Factor**")
                st.markdown(f"**{top_p['label']}** (+{top_p['v']:.1f})")
                st.markdown("")
                st.markdown(f"📉 **Relief Factor**")
                st.markdown(f"**{top_n['label']}** ({top_n['v']:.1f})")
        else:
            st.info(f"SHAP not available for {bundle['meta']['model_name']} (Neural Network).")


st.markdown("---")


# =====================================================================
#  FOOTER
# =====================================================================

st.markdown("""
<div style="text-align:center;font-size:11px;color:#bbb;padding:12px 0;">
    <b style="color:#888;">Lahore AQI Predictor</b> · Built by Adan Malik<br>
    Data: Open-Meteo · Feature Store & Models: Hopsworks · ML: XGBoost + TensorFlow · Explainability: SHAP<br>
    AQI Scale: 0-50 Good · 51-100 Moderate · 101-150 USG · 151-200 Unhealthy · 201-300 Very Unhealthy · 301-500 Hazardous
</div>
""", unsafe_allow_html=True)