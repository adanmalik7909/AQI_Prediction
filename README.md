# Lahore AQI Predictor

End-to-end **100% serverless** AQI forecasting system for Lahore: hourly data
collection, daily model retraining, and a Streamlit dashboard with 24/48/72-hour
forecasts and SHAP explanations.

### 🔗 Live app: **https://aqipredictionbyadan.streamlit.app/**

Deployed on Streamlit Community Cloud. It loads models from the Hopsworks Model
Registry and features from the Hopsworks Feature Store at request time, and
prints a badge at the top of the page stating which source actually served that
run — so the deployment is self-evidencing rather than merely claimed.

## Current accuracy

| Horizon | Model | Backtest R² | Recent-months R² | MAE | vs naive baseline |
|---------|-------|-------------|------------------|-----|-------------------|
| 24h | XGBoost | 0.633 | 0.769 | ±17.0 | 18.4% lower RMSE |
| 48h | XGBoost | 0.511 | 0.621 | ±22.1 | 16.6% lower RMSE |
| 72h | Random Forest | 0.481 | 0.572 | ±23.3 | 16.8% lower RMSE |

Trained from the Hopsworks Feature Store (35,023 hourly rows → 24,409 usable
training rows × 222 features).

Two scores are reported on purpose. *Backtest* averages three expanding-window
folds across different seasons and is the honest figure; *recent-months* is the
single newest holdout. Winter is genuinely harder to predict than summer, so a
model whose score depends on which season landed in the test set should not be
trusted. See [report/ACCURACY_UPGRADE.md](report/ACCURACY_UPGRADE.md) for the
full investigation — including why the original version scored only 0.30.


## Architecture

```
Open-Meteo API  ──► feature_pipeline/   ──► Hopsworks Feature Store
(air quality      (hourly, GitHub Actions)   (aqi_features v5 — store of record,
 + weather)        upserts a rolling 10-day    29 columns, ~35k hourly rows
       │           window, so a missed run     backfilled from 2022-09)
       │           self-heals)                        │
       │                                              │ utils/feature_store_source.py
       │                                              ▼ (shared reader)
       │                                    utils/feature_engineering.py
       ├────────► utils/data_source.py  ──►  (~220 features: lags, rollings,
       │          (archive + weather         dispersion physics, forecast wx)
       │           forecast — FALLBACK             │
       │           source, plus the        ┌───────┴───────────────────┐
       │           forecast half)          ▼                           ▼
       │                        training_pipeline/                webapp/app.py
       │                        (daily: 5 models × 3              (live forecast
       │                         horizons, backtest,               + SHAP + source
       │                         register winner)                    badges)
       │                                   │                           ▲
       │                                   └──► Hopsworks Model Registry ┘
       └──► utils/aqi_daily.py (EPA-correct AQI: 24h PM, 8h O₃/CO)
```

Both training and serving read the **same Feature Store** through the same
reader, and both build features with the *same* module — so the features cannot
silently drift apart. Enforced by `tests/`.

Each Hopsworks read has an automatic fallback; the badges at the top of the
dashboard report which source that particular run actually used.



## Setup

```bash
# Windows prerequisite (hopsworks -> pyjks -> twofish has no Windows wheel):
conda install -c conda-forge twofish
setx CONDA_DLL_SEARCH_MODIFICATION_ENABLE 1   # then open a NEW terminal

cp .env.example .env        # add HOPSWORKS_API_KEY
pip install -r requirements-pipeline.txt
pip install -r requirements-webapp.txt
```

Open-Meteo needs no API key.

## Running

```bash
# 1. Check the environment first (imports, keys, connectivity)
python verify_environment.py

# 2. Train + register. Downloads ~4 years of hourly history on first run
#    (~35k rows, cached under data_cache/ for 12 hours), trains 5 models per
#    horizon, backtests them, and uploads the winner to Hopsworks.
#    Takes ~20-30 min on a laptop.
python training_pipeline/run_training.py

#    Training only, no registry upload:
python training_pipeline/train_models.py

# 3. Verify (24 tests)
python -m pytest tests/ -v


# 4. Launch the dashboard
streamlit run webapp/app.py
```

### Model & feature source priority

Hopsworks is the **primary** source for both, as the project requires. The
dashboard renders a badge showing which one actually served the current run:

| Order | Models | Features (serving) | Features (training) | Badge |
|-------|--------|--------------------|---------------------|-------|
| 1 (primary) | Hopsworks Model Registry | Hopsworks Feature Store | Hopsworks Feature Store | ✅ green |
| 2 (automatic fallback) | `trained_models/<horizon>/registry_bundle/` | Open-Meteo direct | Open-Meteo archive | ⚠️ amber |

The fallback is automatic and needs no configuration — it triggers only when the
Hopsworks call fails, times out, returns the wrong schema, or holds too few rows,
and the badge then names the actual exception. Training applies a higher bar than
serving (≈2 years of rows vs 8 days) because a thin store would silently produce
a much worse model: 2 years scores 24h R² 0.55 against 0.87 for 4 years.

Future-weather features (`f1_`/`f2_`/`f3_`) always come from the Open-Meteo
forecast API, because a feature store records what happened and therefore cannot
hold tomorrow's weather.


`AQI_LOCAL_MODELS=1` is a **development-only** override that skips the network
call to speed up UI iteration. It must be set deliberately, and when it is the
dashboard says so — leave it unset for any demo or grading run:

```bash
# PowerShell
$env:AQI_LOCAL_MODELS='1'; streamlit run webapp/app.py

# bash
AQI_LOCAL_MODELS=1 streamlit run webapp/app.py
```

Optional extras:

```bash
python feature_pipeline/run_pipeline.py       # one hourly Feature Store write
python backfill/backfill_historical.py        # historical Feature Store backfill
python training_pipeline/explain_shap.py      # regenerate SHAP plots
```


## Automation

Both workflows are **live on a schedule** and can also be triggered manually
from the Actions tab.

| Workflow | Schedule | What it does |
|----------|----------|--------------|
| `.github/workflows/feature_pipeline.yml` | hourly, `0 * * * *` | Open-Meteo → features → Hopsworks Feature Store |
| `.github/workflows/training_pipeline.yml` | daily 02:00 UTC, `0 2 * * *` | Retrain 5 models × 3 horizons, backtest, register the winner, refresh SHAP |

The training workflow is gated by tests **before** training (so a broken pipeline
fails in seconds instead of after 20 minutes) and **after** registration (so a
model that cannot actually predict never reaches the dashboard).

Both need `HOPSWORKS_API_KEY` as a repository secret; Open-Meteo is keyless.

## Requirements coverage

| Requirement | Where |
|-------------|-------|
| Fetch weather + pollutant data from external API | `utils/openmeteo_client.py` (Open-Meteo: air quality + weather, keyless) |
| Time-based + derived features | `utils/feature_engineering.py` — ~220 features: calendar/cyclical, lags to 168h, rollings, EMAs, change rates, dispersion physics |
| Feature Store | Hopsworks, `feature_pipeline/push_to_store.py` — hourly rolling upsert; `utils/feature_store_source.py` reads it back for both training and serving |
| Historical backfill | `backfill/backfill_historical.py` — ~35k hourly rows (2022-09 →) into the same feature group |

| Multiple models incl. TensorFlow | `training_pipeline/train_models.py` — Ridge, Random Forest, XGBoost, LightGBM, TensorFlow NN, plus a naive baseline |
| RMSE / MAE / R² evaluation | `train_models.py`, with an expanding-window backtest and per-fold scores |
| Model Registry | Hopsworks, `training_pipeline/register_model.py` |
| Hourly + daily CI/CD | GitHub Actions, two workflows above |
| Web dashboard | `webapp/app.py` (Streamlit + Plotly), live at the link above |
| EDA / trends | `report/ACCURACY_UPGRADE.md`, `report/diagnostics/*.csv` |
| SHAP explainability | `training_pipeline/explain_shap.py` + live per-prediction SHAP in the dashboard |
| Hazardous-AQI alerts | `webapp/app.py` — banner when current or forecast AQI ≥ 150 |
| Statistical → deep learning range | Ridge (linear) → tree ensembles → TensorFlow NN, all backtested and compared |


## Project layout

```
utils/
  aqi_daily.py             EPA-correct AQI (vectorised, with averaging periods)
  aqi_calculator.py        original per-row EPA calculator (kept, still used)
  feature_engineering.py   single source of truth for features
  data_source.py           Open-Meteo history/live loader + cache
  feature_store_source.py  Feature Store reader (shared by training + serving)

  openmeteo_client.py      API wrappers (live, archive, forecast) with retries
  hopsworks_client.py      Feature Store / Model Registry handles with retries
  config.py                city, coordinates, feature group name + version
feature_pipeline/          hourly collection -> Feature Store (rolling upsert)
backfill/                  one-off full-history Feature Store backfill
training_pipeline/         training, registration, SHAP, diagnostics
webapp/app.py              Streamlit dashboard
tests/                     train-serve consistency, end-to-end, app smoke
report/                    accuracy investigation + experiment CSVs
```

## Deployment notes

Streamlit Community Cloud installs from `webapp/requirements.txt` and reads
`HOPSWORKS_API_KEY` from its Secrets manager (Settings → Secrets):

```toml
HOPSWORKS_API_KEY = "your-key"
```

Locally the same key is read from `.env`. The app checks for a `secrets.toml`
before touching `st.secrets`, so a local run does not show a spurious
"no secrets found" error.


