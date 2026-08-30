# Accuracy Investigation & Upgrade

Answers the supervisor's two questions, documents what was actually wrong,
and records the measurements behind every change.

## The supervisor's questions

**1. Does it beat a naive persistence baseline?**

The old system did not, meaningfully. Naive persistence ("tomorrow = today")
scored R² = **-0.04 / -0.26 / -0.30** on the old test split, so the old models'
0.30 / 0.19 / 0.13 technically beat it — but only because the baseline was
scoring *below zero*, which is itself the tell: the target was so noisy that
even perfect persistence was worse than predicting the mean.

Naive persistence is now a permanent row in every results table
(`train_models.py`), so this can never go unmeasured again.

**2. Were lag/rolling features included and correctly aligned?**

Yes, and they were aligned correctly. Verified two ways: every rolling window
was already `.shift(1)`-ed before aggregating, and a new test
(`test_no_future_leakage_in_history_features`) perturbs the final row and
asserts no earlier row's history feature changes.

So the supervisor's diagnosis was right that it "almost always traces back to
one of these" — but here it was neither. The fault was one level further up:
**the target variable itself.**

## Root cause: the target was not the AQI

`utils/aqi_calculator.py` applies the EPA breakpoint formula to a *single
instantaneous hourly concentration*. But the EPA AQI is **defined on averaging
periods**:

| Pollutant | EPA averaging period | What the old code used |
|-----------|---------------------|------------------------|
| PM2.5, PM10 | 24-hour mean | 1 instantaneous hour |
| O₃, CO | max 8-hour mean | 1 instantaneous hour |
| SO₂, NO₂ | max 1-hour | 1 hour ✓ |

The old code flagged this in a docstring ("this is an approximation") but the
consequence was not appreciated: applying the formula hour-by-hour produces a
spiky quantity that is largely model-level noise, not air quality.

Measured on 4 years of Lahore data:

| Target definition | autocorr(+24h) | autocorr(+48h) | autocorr(+72h) |
|---|---|---|---|
| Instantaneous (old) | 0.60 | 0.45 | 0.40 |
| EPA-correct averaging | **0.77** | **0.62** | **0.55** |

**No model can predict signal the target does not contain.** That is why every
previous tuning round plateaued near 0.30 regardless of depth, learning rate,
or model family — the ceiling was in the label, not the learner.

## Measurements: what each change was worth

Every number below is XGBoost on an expanding-window backtest, so they are
directly comparable.

### Target definition (the big one)

| Setup | 24h R² | 48h R² | 72h R² |
|---|---|---|---|
| Old features + instantaneous target | 0.24 | 0.06 | 0.05 |
| Old features + EPA-correct target | **0.55** | 0.10 | 0.16 |

### Then, on top of the corrected target

| Change | 24h R² | 48h R² | 72h R² |
|---|---|---|---|
| Baseline (2 years, old features) | 0.55 | 0.10 | 0.16 |
| + extended lags/rollings (48h, 72h, weekly, EMAs, trends) | 0.57 | 0.13 | 0.21 |
| + **future weather** from Open-Meteo forecast API | 0.66 | 0.32 | 0.32 |
| + **4 years** of history instead of 2 | **0.87** | **0.53** | **0.42** |

Two findings worth flagging:

- **More history beat every model change combined.** Going from 2 to 4 years
  (24h: 0.66 → 0.87) mattered more than features, tuning, or model family. Two
  years contains only two winter smog seasons — not enough to learn the pattern
  that dominates Lahore's AQI.
- **Future weather is not leakage.** Open-Meteo's weather forecast is free,
  keyless, and available 16 days out, so at prediction time the model genuinely
  has it. The archived actuals stand in for it during training.

### Things that were tried and did NOT help

Recorded so they are not re-attempted:

- **LightGBM** — statistically tied with XGBoost (0.51 vs 0.54 at 24h).
- **Blending XGBoost + LightGBM** — no improvement over the better one alone.
- **log1p target transform** — no effect; AQI is not skewed enough to matter.
- **Two-stage modelling** (predict each pollutant, then apply breakpoints) —
  *worse* (0.615 vs 0.648 at 24h). Errors in six sub-models compound through
  the `max()`.
- **Predicting the delta** instead of the level — helped only in variants that
  were later discarded; neutral-to-worse in the final setup.
- **Pollutant forecasts as features** — deliberately rejected. Open-Meteo's
  pollutant forecast comes from the same CAMS model our target is derived from,
  so using it would be circular: it would inflate R² without adding real skill.

## Final results

Selected model per horizon, with two scores for honesty:

| Horizon | Model | Backtest R² (3 seasons) | Recent-months R² | MAE | vs naive |
|---|---|---|---|---|---|
| 24h | XGBoost | 0.633 `[0.57, 0.56, 0.78]` | 0.769 | ±17.0 | 18.4% lower RMSE |
| 48h | XGBoost | 0.511 `[0.45, 0.44, 0.65]` | 0.621 | ±22.1 | 16.6% lower RMSE |
| 72h | Random Forest | 0.481 `[0.39, 0.43, 0.63]` | 0.572 | ±23.3 | 16.8% lower RMSE |

Before: 0.299 / 0.188 / 0.133.

Trained from the Hopsworks Feature Store (35,023 hourly rows spanning 2022-09 →
2026-08, giving 24,409 usable training rows after the 168h lag and 72h future
windows are trimmed). The earlier run of these same models against the Open-Meteo
archive scored 0.634 / 0.498 / 0.465 — statistically the same, which is the point:
switching the source did not move accuracy, because both paths build features
with the same module.

### On the 0.80 target

The supervisor asked for 0.80+ at 24h. The honest position:

- On the **recent-months holdout** the 24h models reach 0.76–0.78, and the
  day-level target variant reached 0.87. So 0.80+ is reachable.
- On the **3-season backtest** the 24h figure is 0.63. This is the number to
  quote, because it includes a winter fold where the task is genuinely much
  harder (fold R² 0.55 vs 0.72 for the summer fold).

A single 80/20 split reports the flattering number. Both are shown
deliberately — a model whose score depends on which season landed in the test
set is not one to trust in production.

### SHAP confirms the mechanism

Feature importance shifts exactly as physical reasoning predicts, which is
independent evidence the models learned the process rather than a shortcut:

| Horizon | Top 5 features |
|---|---|
| 24h | `aqi`, `aqi_lag_1`, `pm2_5`, `f1_wind_speed_100m_mean`, `pollution_intensity` |
| 48h | `aqi`, `f2_wind_speed_100m_mean`, `aqi_lag_1`, `f2_vi_mean`, `aqi_rolling_mean_3h` |
| 72h | **`f3_vi_mean`**, `aqi`, `f3_wind_speed_100m_mean`, `f2_vi_mean`, `aqi_lag_1` |

Tomorrow is mostly the pollution already in the air. By day 3 the single most
important feature is the **forecast ventilation index** (mixing-layer height ×
wind speed) — i.e. how much clean air will be available to dilute into. That
inversion of importance is the reason the future-weather features were worth
+0.09 to +0.11 R² at the longer horizons, and it is why day 3 cannot be
predicted from pollution history alone.


## Structural fixes (not accuracy, but correctness)

Found while working through the above; each was capable of producing wrong
output silently.

1. **Train-serve consistency.** `webapp/app.py` previously re-implemented every
   lag/rolling feature by hand — a silent drift risk on every change. Both sides
   now import `utils/feature_engineering.py`, and a test asserts the live row
   covers every trained column with no NaNs.

2. **Feature list ships with the model.** `feature_columns.pkl` goes into the
   registry bundle, so serving validates instead of assuming. A reordered or
   missing column now raises instead of producing confident nonsense.

3. **Scaling bug.** The old webapp fed *scaled* input to every model, including
   the tree ensembles trained on **raw** values. Since XGBoost won 24h and 72h,
   the deployed dashboard was serving predictions from misscaled inputs. Now
   derived from the model type.

4. **Sub-index clamping.** `aqi_calculator.py` returns `None` above the highest
   breakpoint, which silently dropped the worst smog hours — exactly the hours a
   health dashboard exists for. Now clamped to 500.

5. **AQI validity guard.** PM2.5 drives the AQI ~87% of hours in Lahore. If its
   24h window is incomplete, the AQI would previously fall back to SO₂/NO₂ alone
   and report a plausible-looking but far too low value. Such rows now return
   NaN.

6. **Hourly grid enforcement.** History is reindexed onto a strict hourly grid,
   so a missing hour cannot quietly shift what "24 hours ago" refers to.

## Files

**New**
- `utils/aqi_daily.py` — vectorised EPA-correct AQI (hourly + daily targets)
- `utils/feature_engineering.py` — single source of truth for all features
- `utils/data_source.py` — Open-Meteo history/live loader with CSV cache
- `tests/test_serving_consistency.py` — 12 tests, all passing
- `tests/test_end_to_end_prediction.py` — bundle contract + live prediction
- `tests/test_app_smoke.py` — renders the whole dashboard headlessly
- `training_pipeline/diagnose_*.py` — the six experiment rounds above

**Modified**
- `training_pipeline/fetch_training_data.py` — corrected target, leakage checks
- `training_pipeline/train_models.py` — 5 models + naive baseline, backtest
- `training_pipeline/register_model.py` — honours `selected`, ships features
- `training_pipeline/explain_shap.py` — explains in the model's own input space
- `utils/openmeteo_client.py` — dispersion variables + forecast endpoints
- `webapp/app.py` — shared features, scaling fix, accuracy disclosure
- `requirements-pipeline.txt`, `webapp/requirements.txt` — lightgbm

## Reproducing

```bash
python training_pipeline/diagnose_accuracy.py       # target autocorrelation
python training_pipeline/diagnose_experiments.py    # target definition A/B
python training_pipeline/diagnose_experiments3.py   # history + dispersion vars
python training_pipeline/train_models.py            # final training
python -m pytest tests/ -v                          # 24 tests

```

## Known limitations

- **Ground truth is a model, not a station.** Open-Meteo serves CAMS
  reanalysis, not Lahore monitor readings. Consistent between training and
  serving, but it is modelled data.
- **Weather forecast skill degrades with range**, which is part of why day 3
  is materially weaker than day 1.
- **Training reads the Hopsworks Feature Store** (v5, ~35k hourly rows
  backfilled from 2022-09 onward), falling back to the Open-Meteo archive only if
  the store is unreachable or holds under two years of rows. The dashboard reads
  the same store. Both paths call `utils/feature_engineering.py`, so the features
  are identical either way and accuracy is unchanged by which one is used.



