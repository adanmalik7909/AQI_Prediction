"""
Project-wide constants.
No secrets here - just settings that define WHAT city we're working with.
Secrets (API keys) live only in .env, never here.
"""

CITY_NAME = "Lahore"
LAT = 31.5497
LON = 74.3436

# Hopsworks settings (NOT secret - just names/settings, the API key itself is in .env)
HOPSWORKS_PROJECT_NAME = "aqi_predictor_by_adan"   # <-- change this to match YOUR Hopsworks project name
FEATURE_GROUP_NAME = "aqi_features"

# v5 adds the DISPERSION-METEOROLOGY columns (boundary layer height,
# precipitation, dew point, radiation, wind direction, 100m wind) plus dust and
# aerosol optical depth. v4 held only the five original weather columns, which
# meant the Feature Store could not supply the columns the models are actually
# trained on - so the dashboard was forced onto its Open-Meteo fallback on every
# run. Bumping the version (rather than mutating v4) keeps the old group intact
# as a record of what was collected before.
FEATURE_GROUP_VERSION = 5
