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
FEATURE_GROUP_VERSION = 4