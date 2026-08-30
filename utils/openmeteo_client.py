"""
utils/openmeteo_client.py
--------------------------
Reusable functions to call the Open-Meteo APIs (weather + air quality).
No API key needed - Open-Meteo is completely free and keyless.

THREE MODES used across this project:
  - LIVE (current data)   -> Forecast API + Air Quality API "current" param
  - HISTORICAL (backfill) -> Archive API + Air Quality API "start/end date"
  - FORECAST (weather)    -> Forecast API "hourly" for the next 3-4 days.
    This is what lets the models see WHERE the air is going: wind will
    disperse pollution, rain will wash it out, and a collapsing boundary
    layer traps it. Measured contribution: +0.08 to +0.11 R2 at 48/72h.

IMPORTANT NOTE (documented for the report):
The AIR QUALITY api (air-quality-api.open-meteo.com) serves BOTH live and
historical requests from the exact same underlying model - so our TARGET
variable (AQI, calculated from these pollutant concentrations) is fully
consistent between training data and live predictions.

We deliberately do NOT use Open-Meteo's pollutant FORECAST as a model
feature: it comes from the same CAMS model our target is derived from, so
feeding it back in would be circular and would inflate scores without
improving real skill. Only WEATHER forecasts (a different model) are used.

The WEATHER api uses two different endpoints (live forecast model vs
archive/reanalysis model) because years of historical weather is only
available from the archive. This introduces a small inconsistency, but
since weather is only a "helper feature" (not the target), a minor
difference here is acceptable.
"""

import time

import requests

WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Transient timeouts do happen (measured: roughly one call in ten from
# Pakistan), and without a retry a single blip takes the whole dashboard down
# with a stack trace. Three attempts with backoff turns that into a 5-second
# delay the user never notices.
MAX_RETRIES = 3
RETRY_DELAYS = [2, 5]          # seconds before attempts 2 and 3
REQUEST_TIMEOUT = 45           # per attempt; shorter than before, since we retry
ARCHIVE_TIMEOUT = 300          # years of hourly data legitimately takes longer


def _get_json(url, params, expect, timeout=REQUEST_TIMEOUT):
    """GET with retries, returning the parsed body.

    `expect` is the top-level key the response must contain ('hourly' or
    'current'); Open-Meteo signals errors with a 200 plus a 'reason' field, so
    checking the shape is what actually catches a bad request.
    """
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            data = response.json()

            if expect in data:
                return data

            # A malformed request will fail identically every time, so there is
            # nothing to gain from retrying it.
            raise ValueError(
                f"Open-Meteo returned no '{expect}' block: "
                f"{data.get('reason', data)}")

        except (requests.Timeout, requests.ConnectionError) as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                print(f"  Open-Meteo {type(e).__name__} "
                      f"(attempt {attempt + 1}/{MAX_RETRIES}), "
                      f"retrying in {delay}s...")
                time.sleep(delay)

    raise ConnectionError(
        f"Open-Meteo unreachable after {MAX_RETRIES} attempts: "
        f"{type(last_error).__name__}") from last_error


# Core weather variables, PLUS the dispersion-meteorology variables that
# actually control pollutant build-up:
#   boundary_layer_height - the "lid" on the atmosphere; when it collapses
#                           in winter, the same emissions concentrate into
#                           a much smaller volume (main smog driver)
#   precipitation         - wet deposition; rain scrubs PM out of the air
#   dew_point_2m          - with temperature gives an inversion proxy
#   wind_direction_10m    - source direction (crop burning vs city traffic)
#   shortwave_radiation   - drives photochemical ozone formation
#   wind_speed_100m       - dispersion aloft, less surface-roughness noise
WEATHER_VARS = (
    "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,"
    "cloud_cover,boundary_layer_height,precipitation,dew_point_2m,"
    "wind_direction_10m,shortwave_radiation,wind_speed_100m"
)

AIR_QUALITY_VARS = (
    "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,"
    "ammonia,dust,aerosol_optical_depth"
)

# How many days of hourly weather forecast we need. Targets reach 72h out and
# we summarise each forecast day, so we need >72h of forecast measured from
# the LAST OBSERVED pollutant hour - which itself lags real time by a few
# hours. 7 days leaves comfortable margin (Open-Meteo allows up to 16).
FORECAST_DAYS = 7


def get_current_weather(lat, lon):
    """Fetch CURRENT weather using the live Forecast API."""
    return _get_json(WEATHER_FORECAST_URL, {
        "latitude": lat,
        "longitude": lon,
        "current": WEATHER_VARS,
        "timezone": "UTC",
    }, expect="current")


def get_current_air_quality(lat, lon):
    """Fetch CURRENT air quality using the Air Quality API."""
    return _get_json(AIR_QUALITY_URL, {
        "latitude": lat,
        "longitude": lon,
        "current": AIR_QUALITY_VARS,
        "timezone": "UTC",
    }, expect="current")


def get_weather_forecast(lat, lon, forecast_days=FORECAST_DAYS):
    """Fetch HOURLY weather FORECAST for the next `forecast_days` days.

    Used at prediction time so the models know what weather the pollution
    is heading into. Returns the raw Open-Meteo response ('hourly' block).
    """
    return _get_json(WEATHER_FORECAST_URL, {
        "latitude": lat,
        "longitude": lon,
        "hourly": WEATHER_VARS,
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }, expect="hourly")



def get_recent_weather(lat, lon, past_days=10, forecast_days=FORECAST_DAYS):
    """Fetch recent history AND forecast HOURLY weather in a single call.

    This is exactly what the webapp needs: history for the lag/rolling
    features, forecast for the future-weather features.
    """
    return _get_json(WEATHER_FORECAST_URL, {
        "latitude": lat,
        "longitude": lon,
        "hourly": WEATHER_VARS,
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }, expect="hourly")


def get_recent_air_quality(lat, lon, past_days=10, forecast_days=1):
    """Fetch recent HOURLY air quality (past_days back) in one call.

    The webapp needs a continuous recent history to rebuild lag/rolling
    features; fetching it directly avoids depending on the hourly feature
    pipeline having run without a single missed hour.
    """
    return _get_json(AIR_QUALITY_URL, {
        "latitude": lat,
        "longitude": lon,
        "hourly": AIR_QUALITY_VARS,
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }, expect="hourly")


def get_historical_weather(lat, lon, start_date, end_date):
    """
    Fetch HISTORICAL weather (Archive API) for a date range.
    start_date / end_date must be strings: 'YYYY-MM-DD'.
    Returns HOURLY data for the whole range.
    """
    return _get_json(WEATHER_ARCHIVE_URL, {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": WEATHER_VARS,
        "timezone": "UTC",
    }, expect="hourly", timeout=ARCHIVE_TIMEOUT)


def get_historical_air_quality(lat, lon, start_date, end_date):
    """
    Fetch HISTORICAL air quality for a date range.
    start_date / end_date must be strings: 'YYYY-MM-DD'.
    Returns HOURLY data for the whole range.
    """
    return _get_json(AIR_QUALITY_URL, {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": AIR_QUALITY_VARS,
        "timezone": "UTC",
    }, expect="hourly", timeout=ARCHIVE_TIMEOUT)



# Quick manual test - only runs when this file is executed directly
if __name__ == "__main__":
    import json
    LAHORE_LAT, LAHORE_LON = 31.5497, 74.3436

    print("=== CURRENT WEATHER ===")
    print(json.dumps(get_current_weather(LAHORE_LAT, LAHORE_LON)["current"], indent=2))

    print("\n=== CURRENT AIR QUALITY ===")
    print(json.dumps(get_current_air_quality(LAHORE_LAT, LAHORE_LON)["current"], indent=2))

    fc = get_weather_forecast(LAHORE_LAT, LAHORE_LON)["hourly"]
    print(f"\n=== WEATHER FORECAST: {len(fc['time'])} hours, "
          f"{fc['time'][0]} -> {fc['time'][-1]} ===")
    print("variables:", [k for k in fc if k != "time"])
