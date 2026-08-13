"""
Reusable functions to call the Open-Meteo APIs (weather + air quality).
No API key needed - Open-Meteo is completely free and keyless.

TWO MODES used across this project:
  - LIVE (current data)   -> Forecast API + Air Quality API "current" param
  - HISTORICAL (backfill) -> Archive API + Air Quality API "start/end date"

IMPORTANT NOTE (documented for the report):
The AIR QUALITY api (air-quality-api.open-meteo.com) serves BOTH live and
historical requests from the exact same underlying model - so our TARGET
variable (AQI, calculated from these pollutant concentrations) is fully
consistent between training data and live predictions.

The WEATHER api uses two different endpoints (live forecast model vs
archive/reanalysis model) because 2 years of historical weather is only
available from the archive. This introduces a small inconsistency, but
since weather is only used as a "helper feature" (not the target), a
minor difference here is acceptable - see our earlier discussion on why
features only need to be informative, not perfectly precise.
"""

import requests

WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_VARS = "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,cloud_cover"
AIR_QUALITY_VARS = "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,ammonia"


def get_current_weather(lat, lon):
    """Fetch CURRENT weather using the live Forecast API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": WEATHER_VARS,
        "timezone": "UTC",
    }
    response = requests.get(WEATHER_FORECAST_URL, params=params)
    data = response.json()
    if "current" not in data:
        raise ValueError(f"Open-Meteo weather error: {data}")
    return data


def get_current_air_quality(lat, lon):
    """Fetch CURRENT air quality using the Air Quality API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": AIR_QUALITY_VARS,
        "timezone": "UTC",
    }
    response = requests.get(AIR_QUALITY_URL, params=params)
    data = response.json()
    if "current" not in data:
        raise ValueError(f"Open-Meteo air quality error: {data}")
    return data


def get_historical_weather(lat, lon, start_date, end_date):
    """
    Fetch HISTORICAL weather (Archive API) for a date range.
    start_date / end_date must be strings: 'YYYY-MM-DD'.
    Returns HOURLY data for the whole range.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": WEATHER_VARS,
        "timezone": "UTC",
    }
    response = requests.get(WEATHER_ARCHIVE_URL, params=params)
    data = response.json()
    if "hourly" not in data:
        raise ValueError(f"Open-Meteo historical weather error: {data}")
    return data


def get_historical_air_quality(lat, lon, start_date, end_date):
    """
    Fetch HISTORICAL air quality for a date range.
    start_date / end_date must be strings: 'YYYY-MM-DD'.
    Returns HOURLY data for the whole range.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": AIR_QUALITY_VARS,
        "timezone": "UTC",
    }
    response = requests.get(AIR_QUALITY_URL, params=params)
    data = response.json()
    if "hourly" not in data:
        raise ValueError(f"Open-Meteo historical air quality error: {data}")
    return data


# Quick manual test - only runs when this file is executed directly
if __name__ == "__main__":
    import json
    LAHORE_LAT, LAHORE_LON = 31.5497, 74.3436

    print("=== CURRENT WEATHER ===")
    print(json.dumps(get_current_weather(LAHORE_LAT, LAHORE_LON), indent=2))

    print("\n=== CURRENT AIR QUALITY ===")
    print(json.dumps(get_current_air_quality(LAHORE_LAT, LAHORE_LON), indent=2))