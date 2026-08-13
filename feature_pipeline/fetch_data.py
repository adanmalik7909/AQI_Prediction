"""
feature_pipeline/fetch_data.py
--------------------------------
ORCHESTRATOR for raw data collection - now using Open-Meteo for BOTH
weather and air quality (replaces OpenWeather entirely).

WHY THE SWITCH: we need 2 years of historical data (backfill) AND live
hourly data to come from the SAME provider/model, so training data and
live prediction data are consistent ("train-serve consistency" - see
our earlier discussion). Open-Meteo gives us both, for free, no API key.

This file does NOT know how to call Open-Meteo itself - it just imports
the reusable client functions from utils/ and combines their results
into one clean dictionary. This combined raw data is what
compute_features.py will consume next.
"""

import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))

from openmeteo_client import get_current_weather, get_current_air_quality
from aqi_calculator import calculate_aqi_from_concentrations
from config import LAT, LON, CITY_NAME


def fetch_raw_data():
    """
    Calls Open-Meteo (weather + air quality), computes our own AQI target,
    and returns one combined dict of raw data for the configured city.
    """
    weather_data = get_current_weather(LAT, LON)
    air_quality_data = get_current_air_quality(LAT, LON)

    weather_current = weather_data["current"]
    air_current = air_quality_data["current"]

    # Map Open-Meteo's field names to the standard names our
    # aqi_calculator.py expects (pm2_5, pm10, o3, co, so2, no2)
    components = {
        "pm2_5": air_current.get("pm2_5"),
        "pm10": air_current.get("pm10"),
        "o3": air_current.get("ozone"),
        "co": air_current.get("carbon_monoxide"),
        "so2": air_current.get("sulphur_dioxide"),
        "no2": air_current.get("nitrogen_dioxide"),
    }

    overall_aqi, dominant_pollutant, aqi_breakdown = calculate_aqi_from_concentrations(components)

    combined = {
        "city": CITY_NAME,
        "lat": LAT,
        "lon": LON,
        "weather": weather_current,          # temperature_2m, relative_humidity_2m, surface_pressure, wind_speed_10m, cloud_cover, time
        "air_quality": air_current,          # pm2_5, pm10, ozone, carbon_monoxide, sulphur_dioxide, nitrogen_dioxide, ammonia, time
        "calculated_aqi": {
            "overall_aqi": overall_aqi,
            "dominant_pollutant": dominant_pollutant,
            "breakdown": aqi_breakdown,
        },
    }
    return combined


# Quick manual test - only runs when this file is executed directly
if __name__ == "__main__":
    raw = fetch_raw_data()
    print(json.dumps(raw, indent=2))