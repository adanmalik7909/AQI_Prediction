"""
Reusable function(s) to call the OpenWeather API (weather + air pollution).
Reads the API key from .env (never hardcode it here).
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env file and loads variables into environment

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


def get_weather_data(lat, lon):
    """
    Fetch current weather data (temperature, humidity, wind, pressure, etc.)
    Returns the raw JSON response as a Python dict.
    """
    if not OPENWEATHER_API_KEY:
        raise ValueError("OPENWEATHER_API_KEY not found. Check your .env file.")

    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    if str(data.get("cod")) != "200":
        raise ValueError(f"OpenWeather (weather) API error: {data.get('message')}")

    return data


def get_pollution_data(lat, lon):
    """
    Fetch current air pollution component concentrations (pm2.5, pm10, co, etc.)
    Returns the raw JSON response as a Python dict.
    """
    if not OPENWEATHER_API_KEY:
        raise ValueError("OPENWEATHER_API_KEY not found. Check your .env file.")

    url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}"
    response = requests.get(url)
    data = response.json()

    if "list" not in data:
        raise ValueError(f"OpenWeather (pollution) API error: {data}")

    return data


# Quick manual test - only runs when this file is executed directly
if __name__ == "__main__":
    import json
    LAHORE_LAT, LAHORE_LON = 31.5497, 74.3436
    weather = get_weather_data(LAHORE_LAT, LAHORE_LON)
    pollution = get_pollution_data(LAHORE_LAT, LAHORE_LON)
    print("=== WEATHER ===")
    print(json.dumps(weather, indent=2))
    print("\n=== POLLUTION ===")
    print(json.dumps(pollution, indent=2))