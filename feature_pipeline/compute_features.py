"""
feature_pipeline/compute_features.py
--------------------------------------
Takes the raw combined data (from fetch_data.py, now Open-Meteo based)
and turns it into a clean, flat "feature row" - ready to be stored in
the Feature Store.

NOTE: Same output structure as before (same column names) - so
push_to_store.py and run_pipeline.py did NOT need any changes.
Only the raw-data-producing layer (fetch_data.py) changed.
"""

from datetime import datetime, timezone


def compute_features(raw_data):
    """
    Takes the dict returned by fetch_data.fetch_raw_data() and returns
    a single flat dictionary (one "row") of features + target, ready
    to be pushed to the Feature Store.
    """
    weather = raw_data["weather"]
    air = raw_data["air_quality"]
    aqi_info = raw_data["calculated_aqi"]

    # Open-Meteo gives time as an ISO string like "2026-08-07T01:00"
    # (we requested timezone=UTC, so this IS UTC time already)
    dt_obj = datetime.fromisoformat(weather["time"]).replace(tzinfo=timezone.utc)
    dt_unix = int(dt_obj.timestamp())

    feature_row = {
        # --- Identifiers / timestamp ---
        "city": raw_data["city"],
        "timestamp": dt_obj.isoformat(),
        "unix_time": dt_unix,

        # --- Time-based features ---
        "hour": dt_obj.hour,
        "day": dt_obj.day,
        "month": dt_obj.month,
        "day_of_week": dt_obj.weekday(),   # 0=Monday ... 6=Sunday
        "is_weekend": 1 if dt_obj.weekday() >= 5 else 0,

        # --- Weather features ---
        "temperature": weather.get("temperature_2m"),
        "humidity": weather.get("relative_humidity_2m"),
        "pressure": weather.get("surface_pressure"),
        "wind_speed": weather.get("wind_speed_10m"),
        "cloud_cover": weather.get("cloud_cover"),

        # --- Pollutant concentration features ---
        "pm2_5": air.get("pm2_5"),
        "pm10": air.get("pm10"),
        "o3": air.get("ozone"),
        "co": air.get("carbon_monoxide"),
        "so2": air.get("sulphur_dioxide"),
        "no2": air.get("nitrogen_dioxide"),
        "nh3": air.get("ammonia") if air.get("ammonia") is not None else float("nan"),

        # --- Target variable ---
        "aqi": aqi_info["overall_aqi"],
        "dominant_pollutant": aqi_info["dominant_pollutant"],
    }

    return feature_row


# Quick manual test - only runs when this file is executed directly
if __name__ == "__main__":
    import sys, os, json
    sys.path.append(os.path.dirname(__file__))
    from fetch_data import fetch_raw_data

    raw = fetch_raw_data()
    row = compute_features(raw)
    print(json.dumps(row, indent=2))