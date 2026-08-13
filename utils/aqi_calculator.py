"""
utils/aqi_calculator.py
-------------------------
Converts raw pollutant concentrations (like the ones OpenWeather gives us)
into the standard US EPA AQI (0-500 scale) using the official EPA
breakpoint formula.

This lets us compute our OWN AQI target instead of depending on
AQICN's station being alive.

NOTE: EPA's formula technically expects specific averaging periods
(24-hr for PM2.5/PM10, 8-hr for O3/CO, 1-hr for SO2/NO2). OpenWeather
gives us an instantaneous reading, so this is an approximation -
good enough for a class/internship project, but worth mentioning
as a limitation in your report.
"""

# EPA breakpoint tables: (Conc_low, Conc_high, AQI_low, AQI_high)
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]

PM10_BREAKPOINTS = [
    (0, 54, 0, 50),
    (55, 154, 51, 100),
    (155, 254, 101, 150),
    (255, 354, 151, 200),
    (355, 424, 201, 300),
    (425, 504, 301, 400),
    (505, 604, 401, 500),
]

O3_8HR_PPM_BREAKPOINTS = [
    (0.000, 0.054, 0, 50),
    (0.055, 0.070, 51, 100),
    (0.071, 0.085, 101, 150),
    (0.086, 0.105, 151, 200),
    (0.106, 0.200, 201, 300),
]

CO_8HR_PPM_BREAKPOINTS = [
    (0.0, 4.4, 0, 50),
    (4.5, 9.4, 51, 100),
    (9.5, 12.4, 101, 150),
    (12.5, 15.4, 151, 200),
    (15.5, 30.4, 201, 300),
    (30.5, 40.4, 301, 400),
    (40.5, 50.4, 401, 500),
]

SO2_1HR_PPB_BREAKPOINTS = [
    (0, 35, 0, 50),
    (36, 75, 51, 100),
    (76, 185, 101, 150),
    (186, 304, 151, 200),
]

NO2_1HR_PPB_BREAKPOINTS = [
    (0, 53, 0, 50),
    (54, 100, 51, 100),
    (101, 360, 101, 150),
    (361, 649, 151, 200),
    (650, 1249, 201, 300),
    (1250, 1649, 301, 400),
    (1650, 2049, 401, 500),
]


def _linear_aqi(conc, breakpoints):
    """Apply the EPA piecewise-linear formula for a given breakpoint table."""
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= conc <= bp_hi:
            return round(
                ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (conc - bp_lo) + aqi_lo
            )
    return None  # concentration out of all known ranges (extremely high)


def ugm3_to_ppm(conc_ugm3, molecular_weight):
    """Convert µg/m³ to ppm at 25°C, 1 atm (standard EPA conversion)."""
    return (conc_ugm3 * 24.45) / (molecular_weight * 1000)


def ugm3_to_ppb(conc_ugm3, molecular_weight):
    """Convert µg/m³ to ppb at 25°C, 1 atm."""
    return (conc_ugm3 * 24.45) / molecular_weight


def calculate_aqi_from_concentrations(components):
    """
    Takes a dict of raw pollutant concentrations in µg/m³ with keys:
      pm2_5, pm10, o3, co, so2, no2
    (works for OpenWeather-style OR Open-Meteo-style data, as long as
    fetch_data.py maps the source's field names to these keys first)

    Returns:
      - the overall AQI (max of all individual pollutant AQIs)
      - the dominant pollutant
      - a breakdown of each pollutant's individual AQI
    """
    individual_aqi = {}

    if "pm2_5" in components:
        individual_aqi["pm2_5"] = _linear_aqi(components["pm2_5"], PM25_BREAKPOINTS)

    if "pm10" in components:
        individual_aqi["pm10"] = _linear_aqi(components["pm10"], PM10_BREAKPOINTS)

    if "o3" in components:
        o3_ppm = ugm3_to_ppm(components["o3"], 48)
        individual_aqi["o3"] = _linear_aqi(o3_ppm, O3_8HR_PPM_BREAKPOINTS)

    if "co" in components:
        co_ppm = ugm3_to_ppm(components["co"], 28)
        individual_aqi["co"] = _linear_aqi(co_ppm, CO_8HR_PPM_BREAKPOINTS)

    if "so2" in components:
        so2_ppb = ugm3_to_ppb(components["so2"], 64)
        individual_aqi["so2"] = _linear_aqi(so2_ppb, SO2_1HR_PPB_BREAKPOINTS)

    if "no2" in components:
        no2_ppb = ugm3_to_ppb(components["no2"], 46)
        individual_aqi["no2"] = _linear_aqi(no2_ppb, NO2_1HR_PPB_BREAKPOINTS)

    # Remove any pollutants that fell outside all breakpoint ranges
    valid_aqis = {k: v for k, v in individual_aqi.items() if v is not None}

    if not valid_aqis:
        return None, None, individual_aqi

    dominant_pollutant = max(valid_aqis, key=valid_aqis.get)
    overall_aqi = valid_aqis[dominant_pollutant]

    return overall_aqi, dominant_pollutant, individual_aqi


# ---- Quick manual test using the REAL data you already fetched for Lahore ----
if __name__ == "__main__":
    sample_components = {
        "co": 667.55,
        "no": 0,
        "no2": 6.5,
        "o3": 86.72,
        "so2": 5.33,
        "pm2_5": 85.77,
        "pm10": 94.21,
        "nh3": 14.18,
    }

    overall_aqi, dominant, breakdown = calculate_aqi_from_concentrations(sample_components)

    print("Individual pollutant AQIs:", breakdown)
    print(f"\nDominant pollutant: {dominant}")
    print(f"Overall calculated AQI: {overall_aqi}")