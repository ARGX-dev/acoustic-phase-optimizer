from __future__ import annotations

import json
import ssl
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from typing import Optional

from acoustic_phase_optimizer.weather.models import Location, WeatherData, YearlyAverages
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search?name={name}&count=1&language=en&format=json"
ARCHIVE_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
    "?latitude={lat}&longitude={lon}"
    "&start_date={start}&end_date={end}"
    "&daily=temperature_2m_mean,relative_humidity_2m_mean,surface_pressure_mean"
    "&timezone=auto"
)


def _fetch_json(url: str) -> Optional[dict]:
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(url, headers={"User-Agent": "AcousticPhaseOptimizer/1.0"})
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError) as e:
        logger.error(f"Weather API request failed: {e}")
        return None


def fetch_location_coords(location_name: str) -> Optional[Location]:
    if not location_name.strip():
        return None
    data = _fetch_json(GEOCODING_URL.format(name=urllib.parse.quote(location_name)))
    if not data or "results" not in data or not data["results"]:
        logger.warning(f"Location not found: {location_name}")
        return None
    result = data["results"][0]
    loc = Location(
        name=result.get("name", location_name),
        latitude=result["latitude"],
        longitude=result["longitude"],
        country=result.get("country", ""),
    )
    logger.info(f"Resolved {location_name} → {loc.name}, {loc.latitude:.4f}, {loc.longitude:.4f}")
    return loc


def fetch_yearly_averages(
    location: Location,
    days_back: int = 365,
) -> Optional[YearlyAverages]:
    end = datetime.utcnow()
    start = end - timedelta(days=days_back)

    url = ARCHIVE_URL.format(
        lat=location.latitude,
        lon=location.longitude,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    data = _fetch_json(url)
    if not data or "daily" not in data:
        logger.warning("No weather data returned")
        return None

    daily = data["daily"]
    dates = daily.get("time", [])
    temps = daily.get("temperature_2m_mean", [])
    hums = daily.get("relative_humidity_2m_mean", [])
    pressures = daily.get("surface_pressure_mean", [])

    daily_records = []
    temp_sum = hum_sum = press_sum = 0.0
    count = 0

    for i in range(len(dates)):
        t = temps[i] if i < len(temps) and temps[i] is not None else None
        h = hums[i] if i < len(hums) and hums[i] is not None else None
        p = pressures[i] if i < len(pressures) and pressures[i] is not None else None

        if t is not None:
            temp_sum += t
            hum_sum += (h if h is not None else 50.0)
            press_sum += (p if p is not None else 1013.25)
            count += 1
            daily_records.append(WeatherData(
                date=dates[i],
                temperature_mean=t,
                humidity_mean=h if h is not None else 50.0,
                pressure_mean=p if p is not None else 1013.25,
            ))

    if count == 0:
        logger.warning("No valid weather data points")
        return None

    averages = YearlyAverages(
        location=location,
        temperature_mean=temp_sum / count,
        humidity_mean=hum_sum / count,
        pressure_mean=press_sum / count,
        sample_count=count,
        daily_data=daily_records,
    )
    logger.info(
        f"Weather averages for {location.name}: "
        f"{averages.temperature_mean:.1f}°C, "
        f"{averages.humidity_mean:.1f}%RH, "
        f"{averages.pressure_mean:.1f}hPa "
        f"({count} days)"
    )
    return averages
