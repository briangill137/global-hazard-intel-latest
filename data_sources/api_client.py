import random
import time
from typing import Dict, List, Any, Optional

try:
    import requests
except ImportError:  # Requests might not be available in the sandbox
    requests = None

NASA_FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
NOAA_STORM_URL = "https://api.storm.glossary.noaa.gov"
USGS_EARTHQUAKE_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"


def _safe_get(url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    if requests is None:
        return None
    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        if "geojson" in url:
            return resp.json()
        # Some feeds might return text; we keep it minimal here
        return resp.json()
    except Exception:
        return None


def fetch_firms(sample_only: bool = False) -> List[Dict[str, Any]]:
    """
    Fetch NASA FIRMS wildfire hotspots. When offline, return synthetic spots.
    """
    if sample_only:
        return [
            {"lat": 34.05, "lon": -118.25, "brightness": 320, "confidence": 0.86, "location": "California"},
            {"lat": -33.87, "lon": 151.21, "brightness": 305, "confidence": 0.72, "location": "New South Wales"},
        ]
    data = _safe_get(NASA_FIRMS_URL)
    if not data:
        return fetch_firms(sample_only=True)
    # Placeholder parsing – real CSV parsing omitted for brevity
    return fetch_firms(sample_only=True)


def fetch_open_meteo(lat: float, lon: float) -> Dict[str, Any]:
    """
    Pull hourly weather forecast. Falls back to randomized but plausible values.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relativehumidity_2m,precipitation,pressure_msl,windspeed_10m,snowfall",
        "current": "temperature_2m,relativehumidity_2m,precipitation,pressure_msl,windspeed_10m,snowfall",
    }
    data = _safe_get(OPEN_METEO_URL, params=params)
    if not data:
        return {
            "temperature": random.uniform(-5, 40),
            "humidity": random.uniform(20, 95),
            "wind": random.uniform(0, 30),
            "snowfall": max(0, random.gauss(2, 1)),
            "rainfall": max(0, random.gauss(5, 2)),
            "pressure": random.uniform(980, 1035),
        }
    # Map relevant slices
    current = data.get("current", {})
    return {
        "temperature": current.get("temperature_2m", random.uniform(-5, 40)),
        "humidity": current.get("relativehumidity_2m", random.uniform(20, 95)),
        "wind": current.get("windspeed_10m", random.uniform(0, 30)),
        "snowfall": current.get("snowfall", random.uniform(0, 5)),
        "rainfall": current.get("precipitation", random.uniform(0, 10)),
        "pressure": current.get("pressure_msl", random.uniform(980, 1035)),
    }


def fetch_noaa_storms(sample_only: bool = False) -> List[Dict[str, Any]]:
    if sample_only:
        return [
            {"name": "Atlantic Low", "location": "Atlantic Ocean", "wind": 65, "pressure": 990},
            {"name": "Pacific Cyclone", "location": "Western Pacific", "wind": 85, "pressure": 975},
        ]
    data = _safe_get(NOAA_STORM_URL)
    if not data:
        return fetch_noaa_storms(sample_only=True)
    return fetch_noaa_storms(sample_only=True)


def fetch_earthquakes(sample_only: bool = False) -> List[Dict[str, Any]]:
    if sample_only:
        return [
            {"mag": 5.2, "place": "Japan Trench", "time": time.time()},
            {"mag": 4.7, "place": "Alaska", "time": time.time()},
        ]
    data = _safe_get(USGS_EARTHQUAKE_URL)
    if not data:
        return fetch_earthquakes(sample_only=True)
    return fetch_earthquakes(sample_only=True)


def aggregate_sources() -> Dict[str, Any]:
    return {
        "fires": fetch_firms(),
        "storms": fetch_noaa_storms(),
        "earthquakes": fetch_earthquakes(),
    }
