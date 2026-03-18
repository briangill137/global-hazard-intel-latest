from typing import Dict, Tuple, List

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Default expected min/max for environmental features (approximate climatological bounds)
_FEATURE_BOUNDS = {
    "temperature": (-30, 50),
    "humidity": (0, 100),
    "wind_speed": (0, 60),
    "pressure": (940, 1050),
    "rainfall_last_24h": (0, 300),  # mm
    "rainfall_last_72h": (0, 500),
    "snowfall": (0, 200),  # cm water equivalent
    "soil_moisture_estimate": (0, 1),
    "elevation": (-400, 4500),
    "latitude": (-90, 90),
    "longitude": (-180, 180),
    "rain_intensity_index": (0, 10),
    "heatwave_index": (-10, 15),
    "freeze_index": (-40, 5),
    "drought_index": (-5, 5),
}

FEATURE_NAMES: List[str] = list(_FEATURE_BOUNDS.keys())


def _make_scaler() -> MinMaxScaler:
    mins = [v[0] for v in _FEATURE_BOUNDS.values()]
    maxs = [v[1] for v in _FEATURE_BOUNDS.values()]
    X = np.stack([mins, maxs], axis=0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X)
    return scaler


_SCALER = _make_scaler()


def build_feature_vector(weather: Dict, lat: float, lon: float, elevation: float | None = None) -> Tuple[np.ndarray, Dict]:
    """
    Convert raw API/weather data into engineered + normalized feature vector.
    Returns tuple (normalized_vector, raw_feature_dict)
    """
    temp = float(weather.get("temperature", 20.0))
    humidity = float(weather.get("humidity", 60.0))
    wind = float(weather.get("wind", weather.get("wind_speed", 5.0)))
    pressure = float(weather.get("pressure", 1012.0))
    rain24 = float(weather.get("rainfall_24h", weather.get("rainfall", 0.0)))
    rain72 = float(weather.get("rainfall_72h", rain24 * 1.5))
    snow = float(weather.get("snowfall", 0.0))

    # Simple heuristics for soil moisture and elevation if missing
    soil_moisture = float(weather.get("soil_moisture", humidity / 100 * 0.4 + rain72 / 500))
    soil_moisture = max(0.0, min(1.0, soil_moisture))
    elev = float(elevation) if elevation is not None else float(weather.get("elevation", 250.0))

    # Derived indices
    rain_intensity_index = rain24 / 25.0  # 0-10 scale approx
    heatwave_index = temp - 30.0
    freeze_index = -temp if temp < 5 else 0.0
    drought_index = (pressure - 1015) / 10.0 - (humidity / 100.0) - (rain72 / 200.0)

    raw_features = {
        "temperature": temp,
        "humidity": humidity,
        "wind_speed": wind,
        "pressure": pressure,
        "rainfall_last_24h": rain24,
        "rainfall_last_72h": rain72,
        "snowfall": snow,
        "soil_moisture_estimate": soil_moisture,
        "elevation": elev,
        "latitude": lat,
        "longitude": lon,
        "rain_intensity_index": rain_intensity_index,
        "heatwave_index": heatwave_index,
        "freeze_index": freeze_index,
        "drought_index": drought_index,
    }

    vector = np.array([raw_features[name] for name in FEATURE_NAMES], dtype=float).reshape(1, -1)
    normalized = _SCALER.transform(vector)
    return normalized, raw_features
