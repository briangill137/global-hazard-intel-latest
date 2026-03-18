import datetime
from typing import Dict, Tuple

import numpy as np

from ai_models.features import build_feature_vector
from ai_models.predictor import EnsemblePredictor
from data_sources.api_client import fetch_open_meteo
from database.db import Database


class PredictionEngine:
    """Wrapper to run hazard predictions and persist outputs."""

    def __init__(self, db: Database, predictor: EnsemblePredictor | None = None) -> None:
        self.db = db
        self.model = predictor or EnsemblePredictor()

    def predict_for_location(self, city: str, lat: float, lon: float) -> Dict[str, float]:
        weather = fetch_open_meteo(lat, lon)
        feats, raw = build_feature_vector(weather, lat, lon, elevation=raw_elevation(weather))
        probs = self.model.predict(feats)
        # Convert to percentages
        output = {
            "flood_probability": probs["flood_probability"] * 100,
            "snowmelt_flood_risk": probs["snowmelt_flood_risk"] * 100,
            "freezing_probability": probs["freezing_probability"] * 100,
            "heatwave_probability": probs["heatwave_probability"] * 100,
            "wildfire_spread_risk": probs["wildfire_spread_risk"] * 100,
            "storm_severity_risk": probs["storm_severity_risk"] * 100,
            "confidence": probs["confidence"] * 100,
        }
        record = {
            "city": city,
            "latitude": lat,
            "longitude": lon,
            "flood": output["flood_probability"],
            "snowmelt": output["snowmelt_flood_risk"],
            "freezing": output["freezing_probability"],
            "heatwave": output["heatwave_probability"],
            "wildfire": output["wildfire_spread_risk"],
            "storm": output["storm_severity_risk"],
            "confidence": output["confidence"],
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "location": city or f"{lat},{lon}",
        }
        self.db.insert_prediction(record)
        return output, raw


def raw_elevation(weather: Dict) -> float:
    # fallback pseudo-elevation if none provided
    return float(weather.get("elevation", 250.0))
