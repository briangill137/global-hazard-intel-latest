import datetime
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from joblib import dump, load
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from ai_models.features import FEATURE_NAMES

try:
    from xgboost import XGBClassifier
except Exception:  # noqa: BLE001
    XGBClassifier = None

MODEL_DIR = Path("models")


def _safe_load(path: Path):
    if path.exists():
        try:
            return load(path)
        except Exception:
            return None
    return None


class EnsemblePredictor:
    """
    Ensemble classifier using RF, GBDT, and XGBoost.
    Each model outputs probabilities per hazard class; results are averaged.
    """

    def __init__(self) -> None:
        self.models = {
            "flood": _safe_load(MODEL_DIR / "flood_model.pkl"),
            "snow": _safe_load(MODEL_DIR / "snow_model.pkl"),
            "fire": _safe_load(MODEL_DIR / "fire_model.pkl"),
            "storm": _safe_load(MODEL_DIR / "storm_model.pkl"),
        }
        # if any missing, build lightweight defaults
        for key, model in list(self.models.items()):
            if model is None:
                self.models[key] = self._fallback_model()

    @staticmethod
    def _fallback_model():
        return RandomForestClassifier(n_estimators=20, random_state=7).fit(
            np.random.rand(50, len(FEATURE_NAMES)),
            np.random.randint(0, 2, size=50),
        )

    def predict(self, features: np.ndarray) -> Dict[str, float]:
        probs = {}
        variance = []
        mapping = {
            "flood": "flood_probability",
            "snow": "snowmelt_flood_risk",
            "fire": "wildfire_spread_risk",
            "storm": "storm_severity_risk",
        }
        for key, model in self.models.items():
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(features)[0][1]
            else:
                p = float(model.predict(features)[0])
            probs[mapping[key]] = float(p)
            variance.append(p)

        # Derive heatwave and freezing from temperature-like dimensions if available
        # Temperature is feature 0 after normalization; approximate
        temp_norm = float(features[0, 0])
        humidity_norm = float(features[0, 1])
        heatwave = max(0.0, temp_norm * 1.2 - 0.2)  # heuristic
        freezing = max(0.0, (1 - temp_norm) * 0.9)
        probs["heatwave_probability"] = min(1.0, heatwave)
        probs["freezing_probability"] = min(1.0, freezing)

        confidence = 1.0 - float(np.var(variance))

        probs["confidence"] = max(0.0, min(1.0, confidence))
        return probs


def save_models(models: Dict[str, object]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        dump(model, MODEL_DIR / f"{name}_model.pkl")


def train_ensemble(X: np.ndarray, y_dict: Dict[str, np.ndarray]) -> Dict[str, object]:
    trained = {}
    for key, y in y_dict.items():
        rf = RandomForestClassifier(n_estimators=120, max_depth=None, random_state=3)
        gb = GradientBoostingClassifier(random_state=4)
        if XGBClassifier:
            xgb = XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                verbosity=0,
                random_state=5,
            )
        else:
            xgb = None

        # Fit models
        rf.fit(X, y)
        gb.fit(X, y)
        if xgb:
            xgb.fit(X, y)
        # Simple averaging ensemble stored as list
        trained[key] = [rf, gb, xgb] if xgb else [rf, gb]
    return trained


class AveragingWrapper:
    """Wraps a list of models to present a predict_proba interface."""

    def __init__(self, models: List[object]):
        self.models = [m for m in models if m is not None]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = []
        for m in self.models:
            if hasattr(m, "predict_proba"):
                probs.append(m.predict_proba(X)[:, 1])
            else:
                probs.append(m.predict(X))
        mean = np.mean(probs, axis=0)
        # return two-class proba for compatibility
        return np.vstack([1 - mean, mean]).T


def pack_and_save(trained: Dict[str, List[object]]) -> None:
    wrappers = {name: AveragingWrapper(models) for name, models in trained.items()}
    save_models({k: v for k, v in wrappers.items()})
