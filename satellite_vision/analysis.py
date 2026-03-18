from typing import Dict, Any, List

import numpy as np

try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None


class SatelliteAnalyzer:
    """
    Basic OpenCV-based feature spotting on satellite-like rasters.
    If OpenCV is unavailable, uses numpy heuristics to fake detections.
    """

    def analyze(self, image: Any = None) -> Dict[str, Any]:
        if cv2 and image is not None:
            return self._analyze_with_cv(image)
        return self._fallback()

    def _analyze_with_cv(self, image: Any) -> Dict[str, Any]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        wildfire_clusters = len(contours)

        # Simple gradient analysis to mimic cloud mass detection
        grad = cv2.Laplacian(gray, cv2.CV_64F)
        storm_intensity = float(np.clip(np.std(grad) / 5, 0, 100))

        smoke_mask = cv2.inRange(gray, 120, 170)
        smoke_pixels = cv2.countNonZero(smoke_mask)
        smoke_score = float(np.clip(smoke_pixels / (gray.size / 1000), 0, 100))

        return {
            "wildfire_clusters": wildfire_clusters,
            "storm_cloud_mass": storm_intensity,
            "smoke_plumes": smoke_score,
            "large_storm_systems": int(storm_intensity // 20),
        }

    def _fallback(self) -> Dict[str, Any]:
        rng = np.random.default_rng()
        return {
            "wildfire_clusters": int(rng.integers(0, 6)),
            "storm_cloud_mass": float(rng.uniform(5, 70)),
            "smoke_plumes": float(rng.uniform(0, 40)),
            "large_storm_systems": int(rng.integers(0, 3)),
        }


def demo_raster(width: int = 128, height: int = 128) -> np.ndarray:
    """Produce a synthetic raster for quick testing."""
    rng = np.random.default_rng()
    raster = rng.normal(120, 40, size=(height, width, 3)).astype(np.uint8)
    # Add a bright cluster to mimic a fire hotspot
    cv2_circle = None if cv2 is None else cv2.circle
    if cv2_circle:
        cv2_circle(raster, (width // 2, height // 2), 12, (250, 80, 30), -1)
    return raster
