from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field
from typing import Dict

import numpy as np


def seasonal_features(timestamp: datetime.datetime | None = None) -> np.ndarray:
    """Return seasonal sin/cos encoding for a timestamp."""

    ts = timestamp or datetime.datetime.utcnow()
    day_of_year = ts.timetuple().tm_yday
    ratio = day_of_year / 365.25
    return np.array([math.sin(2 * math.pi * ratio), math.cos(2 * math.pi * ratio)], dtype=np.float32)


@dataclass
class SeasonalBaseline:
    """Lightweight historical trend tracker for seasonal glacier stress cycles."""

    monthly_means: Dict[int, float] = field(default_factory=lambda: {m: 0.0 for m in range(1, 13)})
    monthly_counts: Dict[int, int] = field(default_factory=lambda: {m: 0 for m in range(1, 13)})

    def update(self, timestamp: datetime.datetime, value: float) -> None:
        month = timestamp.month
        count = self.monthly_counts[month]
        mean = self.monthly_means[month]
        new_mean = (mean * count + value) / max(1, count + 1)
        self.monthly_counts[month] = count + 1
        self.monthly_means[month] = new_mean

    def predict(self, timestamp: datetime.datetime) -> float:
        return float(self.monthly_means[timestamp.month])
