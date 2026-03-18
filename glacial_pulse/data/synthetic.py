import math
from typing import Tuple

import numpy as np


_DEF_NOISE_SCALE = 0.03


def simulate_glacial_audio(
    duration_sec: float,
    sample_rate: int,
    fracture: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """Generate synthetic low-frequency glacier stress audio."""

    rng = np.random.default_rng(seed)
    n = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n, endpoint=False)

    base_noise = _DEF_NOISE_SCALE * rng.standard_normal(n)
    low_rumble = 0.08 * np.sin(2 * math.pi * 0.6 * t) + 0.04 * np.sin(2 * math.pi * 1.5 * t)
    signal = base_noise + low_rumble

    if fracture:
        burst_count = int(rng.integers(2, 5))
        for _ in range(burst_count):
            center = int(rng.integers(0, n))
            width = int(rng.integers(sample_rate // 20, sample_rate // 5))
            amplitude = float(rng.uniform(0.25, 0.8))
            idx = np.arange(n)
            envelope = amplitude * np.exp(-((idx - center) ** 2) / (2 * (width**2)))
            signal += envelope * rng.choice([-1.0, 1.0])
        chirp = 0.12 * np.sin(2 * math.pi * (0.5 + 3.0 * t / duration_sec) * t)
        signal += chirp

    peak = np.max(np.abs(signal)) + 1e-6
    signal = signal / peak
    return signal.astype(np.float32)


def simulate_temperature(duration_sec: float, seed: int | None = None) -> Tuple[float, float]:
    """Return a mean temperature and variance for a polar window."""

    rng = np.random.default_rng(seed)
    mean_temp = float(rng.uniform(-35, -5))
    variance = float(rng.uniform(0.5, 3.0))
    return mean_temp, variance
