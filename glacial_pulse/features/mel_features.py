from __future__ import annotations

import datetime
from typing import Tuple

import numpy as np
import torch

from glacial_pulse.features.seasonal import seasonal_features
from glacial_pulse.preprocessing.spectrogram import mel_spectrogram

AUX_FEATURE_DIM = 7


def log_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> torch.Tensor:
    mel = mel_spectrogram(audio, sample_rate, n_fft, hop_length, win_length, n_mels, fmin, fmax)
    return torch.log1p(mel)


def temporal_fft_features(mel_spec: torch.Tensor) -> np.ndarray:
    """Capture temporal rhythm patterns via FFT energy bands."""

    data = mel_spec.detach().cpu().numpy()
    if data.ndim == 2:
        time_series = data.mean(axis=0)
    else:
        time_series = data.reshape(-1)
    fft = np.abs(np.fft.rfft(time_series))
    if len(fft) < 4:
        return np.zeros(3, dtype=np.float32)
    third = max(1, len(fft) // 3)
    low = float(np.mean(fft[:third]))
    mid = float(np.mean(fft[third : 2 * third]))
    high = float(np.mean(fft[2 * third :]))
    scale = max(low + mid + high, 1e-6)
    return np.array([low / scale, mid / scale, high / scale], dtype=np.float32)


def low_frequency_anomaly_score(mel_spec: torch.Tensor) -> float:
    data = mel_spec.detach().cpu().numpy()
    mel_bins = data.shape[0]
    low_bins = max(1, int(mel_bins * 0.2))
    low_energy = float(np.sum(data[:low_bins, :]))
    total_energy = float(np.sum(data)) + 1e-6
    return low_energy / total_energy


def anomaly_heatmap(mel_spec: torch.Tensor) -> np.ndarray:
    data = mel_spec.detach().cpu().numpy()
    mean = np.mean(data)
    std = np.std(data) + 1e-6
    z = (data - mean) / std
    return np.clip(z, -3, 3)


def build_aux_features(
    mel_spec: torch.Tensor,
    temperature: float,
    timestamp: datetime.datetime | None = None,
) -> np.ndarray:
    """Create auxiliary feature vector (temperature + rhythm + low-freq + season)."""

    temp_norm = (temperature + 50.0) / 100.0
    temp_norm = float(np.clip(temp_norm, 0.0, 1.0))
    fft_feats = temporal_fft_features(mel_spec)
    low_score = low_frequency_anomaly_score(mel_spec)
    seasonal = seasonal_features(timestamp)
    return np.concatenate([[temp_norm], fft_feats, [low_score], seasonal]).astype(np.float32)
