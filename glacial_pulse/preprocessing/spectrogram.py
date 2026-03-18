from __future__ import annotations

import math
from typing import List

import numpy as np
import torch


def bandpass_filter(audio: np.ndarray, sample_rate: int, low_freq: float, high_freq: float) -> np.ndarray:
    """Simple FFT-based bandpass filter for low-frequency seismic range."""

    if low_freq <= 0 and high_freq >= sample_rate / 2:
        return audio
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sample_rate)
    mask = (freqs >= low_freq) & (freqs <= high_freq)
    spectrum[~mask] = 0
    filtered = np.fft.irfft(spectrum, n=len(audio))
    return filtered.astype(np.float32)


def segment_audio(
    audio: np.ndarray,
    sample_rate: int,
    window_seconds: float,
    hop_seconds: float,
) -> List[np.ndarray]:
    """Split audio into overlapping windows."""

    win = int(window_seconds * sample_rate)
    hop = int(hop_seconds * sample_rate)
    if win <= 0:
        return [audio]
    windows: List[np.ndarray] = []
    for start in range(0, max(1, len(audio) - win + 1), hop):
        windows.append(audio[start : start + win])
    if not windows:
        windows.append(audio)
    return windows


def stft_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> torch.Tensor:
    """Return power spectrogram (freq x time)."""

    tensor = torch.tensor(audio, dtype=torch.float32)
    window = torch.hann_window(win_length)
    spec = torch.stft(
        tensor,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    power = spec.abs().pow(2)
    return power


def _hz_to_mel(freq: float) -> float:
    return 2595.0 * math.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> torch.Tensor:
    """Build Mel filterbank matrix."""

    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz = np.array([_mel_to_hz(m) for m in mels])
    bins = np.floor((n_fft + 1) * hz / sample_rate).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        start, center, end = bins[i], bins[i + 1], bins[i + 2]
        if end <= start:
            continue
        for j in range(start, center):
            if 0 <= j < filterbank.shape[1]:
                filterbank[i, j] = (j - start) / max(center - start, 1)
        for j in range(center, end):
            if 0 <= j < filterbank.shape[1]:
                filterbank[i, j] = (end - j) / max(end - center, 1)
    return torch.tensor(filterbank, dtype=torch.float32)


def mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> torch.Tensor:
    """Compute Mel spectrogram from raw audio."""

    power = stft_spectrogram(audio, sample_rate, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mel_fb = mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax)
    mel = torch.matmul(mel_fb, power)
    return mel
