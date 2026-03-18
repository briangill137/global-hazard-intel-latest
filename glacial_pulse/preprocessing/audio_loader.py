from __future__ import annotations

import wave
from pathlib import Path
from typing import Tuple

import numpy as np

from glacial_pulse.data.synthetic import simulate_glacial_audio


SUPPORTED_EXTENSIONS = {".wav", ".mseed"}


def _read_wav(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        dtype = np.int16 if wf.getsampwidth() == 2 else np.int32
        audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)
        audio = audio / (np.max(np.abs(audio)) + 1e-6)
        return audio, sample_rate


def _read_mseed(path: Path) -> Tuple[np.ndarray, int]:
    try:
        from obspy import read  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise ImportError("Reading .mseed requires obspy. Install obspy or provide .wav files.") from exc

    stream = read(str(path))
    trace = stream[0]
    audio = trace.data.astype(np.float32)
    sample_rate = int(trace.stats.sampling_rate)
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    return audio, sample_rate


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    new_len = int(duration * target_sr)
    x_old = np.linspace(0, duration, len(audio), endpoint=False)
    x_new = np.linspace(0, duration, new_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def load_audio(path: str | Path, target_sr: int | None = None) -> Tuple[np.ndarray, int]:
    """Load .wav or .mseed audio and optionally resample."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported audio format: {path.suffix}")

    if path.suffix.lower() == ".wav":
        audio, sr = _read_wav(path)
    else:
        audio, sr = _read_mseed(path)

    if target_sr:
        audio = _resample(audio, sr, target_sr)
        sr = target_sr
    return audio.astype(np.float32), sr


def load_or_simulate_audio(
    path: str | Path | None,
    duration_sec: float,
    sample_rate: int,
    fracture: bool = False,
) -> Tuple[np.ndarray, int]:
    """Load audio or generate a synthetic glacier stress window."""

    if path:
        try:
            return load_audio(path, target_sr=sample_rate)
        except Exception:
            pass
    audio = simulate_glacial_audio(duration_sec, sample_rate, fracture=fracture)
    return audio, sample_rate


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = audio.astype(np.float32)
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio)) + 1e-6)
    return audio / peak
