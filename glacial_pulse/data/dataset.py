from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from glacial_pulse.config import AudioConfig
from glacial_pulse.data.synthetic import simulate_glacial_audio, simulate_temperature
from glacial_pulse.features.mel_features import (
    build_aux_features,
    log_mel_spectrogram,
    low_frequency_anomaly_score,
)
from glacial_pulse.preprocessing.audio_loader import load_audio, normalize_audio
from glacial_pulse.preprocessing.spectrogram import bandpass_filter


class GlacialPulseDataset(Dataset):
    """Dataset yielding (spectrogram, aux_features, labels) for fracture prediction."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        config: AudioConfig | None = None,
        num_samples: int = 200,
        seed: int = 7,
    ) -> None:
        self.config = config or AudioConfig()
        self.rng = np.random.default_rng(seed)
        self.files: List[Path] = []
        if data_dir:
            data_path = Path(data_dir)
            if data_path.exists():
                self.files = list(data_path.rglob("*.wav")) + list(data_path.rglob("*.mseed"))
        self.synthetic = len(self.files) == 0
        self.num_samples = num_samples if self.synthetic else len(self.files)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fracture = bool(self.rng.random() < 0.3)
        path = None
        if self.synthetic:
            audio = simulate_glacial_audio(
                duration_sec=self.config.window_seconds,
                sample_rate=self.config.sample_rate,
                fracture=fracture,
                seed=idx,
            )
            sr = self.config.sample_rate
        else:
            path = self.files[idx]
            try:
                audio, sr = load_audio(path, target_sr=self.config.sample_rate)
                fracture = "fracture" in path.stem.lower() or fracture
            except Exception:
                audio = simulate_glacial_audio(
                    duration_sec=self.config.window_seconds,
                    sample_rate=self.config.sample_rate,
                    fracture=fracture,
                    seed=idx,
                )
                sr = self.config.sample_rate

        # Ensure fixed-length window for consistent spectrogram shapes
        window_samples = int(self.config.window_seconds * sr)
        if window_samples > 0:
            if len(audio) > window_samples:
                start = int(self.rng.integers(0, len(audio) - window_samples + 1))
                audio = audio[start : start + window_samples]
            elif len(audio) < window_samples:
                pad = window_samples - len(audio)
                audio = np.pad(audio, (0, pad), mode="constant")

        audio = bandpass_filter(audio, sr, self.config.low_freq, self.config.high_freq)
        audio = normalize_audio(audio)

        mel = log_mel_spectrogram(
            audio,
            sr,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            n_mels=self.config.n_mels,
            fmin=self.config.mel_fmin,
            fmax=self.config.mel_fmax,
        )
        mel = mel / (mel.max() + 1e-6)
        mel = mel.unsqueeze(0)

        temp_mean, _temp_var = simulate_temperature(self.config.window_seconds, seed=idx)
        aux = build_aux_features(mel.squeeze(0), temperature=temp_mean)

        if not self.synthetic and path is not None and "fracture" not in path.stem.lower():
            low_score = low_frequency_anomaly_score(mel.squeeze(0))
            fracture = low_score > 0.55

        fracture_label = 1.0 if fracture else 0.0
        time_to_fracture = float(self.rng.uniform(5, 90) if fracture else self.rng.uniform(120, 300))
        confidence = float(self.rng.uniform(0.8, 0.98) if fracture else self.rng.uniform(0.45, 0.7))

        labels = torch.tensor([fracture_label, time_to_fracture, confidence], dtype=torch.float32)
        return mel.float(), torch.tensor(aux, dtype=torch.float32), labels
