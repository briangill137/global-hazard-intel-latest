from __future__ import annotations

import argparse
import datetime
import math
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import torch

from glacial_pulse.config import AudioConfig, ModelConfig
from glacial_pulse.data.synthetic import simulate_glacial_audio, simulate_temperature
from glacial_pulse.features.mel_features import (
    anomaly_heatmap,
    build_aux_features,
    low_frequency_anomaly_score,
    log_mel_spectrogram,
)
from glacial_pulse.features.seasonal import SeasonalBaseline
from glacial_pulse.models.autoencoder import SpectrogramAutoencoder
from glacial_pulse.models.fusion_model import build_model
from glacial_pulse.preprocessing.audio_loader import normalize_audio
from glacial_pulse.preprocessing.spectrogram import bandpass_filter
from glacial_pulse.alerts.alert_engine import GlacialAlertEngine


class GlacialPulseInferencer:
    """Real-time sliding window inference for ice shelf fracture prediction."""

    def __init__(
        self,
        audio_config: AudioConfig | None = None,
        model_config: ModelConfig | None = None,
        model_path: str | Path | None = None,
        autoencoder_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self.audio_cfg = audio_config or AudioConfig()
        self.model_cfg = model_config or ModelConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = build_model(self.model_cfg, n_mels=self.audio_cfg.n_mels).to(self.device)
        self.autoencoder = SpectrogramAutoencoder().to(self.device)
        self.model_loaded = False
        self.ae_loaded = False

        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model_loaded = True
        if autoencoder_path and Path(autoencoder_path).exists():
            self.autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=self.device))
            self.ae_loaded = True

        self.model.eval()
        self.autoencoder.eval()
        self.seasonal_baseline = SeasonalBaseline()

    def infer_audio_window(
        self,
        audio: np.ndarray,
        sample_rate: int,
        temperature: float,
        timestamp: datetime.datetime | None = None,
    ) -> Dict:
        """Run inference on a single window and return prediction + diagnostics."""

        ts = timestamp or datetime.datetime.utcnow()
        audio = bandpass_filter(audio, sample_rate, self.audio_cfg.low_freq, self.audio_cfg.high_freq)
        audio = normalize_audio(audio)

        mel = log_mel_spectrogram(
            audio,
            sample_rate,
            n_fft=self.audio_cfg.n_fft,
            hop_length=self.audio_cfg.hop_length,
            win_length=self.audio_cfg.win_length,
            n_mels=self.audio_cfg.n_mels,
            fmin=self.audio_cfg.mel_fmin,
            fmax=self.audio_cfg.mel_fmax,
        )
        mel = mel / (mel.max() + 1e-6)

        aux = build_aux_features(mel, temperature=temperature, timestamp=ts)
        aux_tensor = torch.tensor(aux, dtype=torch.float32, device=self.device).unsqueeze(0)
        mel_tensor = mel.unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(mel_tensor, aux_tensor)
            fracture_prob = torch.sigmoid(outputs["fracture_logits"]).item()
            time_to_fracture = float(outputs["time_to_fracture"].item())
            confidence = float(outputs["confidence"].item())

            recon = self.autoencoder(mel_tensor)
            recon_error = torch.mean((recon - mel_tensor) ** 2).item()

        low_freq_score = low_frequency_anomaly_score(mel)
        anomaly_score = min(1.0, 0.6 * low_freq_score + 0.4 * min(1.0, recon_error * 10))

        if not self.model_loaded:
            heuristic = 1 / (1 + math.exp(-8 * (anomaly_score - 0.5)))
            fracture_prob = 0.5 * fracture_prob + 0.5 * heuristic
        if not self.ae_loaded:
            recon_error = anomaly_score

        seasonal_baseline = self.seasonal_baseline.predict(ts)
        seasonal_deviation = anomaly_score - seasonal_baseline
        self.seasonal_baseline.update(ts, anomaly_score)

        time_to_fracture = max(0.0, time_to_fracture)
        return {
            "fracture_prob": fracture_prob,
            "time_to_fracture_sec": time_to_fracture,
            "confidence": confidence,
            "anomaly_score": anomaly_score,
            "seasonal_deviation": seasonal_deviation,
            "spectrogram": mel.detach().cpu().numpy(),
            "anomaly_map": anomaly_heatmap(mel),
            "timestamp": ts,
        }

    def stream_synthetic(
        self,
        steps: int = 10,
        alert_engine: GlacialAlertEngine | None = None,
        location: str = "Antarctica",
        alert_threshold: float = 0.8,
        anomaly_threshold: float = 0.6,
        on_result: Callable[[Dict], None] | None = None,
    ) -> None:
        """Run a synthetic streaming session with optional alert dispatch."""

        for step in range(steps):
            fracture = bool(step % 4 == 0)
            audio = simulate_glacial_audio(
                self.audio_cfg.window_seconds,
                self.audio_cfg.sample_rate,
                fracture=fracture,
                seed=step,
            )
            temp, _ = simulate_temperature(self.audio_cfg.window_seconds, seed=step)
            result = self.infer_audio_window(audio, self.audio_cfg.sample_rate, temperature=temp)
            if alert_engine and result["fracture_prob"] > alert_threshold and result["anomaly_score"] > anomaly_threshold:
                alert_engine.handle_detection(
                    location=location,
                    fracture_prob=result["fracture_prob"],
                    anomaly_score=result["anomaly_score"],
                    time_to_fracture_sec=result["time_to_fracture_sec"],
                    confidence=result["confidence"],
                )
            if on_result:
                on_result(result)


def run_demo(args: argparse.Namespace) -> None:
    audio_cfg = AudioConfig()
    inferencer = GlacialPulseInferencer(
        audio_config=audio_cfg,
        model_path=args.model_path,
        autoencoder_path=args.autoencoder_path,
    )

    for step in range(args.steps):
        fracture = bool(step % 4 == 0)
        audio = simulate_glacial_audio(audio_cfg.window_seconds, audio_cfg.sample_rate, fracture=fracture, seed=step)
        temp, _ = simulate_temperature(audio_cfg.window_seconds, seed=step)
        result = inferencer.infer_audio_window(audio, audio_cfg.sample_rate, temperature=temp)
        print(
            f"[{result['timestamp']}] prob={result['fracture_prob']:.2f} "
            f"anomaly={result['anomaly_score']:.2f} eta={result['time_to_fracture_sec']:.1f}s"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Glacial Pulse real-time inference demo.")
    parser.add_argument("--model-path", type=str, default="glacial_pulse/models/checkpoints/glacial_pulse.pt")
    parser.add_argument("--autoencoder-path", type=str, default="glacial_pulse/models/checkpoints/autoencoder.pt")
    parser.add_argument("--steps", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())
