from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class AudioConfig:
    """Audio preprocessing configuration for sub-zero seismic streams."""

    sample_rate: int = 100
    low_freq: float = 0.5
    high_freq: float = 12.0
    window_seconds: float = 10.0
    hop_seconds: float = 5.0
    n_fft: int = 256
    win_length: int = 256
    hop_length: int = 64
    n_mels: int = 64
    mel_fmin: float = 0.5
    mel_fmax: float = 25.0


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration for dual-path CNN + Transformer fusion."""

    cnn_channels: Tuple[int, int, int] = (16, 32, 64)
    cnn_embedding_dim: int = 128
    transformer_d_model: int = 64
    transformer_heads: int = 4
    transformer_layers: int = 2
    transformer_dropout: float = 0.1
    transformer_embedding_dim: int = 128
    fusion_dim: int = 128
    extra_feature_dim: int = 7
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainConfig:
    """Training defaults for Glacial Pulse models."""

    batch_size: int = 8
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    ae_weight: float = 0.2
