from __future__ import annotations

import torch
from torch import nn


class CNNEncoder(nn.Module):
    """CNN encoder for spatial spectrogram features."""

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, int, int] = (16, 32, 64),
        embedding_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        current = in_channels
        for ch in channels:
            layers.append(nn.Conv2d(current, ch, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout2d(dropout))
            current = ch
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(current, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        pooled = self.pool(feats).flatten(1)
        return self.proj(pooled)
