from __future__ import annotations

import torch
from torch import nn


class SpectrogramAutoencoder(nn.Module):
    """Unsupervised autoencoder for anomaly detection on spectrograms."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        if recon.shape != x.shape:
            recon = torch.nn.functional.interpolate(recon, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return recon
