from __future__ import annotations

import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder for temporal seismic patterns."""

    def __init__(
        self,
        n_mels: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(n_mels, d_model)
        self.pos = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out = nn.Linear(d_model, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_mels, time)
        if x.dim() == 4:
            x = x.squeeze(1)
        x = x.transpose(1, 2)  # (batch, time, n_mels)
        x = self.proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        pooled = self.pool(x.transpose(1, 2)).squeeze(-1)
        return self.out(pooled)
