from __future__ import annotations

import torch
from torch import nn

from glacial_pulse.config import ModelConfig
from glacial_pulse.models.cnn_encoder import CNNEncoder
from glacial_pulse.models.transformer_encoder import TransformerEncoder


class GlacialPulseModel(nn.Module):
    """Dual-path CNN + Transformer fusion with multi-head outputs."""

    def __init__(
        self,
        cnn: CNNEncoder,
        transformer: TransformerEncoder,
        extra_feature_dim: int = 0,
        fusion_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cnn = cnn
        self.transformer = transformer
        self.extra_feature_dim = extra_feature_dim
        fused_dim = cnn.proj.out_features + transformer.out.out_features + extra_feature_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.fracture_head = nn.Linear(fusion_dim, 1)
        self.time_head = nn.Linear(fusion_dim, 1)
        self.conf_head = nn.Linear(fusion_dim, 1)

    def forward(self, mel_spec: torch.Tensor, extra_features: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        cnn_emb = self.cnn(mel_spec)
        trans_emb = self.transformer(mel_spec)
        if self.extra_feature_dim and extra_features is None:
            extra_features = torch.zeros((mel_spec.size(0), self.extra_feature_dim), device=mel_spec.device)
        if extra_features is None:
            fused = torch.cat([cnn_emb, trans_emb], dim=1)
        else:
            fused = torch.cat([cnn_emb, trans_emb, extra_features], dim=1)
        x = self.fusion(fused)
        return {
            "fracture_logits": self.fracture_head(x).squeeze(-1),
            "time_to_fracture": self.time_head(x).squeeze(-1),
            "confidence": torch.sigmoid(self.conf_head(x).squeeze(-1)),
        }


def build_model(config: ModelConfig, n_mels: int) -> GlacialPulseModel:
    cnn = CNNEncoder(
        in_channels=1,
        channels=config.cnn_channels,
        embedding_dim=config.cnn_embedding_dim,
        dropout=config.dropout,
    )
    transformer = TransformerEncoder(
        n_mels=n_mels,
        d_model=config.transformer_d_model,
        n_heads=config.transformer_heads,
        num_layers=config.transformer_layers,
        dropout=config.transformer_dropout,
        embedding_dim=config.transformer_embedding_dim,
    )
    return GlacialPulseModel(
        cnn=cnn,
        transformer=transformer,
        extra_feature_dim=config.extra_feature_dim,
        fusion_dim=config.fusion_dim,
        dropout=config.dropout,
    )
