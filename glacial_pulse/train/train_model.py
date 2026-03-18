from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from glacial_pulse.config import AudioConfig, ModelConfig, TrainConfig
from glacial_pulse.data.dataset import GlacialPulseDataset
from glacial_pulse.data.fdsn_fetch import FDSNRequest, fetch_fdsn_waveforms
from glacial_pulse.models.autoencoder import SpectrogramAutoencoder
from glacial_pulse.models.fusion_model import build_model


def train(args: argparse.Namespace) -> None:
    audio_cfg = AudioConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig(batch_size=args.batch_size, epochs=args.epochs)

    data_dir = args.data_dir
    if args.fetch_fdsn:
        req = FDSNRequest(
            base_url=args.fdsn_base_url,
            network=args.fdsn_network,
            station=args.fdsn_station,
            location=args.fdsn_location,
            channel=args.fdsn_channel,
            start=args.fdsn_start,
            end=args.fdsn_end,
            chunk_minutes=args.fdsn_chunk_minutes,
            out_dir=args.fdsn_out_dir,
            timeout_sec=args.fdsn_timeout_sec,
            retries=args.fdsn_retries,
            retry_backoff=args.fdsn_retry_backoff,
        )
        paths = fetch_fdsn_waveforms(req)
        if data_dir is None:
            data_dir = req.out_dir
        print(f"Fetched {len(paths)} waveform files from FDSN.")

    dataset = GlacialPulseDataset(data_dir=data_dir, config=audio_cfg, num_samples=args.samples)
    loader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg, n_mels=audio_cfg.n_mels).to(device)
    autoencoder = SpectrogramAutoencoder().to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    ae_opt = torch.optim.Adam(autoencoder.parameters(), lr=train_cfg.lr)

    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    model.train()
    autoencoder.train()

    for epoch in range(train_cfg.epochs):
        total_loss = 0.0
        for mel, aux, labels in loader:
            mel = mel.to(device)
            aux = aux.to(device)
            labels = labels.to(device)

            outputs = model(mel, aux)
            fracture_label = labels[:, 0]
            time_target = labels[:, 1]
            conf_target = labels[:, 2]

            loss_fracture = bce(outputs["fracture_logits"], fracture_label)
            loss_time = mse(outputs["time_to_fracture"], time_target)
            loss_conf = mse(outputs["confidence"], conf_target)

            recon = autoencoder(mel)
            loss_ae = mse(recon, mel)

            loss = loss_fracture + loss_time + 0.5 * loss_conf + train_cfg.ae_weight * loss_ae

            opt.zero_grad()
            ae_opt.zero_grad()
            loss.backward()
            opt.step()
            ae_opt.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(loader))
        print(f"Epoch {epoch + 1}/{train_cfg.epochs} - loss: {avg_loss:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "glacial_pulse.pt")
    torch.save(autoencoder.state_dict(), output_dir / "autoencoder.pt")
    print(f"Saved checkpoints to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Glacial Pulse models on seismic audio.")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory with .wav/.mseed files")
    parser.add_argument("--samples", type=int, default=200, help="Number of synthetic samples when no data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="glacial_pulse/models/checkpoints")
    parser.add_argument("--fetch-fdsn", action="store_true", help="Download FDSN waveforms before training")
    parser.add_argument("--fdsn-base-url", type=str, default="https://service.earthscope.org")
    parser.add_argument("--fdsn-network", type=str, default="IU")
    parser.add_argument("--fdsn-station", type=str, default="PMSA")
    parser.add_argument("--fdsn-location", type=str, default="*")
    parser.add_argument("--fdsn-channel", type=str, default="BH?")
    parser.add_argument("--fdsn-start", type=str, default="2025-01-01T00:00:00")
    parser.add_argument("--fdsn-end", type=str, default="2025-01-01T02:00:00")
    parser.add_argument("--fdsn-chunk-minutes", type=int, default=10)
    parser.add_argument("--fdsn-out-dir", type=str, default="glacial_pulse/data/raw")
    parser.add_argument("--fdsn-timeout-sec", type=int, default=60)
    parser.add_argument("--fdsn-retries", type=int, default=2)
    parser.add_argument("--fdsn-retry-backoff", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
