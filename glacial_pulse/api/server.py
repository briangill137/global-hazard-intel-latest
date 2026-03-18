from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Tuple

import numpy as np

from glacial_pulse.config import AudioConfig
from glacial_pulse.infer.real_time_infer import GlacialPulseInferencer
from glacial_pulse.preprocessing.audio_loader import load_audio


class GlacialPulseRequestHandler(BaseHTTPRequestHandler):
    inferencer: GlacialPulseInferencer

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/infer":
            self._send(404, {"error": "Not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send(400, {"error": "Invalid JSON"})
            return

        try:
            audio, sr = self._load_audio(payload)
        except Exception as exc:  # noqa: BLE001
            self._send(400, {"error": str(exc)})
            return

        temperature = float(payload.get("temperature", -15.0))
        result = self.inferencer.infer_audio_window(audio, sr, temperature=temperature)
        safe = {
            "fracture_prob": result["fracture_prob"],
            "time_to_fracture_sec": result["time_to_fracture_sec"],
            "confidence": result["confidence"],
            "anomaly_score": result["anomaly_score"],
            "timestamp": result["timestamp"].isoformat() + "Z",
        }
        self._send(200, safe)

    def _load_audio(self, payload: dict) -> Tuple[np.ndarray, int]:
        if "audio_path" in payload:
            path = Path(payload["audio_path"])
            return load_audio(path, target_sr=None)
        if "samples" in payload:
            samples = np.array(payload["samples"], dtype=np.float32)
            sr = int(payload.get("sample_rate", AudioConfig().sample_rate))
            return samples, sr
        raise ValueError("Provide 'audio_path' or 'samples' in request")


def run_server(args: argparse.Namespace) -> None:
    inferencer = GlacialPulseInferencer(
        model_path=args.model_path,
        autoencoder_path=args.autoencoder_path,
    )
    handler = GlacialPulseRequestHandler
    handler.inferencer = inferencer
    server = HTTPServer((args.host, args.port), handler)
    print(f"Glacial Pulse API running on http://{args.host}:{args.port}")
    server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Glacial Pulse API server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8084)
    parser.add_argument("--model-path", type=str, default="glacial_pulse/models/checkpoints/glacial_pulse.pt")
    parser.add_argument("--autoencoder-path", type=str, default="glacial_pulse/models/checkpoints/autoencoder.pt")
    return parser.parse_args()


if __name__ == "__main__":
    run_server(parse_args())
