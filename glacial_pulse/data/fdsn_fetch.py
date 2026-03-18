from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List
import time


try:
    from obspy import UTCDateTime  # type: ignore
    from obspy.clients.fdsn import Client  # type: ignore
    _OBSPY_AVAILABLE = True
except Exception:  # noqa: BLE001
    UTCDateTime = None  # type: ignore[assignment]
    Client = None  # type: ignore[assignment]
    _OBSPY_AVAILABLE = False


@dataclass
class FDSNRequest:
    base_url: str = "https://service.earthscope.org"
    network: str = "IU"
    station: str = "PMSA"
    location: str = "*"
    channel: str = "BH?"
    start: str = "2025-01-01T00:00:00"
    end: str = "2025-01-01T02:00:00"
    chunk_minutes: int = 10
    out_dir: str | Path = "glacial_pulse/data/raw"
    timeout_sec: int = 60
    retries: int = 2
    retry_backoff: float = 2.0


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "x", value)


def _require_obspy() -> None:
    if not _OBSPY_AVAILABLE:
        raise ImportError("ObsPy is required for FDSN downloads. Install with: pip install obspy")


def _safe_ts(utc) -> str:
    return utc.strftime("%Y%m%dT%H%M%S")


def fetch_fdsn_waveforms(request: FDSNRequest) -> List[Path]:
    """Download waveform windows from an FDSN dataselect service."""

    _require_obspy()
    client = Client(request.base_url, timeout=request.timeout_sec)
    start = UTCDateTime(request.start)
    end = UTCDateTime(request.end)
    if end <= start:
        raise ValueError("End time must be after start time")

    out_dir = Path(request.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    net = _safe_id(request.network)
    sta = _safe_id(request.station)
    loc = _safe_id(request.location)
    cha = _safe_id(request.channel)

    saved: List[Path] = []
    cursor = start
    step = max(1, int(request.chunk_minutes)) * 60

    while cursor < end:
        next_cursor = min(end, cursor + step)
        stream = None
        for attempt in range(request.retries + 1):
            try:
                stream = client.get_waveforms(
                    network=request.network,
                    station=request.station,
                    location=request.location,
                    channel=request.channel,
                    starttime=cursor,
                    endtime=next_cursor,
                )
                break
            except Exception as exc:  # noqa: BLE001
                if attempt >= request.retries:
                    print(f"[FDSN] Skipping {cursor} - {next_cursor}: {exc}")
                else:
                    wait = request.retry_backoff ** attempt
                    print(f"[FDSN] Retry {attempt + 1} for {cursor} - {next_cursor} in {wait:.1f}s ({exc})")
                    time.sleep(wait)
        if stream is None:
            cursor = next_cursor
            continue

        if len(stream) == 0:
            cursor = next_cursor
            continue

        stream.merge(method=1, fill_value="interpolate")
        filename = f"{net}.{sta}.{loc}.{cha}_{_safe_ts(cursor)}_{_safe_ts(next_cursor)}.mseed"
        path = out_dir / filename
        stream.write(str(path), format="MSEED")
        saved.append(path)
        cursor = next_cursor

    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch seismic data from an FDSN service.")
    parser.add_argument("--base-url", type=str, default="https://service.earthscope.org")
    parser.add_argument("--network", type=str, default="IU")
    parser.add_argument("--station", type=str, default="PMSA")
    parser.add_argument("--location", type=str, default="*")
    parser.add_argument("--channel", type=str, default="BH?")
    parser.add_argument("--start", type=str, default="2025-01-01T00:00:00")
    parser.add_argument("--end", type=str, default="2025-01-01T02:00:00")
    parser.add_argument("--chunk-minutes", type=int, default=10)
    parser.add_argument("--out-dir", type=str, default="glacial_pulse/data/raw")
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    req = FDSNRequest(
        base_url=args.base_url,
        network=args.network,
        station=args.station,
        location=args.location,
        channel=args.channel,
        start=args.start,
        end=args.end,
        chunk_minutes=args.chunk_minutes,
        out_dir=args.out_dir,
        timeout_sec=args.timeout_sec,
        retries=args.retries,
        retry_backoff=args.retry_backoff,
    )
    paths = fetch_fdsn_waveforms(req)
    print(f"Downloaded {len(paths)} files into {req.out_dir}")


if __name__ == "__main__":
    main()
