# Global Hazard Intel v2.5

Global Hazard Intel is a modular research platform for monitoring and forecasting extreme hazards. Version 2.5 introduces **Glacial Pulse**, a production-grade seismic audio pipeline for predicting ice shelf fractures before visible collapse.

## Highlights (v2.5)
- Glacial Pulse module for sub-zero seismic audio analysis
- Dual-path CNN + Transformer architecture with multi-head outputs
- Real-time streaming inference and alert integration
- Unsupervised autoencoder for anomaly scoring
- Seasonal trend learning for glacier stress cycles
- Monitoring API endpoint for external systems

## Quick Start
```bash
python main.py
```

The dashboard includes the new **Glacial Pulse** panel with live spectrograms, probability trends, anomaly heatmaps, and alert timelines.

## Glacial Pulse Usage
### Training (synthetic or real data)
```bash
python -m glacial_pulse.train.train_model --data-dir glacial_pulse/data --epochs 5
```

### Training with live FDSN seismic data (API fetch)
```bash
python -m glacial_pulse.train.train_model --fetch-fdsn --epochs 5
```

Default FDSN settings pull Antarctic data from the EarthScope dataselect service:
```text
Base URL: https://service.earthscope.org/fdsnws/dataselect/1/
Network: IU
Station: PMSA
Channel: BH?
```
Note: FDSN waveform downloads are unlabeled; the training dataset applies a low-frequency anomaly heuristic to create pseudo-labels.

### Real-time inference demo
```bash
python -m glacial_pulse.infer.real_time_infer --steps 8
```

### API server
```bash
python -m glacial_pulse.api.server --port 8084
```

Example request (JSON):
```json
{
  "audio_path": "path/to/window.wav",
  "temperature": -18.5
}
```

## Glacial Pulse Pipeline
1. **Input**: `.wav` or `.mseed` seismic audio (synthetic fallback supported)
2. **Preprocessing**: bandpass filtering, normalization, windowing
3. **Features**: log-mel spectrograms, temporal FFT rhythms, low-frequency anomaly scores
4. **Model**: CNN + Transformer fusion with heads for fracture probability, time-to-fracture, and confidence
5. **Anomaly**: autoencoder-based reconstruction error and low-frequency spike detection
6. **Alerts**: triggers when `fracture_prob > 0.8` and anomaly score is high

## Project Structure
```
glacial_pulse/
  data/
  preprocessing/
  features/
  models/
  train/
  infer/
  alerts/
  visualization/
  api/
```

## Dependencies
Core runtime:
- Python 3.10+
- `torch`
- `numpy`
- `matplotlib`
- `scikit-learn`

Optional:
- `obspy` for `.mseed` seismic files
- `obspy` is required for FDSN API downloads

## Notes
- If no datasets are available, the module automatically generates synthetic glacier stress audio for demos and training.
- Alerts are integrated into the existing Global Hazard Intel alert stream.
