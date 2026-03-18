from __future__ import annotations

import random
import tkinter as tk
from typing import Any
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from glacial_pulse.alerts.alert_engine import GlacialAlertEngine
from glacial_pulse.config import AudioConfig
from glacial_pulse.data.synthetic import simulate_glacial_audio, simulate_temperature
from glacial_pulse.infer.real_time_infer import GlacialPulseInferencer


class GlacialPulsePanel(ttk.Frame):
    """Research-grade visualization panel for Glacial Pulse streams."""

    def __init__(
        self,
        parent: tk.Widget,
        alert_engine: GlacialAlertEngine,
        inferencer: GlacialPulseInferencer | None = None,
        palette: dict | None = None,
    ) -> None:
        super().__init__(parent, padding=12, style="Glass.TFrame")
        self.alert_engine = alert_engine
        self.inferencer = inferencer or GlacialPulseInferencer()
        self.audio_cfg = self.inferencer.audio_cfg
        self.palette = palette or {
            "bg": "#0b1220",
            "panel": "#121a2c",
            "accent": "#4da3ff",
            "text": "#e6edf7",
            "glass": "#1e2a44",
        }
        self.running = False
        self.prob_history = []
        self.time_history = []
        self._last_alert_at = None
        self._alert_cooldown_sec = 20

        self._build_layout()

    def _build_layout(self) -> None:
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        header = ttk.Frame(self, padding=8, style="Glass.Card.TFrame")
        header.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        header.grid_columnconfigure(4, weight=1)

        ttk.Label(header, text="Glacial Pulse — Live Seismic Intelligence", style="Header.TLabel").grid(row=0, column=0, sticky="w")

        self.status_var = tk.StringVar(value="Stream idle")
        ttk.Label(header, textvariable=self.status_var, background=self.palette["panel"], foreground=self.palette["text"]).grid(row=0, column=1, padx=12)

        self.metrics_var = tk.StringVar(value="prob: -- | anomaly: -- | eta: --")
        ttk.Label(header, textvariable=self.metrics_var, background=self.palette["panel"], foreground=self.palette["text"]).grid(row=0, column=2, padx=12)

        ttk.Button(header, text="Start Stream", style="Glass.TButton", command=self.start_stream).grid(row=0, column=3, padx=6)
        ttk.Button(header, text="Stop", style="Glass.TButton", command=self.stop_stream).grid(row=0, column=4, padx=6, sticky="e")

        grid = ttk.Frame(self, padding=8, style="Glass.TFrame")
        grid.grid(row=1, column=0, sticky="nsew")
        for r in range(2):
            grid.grid_rowconfigure(r, weight=1)
        for c in range(2):
            grid.grid_columnconfigure(c, weight=1)

        self.spec_fig, self.spec_ax = self._build_figure("Spectrogram")
        self.spec_canvas = FigureCanvasTkAgg(self.spec_fig, master=grid)
        self.spec_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        self.prob_fig, self.prob_ax = self._build_figure("Fracture Probability")
        self.prob_canvas = FigureCanvasTkAgg(self.prob_fig, master=grid)
        self.prob_canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew", padx=6, pady=6)

        self.anom_fig, self.anom_ax = self._build_figure("Anomaly Heatmap")
        self.anom_canvas = FigureCanvasTkAgg(self.anom_fig, master=grid)
        self.anom_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=6, pady=6)

        alerts_frame = ttk.Frame(grid, padding=6, style="Glass.Card.TFrame")
        alerts_frame.grid(row=1, column=1, sticky="nsew", padx=6, pady=6)
        ttk.Label(alerts_frame, text="Alert Timeline", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        self.alert_list = tk.Listbox(alerts_frame, bg=self.palette["panel"], fg=self.palette["text"], highlightthickness=0, borderwidth=0, height=12)
        self.alert_list.grid(row=1, column=0, sticky="nsew", pady=6)
        alerts_frame.grid_rowconfigure(1, weight=1)

    def _build_figure(self, title: str) -> tuple[Figure, Any]:
        fig = Figure(figsize=(4.2, 3.0), dpi=100)
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor(self.palette["panel"])
        ax.set_facecolor(self.palette["glass"])
        ax.set_title(title, color=self.palette["text"], fontsize=10)
        ax.tick_params(colors=self.palette["text"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(self.palette["glass"])
        return fig, ax

    def start_stream(self) -> None:
        if self.running:
            return
        self.running = True
        self.status_var.set("Streaming...")
        self._step_stream()

    def stop_stream(self) -> None:
        self.running = False
        self.status_var.set("Stream idle")

    def _step_stream(self) -> None:
        if not self.running:
            return
        fracture = random.random() < 0.3
        audio = simulate_glacial_audio(
            self.audio_cfg.window_seconds,
            self.audio_cfg.sample_rate,
            fracture=fracture,
            seed=random.randint(0, 10000),
        )
        temp, _ = simulate_temperature(self.audio_cfg.window_seconds)
        result = self.inferencer.infer_audio_window(audio, self.audio_cfg.sample_rate, temperature=temp)
        self._update_visuals(result)
        self.after(1500, self._step_stream)

    def _update_visuals(self, result: dict) -> None:
        prob = float(result["fracture_prob"])
        anomaly = float(result["anomaly_score"])
        eta = float(result["time_to_fracture_sec"])
        confidence = float(result["confidence"])

        self.prob_history.append(prob)
        if len(self.prob_history) > 30:
            self.prob_history.pop(0)

        self.time_history.append(eta)
        if len(self.time_history) > 30:
            self.time_history.pop(0)

        self.spec_ax.clear()
        self.spec_ax.imshow(result["spectrogram"], aspect="auto", origin="lower", cmap="magma")
        self.spec_ax.set_title("Spectrogram", color=self.palette["text"], fontsize=10)

        self.prob_ax.clear()
        self.prob_ax.plot(self.prob_history, color=self.palette["accent"], linewidth=1.8)
        self.prob_ax.set_ylim(0, 1)
        self.prob_ax.set_title("Fracture Probability", color=self.palette["text"], fontsize=10)

        self.anom_ax.clear()
        self.anom_ax.imshow(result["anomaly_map"], aspect="auto", origin="lower", cmap="inferno")
        self.anom_ax.set_title("Anomaly Heatmap", color=self.palette["text"], fontsize=10)

        self.spec_canvas.draw()
        self.prob_canvas.draw()
        self.anom_canvas.draw()

        self.metrics_var.set(f"prob: {prob:.2f} | anomaly: {anomaly:.2f} | eta: {eta:.0f}s | conf: {confidence:.2f}")

        if prob > 0.8 and anomaly > 0.6:
            now = result["timestamp"]
            if not self._last_alert_at or (now - self._last_alert_at).total_seconds() > self._alert_cooldown_sec:
                alert = self.alert_engine.handle_detection(
                    location="Antarctica",
                    fracture_prob=prob,
                    anomaly_score=anomaly,
                    time_to_fracture_sec=eta,
                    confidence=confidence,
                )
                self.alert_list.insert(0, f"{alert['timestamp']} | {alert['type']} (sev {alert['severity']:.1f})")
                self._last_alert_at = now
