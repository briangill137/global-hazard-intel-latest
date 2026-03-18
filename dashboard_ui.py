import datetime
import queue
import random
import webbrowser
from pathlib import Path
from typing import Dict, List

import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib

# Use TkAgg for embedding charts
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from alert_engine import build_alert_engine
from data_sources.api_client import fetch_open_meteo
from map_engine import build_map, MAP_PATH
from monitor_engine import build_monitor
from prediction_engine import PredictionEngine
from satellite_vision.analysis import SatelliteAnalyzer, demo_raster
from database.db import Database
from glacial_pulse.alerts.alert_engine import GlacialAlertEngine
from glacial_pulse.visualization.dashboard import GlacialPulsePanel
from version import APP_NAME, APP_VERSION

# ----- Palette -----
COLOR_BG = "#0b1220"
COLOR_PANEL = "#121a2c"
COLOR_GLASS = "#1e2a44"
COLOR_TEXT = "#e6edf7"
COLOR_ACCENT = "#4da3ff"


class DashboardApp:
    def __init__(self, root: tk.Tk, db: Database) -> None:
        self.root = root
        self.db = db
        self.queue: queue.Queue = queue.Queue()
        self.alert_manager = build_alert_engine(db, on_alert=self._enqueue_alert)
        self.glacial_alert_engine = GlacialAlertEngine(db, self.alert_manager, on_event=self._enqueue_event)
        self.monitor = build_monitor(db, self.alert_manager, interval=60, on_event=self._enqueue_event)
        self.prediction_engine = PredictionEngine(db)
        self.analyzer = SatelliteAnalyzer()

        self.events: List[Dict] = []
        self.alerts: List[Dict] = []

        self._build_style()
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.monitor.start()
        self._bootstrap_demo_data()
        self.root.after(1000, self._process_queue)
        self.root.after(4000, self._update_charts)

    # ---------- UI construction ----------
    def _build_style(self) -> None:
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.root.configure(bg=COLOR_BG)
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure(
            ".",
            background=COLOR_BG,
            foreground=COLOR_TEXT,
            fieldbackground=COLOR_PANEL,
            bordercolor=COLOR_GLASS,
            focuscolor=COLOR_ACCENT,
        )
        style.configure("Glass.TFrame", background=COLOR_PANEL, relief="flat", borderwidth=1)
        style.configure("Glass.Card.TFrame", background=COLOR_PANEL, relief="flat", borderwidth=1)
        style.configure("Glass.TButton", background=COLOR_GLASS, foreground=COLOR_TEXT, padding=10, borderwidth=0)
        style.map(
            "Glass.TButton",
            background=[("active", COLOR_ACCENT)],
            foreground=[("active", COLOR_BG)],
        )
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 14), background=COLOR_BG, foreground=COLOR_ACCENT)
        style.configure("Header.TLabel", font=("Segoe UI Semibold", 11), background=COLOR_PANEL, foreground=COLOR_TEXT)
        style.configure("CardHeader.TLabel", font=("Segoe UI Semibold", 11), background=COLOR_GLASS, foreground=COLOR_TEXT)
        style.configure("Glass.Treeview", background=COLOR_PANEL, fieldbackground=COLOR_PANEL, foreground=COLOR_TEXT, rowheight=28, borderwidth=0)
        style.configure("Glass.Treeview.Heading", background=COLOR_GLASS, foreground="white", font=("Segoe UI Semibold", 10))
        style.configure("Treeview.Heading", background=COLOR_GLASS, foreground="white", font=("Segoe UI Semibold", 10))
        style.map("Glass.Treeview", background=[("selected", COLOR_ACCENT)], foreground=[("selected", COLOR_BG)])
        style.configure("Glass.Horizontal.TProgressbar", troughcolor=COLOR_GLASS, background=COLOR_ACCENT, bordercolor=COLOR_GLASS, lightcolor=COLOR_ACCENT, darkcolor=COLOR_GLASS, thickness=12)

    def _build_layout(self) -> None:
        self.root.geometry("1420x860")
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=3)
        self.root.grid_columnconfigure(2, weight=1)

        self._build_topbar()
        self._build_sidebar()
        self._build_view_container()
        self._build_right_alerts()
        self._build_bottom_charts()

    def _build_topbar(self) -> None:
        bar = ttk.Frame(self.root, padding=12, style="Glass.TFrame")
        bar.grid(row=0, column=0, columnspan=3, sticky="nsew", pady=(10, 6), padx=10)
        bar.grid_columnconfigure(1, weight=1)

        title = ttk.Label(bar, text=f"GLOBAL HAZARD INTELLIGENCE AI v{APP_VERSION}", style="Title.TLabel")
        title.grid(row=0, column=0, sticky="w")

        self.status_indicator = tk.Canvas(bar, width=20, height=20, bg=COLOR_PANEL, highlightthickness=0)
        self.status_indicator.grid(row=0, column=2, sticky="e")
        self._set_status_light("#2ecc71")

        self.last_update_var = tk.StringVar(value="Last update: --")
        ttk.Label(bar, textvariable=self.last_update_var, background=COLOR_BG, foreground=COLOR_TEXT).grid(row=0, column=3, sticky="e", padx=(10, 6))

        self.hazard_count_var = tk.StringVar(value="Hazards: 0")
        ttk.Label(bar, textvariable=self.hazard_count_var, style="Header.TLabel").grid(row=0, column=4, sticky="e")

    def _build_sidebar(self) -> None:
        sidebar = ttk.Frame(self.root, padding=12, style="Glass.TFrame")
        sidebar.grid(row=1, column=0, sticky="nsew", padx=10, pady=6)
        self.sidebar_frame = sidebar
        sidebar.grid_rowconfigure(len(self._modules()) + 2, weight=1)

        ttk.Label(sidebar, text="Modules", style="Header.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.nav_buttons = {}
        for idx, mod in enumerate(self._modules(), start=1):
            btn = ttk.Button(
                sidebar,
                text=mod,
                style="Glass.TButton",
                command=lambda m=mod: self._switch_view(m),
            )
            btn.grid(row=idx, column=0, sticky="ew", pady=3)
            self.nav_buttons[mod] = btn

    def _build_view_container(self) -> None:
        container = ttk.Frame(self.root, padding=8, style="Glass.TFrame")
        container.grid(row=1, column=1, sticky="nsew", pady=6)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.view_container = container

        self.views: Dict[str, ttk.Frame] = {}
        self.module_tables: Dict[str, Dict] = {}
        self._create_views()
        self._switch_view("Dashboard")

    def _build_right_alerts(self) -> None:
        right = ttk.Frame(self.root, padding=12, style="Glass.TFrame")
        right.grid(row=1, column=2, sticky="nsew", padx=10, pady=6)
        right.grid_rowconfigure(1, weight=1)
        ttk.Label(right, text="Live Alerts", style="Header.TLabel").grid(row=0, column=0, sticky="w")

        # Scrollable alert cards
        self.alert_canvas = tk.Canvas(right, bg=COLOR_PANEL, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(right, orient="vertical", command=self.alert_canvas.yview)
        self.alert_canvas.configure(yscrollcommand=scrollbar.set)
        self.alert_frame = ttk.Frame(self.alert_canvas, style="Glass.TFrame")

        self.alert_canvas.create_window((0, 0), window=self.alert_frame, anchor="nw")
        self.alert_frame.bind("<Configure>", lambda e: self.alert_canvas.configure(scrollregion=self.alert_canvas.bbox("all")))

        self.alert_canvas.grid(row=1, column=0, sticky="nsew", pady=6)
        scrollbar.grid(row=1, column=1, sticky="ns")
        ttk.Button(right, text="View Recent Alerts", style="Glass.TButton", command=self._load_recent_alerts).grid(row=2, column=0, columnspan=2, sticky="ew")

    def _build_bottom_charts(self) -> None:
        bottom = ttk.Frame(self.root, padding=8, style="Glass.TFrame")
        bottom.grid(row=2, column=0, columnspan=3, sticky="nsew", padx=10, pady=(0, 10))
        for i in range(4):
            bottom.grid_columnconfigure(i, weight=1)
        self.chart_figures: List[Figure] = []
        self.chart_axes: List = []
        self.chart_canvases: List[FigureCanvasTkAgg] = []
        titles = ["Temperature Trend", "Hazard Severity", "Wind Speed", "Storm Tracking"]
        self.chart_titles = titles
        self.chart_series = [[random.uniform(10, 30) for _ in range(10)] for _ in range(4)]
        for idx, title in enumerate(titles):
            fig = Figure(figsize=(3.4, 2.1), dpi=100)
            ax = fig.add_subplot(111)
            fig.patch.set_facecolor(COLOR_PANEL)
            ax.set_facecolor(COLOR_GLASS)
            ax.tick_params(colors=COLOR_TEXT, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(COLOR_GLASS)
            ax.plot(self.chart_series[idx], color=COLOR_ACCENT, linewidth=1.8)
            ax.set_title(title, color=COLOR_TEXT, fontsize=10)
            canvas = FigureCanvasTkAgg(fig, master=bottom)
            canvas.draw()
            canvas.get_tk_widget().grid(row=0, column=idx, sticky="nsew", padx=6, pady=4)
            self.chart_figures.append(fig)
            self.chart_axes.append(ax)
            self.chart_canvases.append(canvas)

    # ---------- Views ----------
    def _modules(self) -> List[str]:
        return [
            "Dashboard",
            "Global Map",
            "Glacial Pulse",
            "Wildfire Monitor",
            "Storm Tracker",
            "Flood Intelligence",
            "Snow Monitor",
            "Satellite Vision",
            "Prediction Engine",
            "Alerts",
            "Reports",
            "System Logs",
        ]

    def _create_views(self) -> None:
        creators = {
            "Dashboard": self._create_dashboard_view,
            "Global Map": self._create_map_view,
            "Glacial Pulse": self._create_glacial_view,
            "Wildfire Monitor": lambda name: self._create_module_table_view(name, ["Wildfire"]),
            "Storm Tracker": lambda name: self._create_module_table_view(name, ["Storm", "Hurricane", "Extreme Wind"]),
            "Flood Intelligence": lambda name: self._create_module_table_view(name, ["Flood Risk"]),
            "Snow Monitor": lambda name: self._create_module_table_view(name, ["Heavy Snow"]),
            "Satellite Vision": self._create_sat_view,
            "Prediction Engine": self._create_prediction_view,
            "Alerts": self._create_alerts_view,
            "Reports": self._create_reports_view,
            "System Logs": self._create_logs_view,
        }
        for name, fn in creators.items():
            frame = fn(name)
            frame.grid(row=0, column=0, sticky="nsew")
            self.views[name] = frame

    def _switch_view(self, name: str) -> None:
        frame = self.views.get(name)
        if not frame:
            return
        frame.tkraise()
        for btn_name, btn in self.nav_buttons.items():
            if btn_name == name:
                btn.state(["pressed"])
            else:
                btn.state(["!pressed"])

    # --- Individual view builders ---
    def _create_dashboard_view(self, *_args) -> ttk.Frame:
        frame = ttk.Frame(self.view_container, padding=8, style="Glass.TFrame")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=2)
        frame.grid_columnconfigure(1, weight=1)

        # Hazard table
        table_card = ttk.Frame(frame, padding=8, style="Glass.Card.TFrame")
        table_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        ttk.Label(table_card, text="Hazard Events", style="Header.TLabel").pack(anchor="w", pady=(0, 6))

        columns = ("type", "location", "severity", "confidence", "time", "source")
        self.hazard_tree = ttk.Treeview(
            table_card,
            columns=columns,
            show="headings",
            height=16,
            style="Glass.Treeview",
        )
        self.hazard_tree.tag_configure("odd", background="#141c30")
        self.hazard_tree.tag_configure("even", background="#121a2c")
        widths = [110, 170, 90, 100, 190, 110]
        for col, width in zip(columns, widths):
            self.hazard_tree.heading(col, text=col.title(), anchor="w")
            self.hazard_tree.column(col, width=width, anchor="w")
        self.hazard_tree.pack(fill="both", expand=True)

        # Quick stats / satellite
        right_col = ttk.Frame(frame, padding=8, style="Glass.Card.TFrame")
        right_col.grid(row=0, column=1, sticky="nsew")
        right_col.grid_rowconfigure(2, weight=1)

        ttk.Label(right_col, text="Quick Actions", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(right_col, text="Open Global Map", style="Glass.TButton", command=self._open_map_external).grid(row=1, column=0, sticky="ew", pady=4)
        ttk.Button(right_col, text="Run Satellite Snapshot", style="Glass.TButton", command=self._run_satellite_demo).grid(row=2, column=0, sticky="ew", pady=4)

        ttk.Label(right_col, text="Satellite Vision", style="Header.TLabel").grid(row=3, column=0, sticky="w", pady=(10, 4))
        self.sat_summary = tk.StringVar(value="No analysis yet.")
        ttk.Label(right_col, textvariable=self.sat_summary, wraplength=260, background=COLOR_PANEL, foreground=COLOR_TEXT).grid(row=4, column=0, sticky="w")

        return frame

    def _create_map_view(self, *_args) -> ttk.Frame:
        frame = ttk.Frame(self.view_container, padding=12, style="Glass.TFrame")
        frame.grid_columnconfigure(0, weight=1)
        ttk.Label(frame, text="Global Map", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(frame, text="Generate and open the interactive map in your browser.", background=COLOR_PANEL, foreground=COLOR_TEXT).grid(row=1, column=0, sticky="w", pady=(4, 10))
        ttk.Button(frame, text="Open Global Map", style="Glass.TButton", command=self._open_map_external).grid(row=2, column=0, sticky="w", pady=6)
        self.map_status = tk.StringVar(value="Map not generated yet.")
        ttk.Label(frame, textvariable=self.map_status, background=COLOR_PANEL, foreground=COLOR_TEXT).grid(row=3, column=0, sticky="w")
        return frame

    def _create_sat_view(self, *_args) -> ttk.Frame:
        frame = ttk.Frame(self.view_container, padding=12, style="Glass.TFrame")
        frame.grid_columnconfigure(0, weight=1)
        ttk.Label(frame, text="Satellite Vision", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Button(frame, text="Run Quick Analysis", style="Glass.TButton", command=self._run_satellite_demo).grid(row=1, column=0, sticky="w", pady=8)
        self.sat_summary_long = tk.StringVar(value="No analysis yet.")
        ttk.Label(frame, textvariable=self.sat_summary_long, wraplength=760, background=COLOR_PANEL, foreground=COLOR_TEXT).grid(row=2, column=0, sticky="w")
        return frame

    def _create_glacial_view(self, *_args) -> ttk.Frame:
        palette = {
            "bg": COLOR_BG,
            "panel": COLOR_PANEL,
            "accent": COLOR_ACCENT,
            "text": COLOR_TEXT,
            "glass": COLOR_GLASS,
        }
        panel = GlacialPulsePanel(self.view_container, alert_engine=self.glacial_alert_engine, palette=palette)
        return panel

    def _create_prediction_view(self, *_args) -> ttk.Frame:
        frame = ttk.Frame(self.view_container, padding=12, style="Glass.TFrame")
        frame.grid_columnconfigure(1, weight=1)

        form = ttk.Frame(frame, padding=10, style="Glass.Card.TFrame")
        form.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        ttk.Label(form, text="AI Hazard Prediction", style="Header.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

        labels = ["City", "Latitude", "Longitude"]
        self.city_var = tk.StringVar(value="Bangkok")
        self.lat_var = tk.StringVar(value="13.7563")
        self.lon_var = tk.StringVar(value="100.5018")
        entries = [self.city_var, self.lat_var, self.lon_var]
        for i, (lab, var) in enumerate(zip(labels, entries), start=1):
            ttk.Label(form, text=lab).grid(row=i, column=0, sticky="w", pady=3)
            ttk.Entry(form, textvariable=var).grid(row=i, column=1, sticky="ew", pady=3)
        form.grid_columnconfigure(1, weight=1)

        ttk.Button(form, text="Run AI Prediction", style="Glass.TButton", command=self._handle_prediction).grid(row=4, column=0, columnspan=2, pady=10, sticky="ew")

        result = ttk.Frame(frame, padding=10, style="Glass.Card.TFrame")
        result.grid(row=0, column=1, sticky="nsew")
        ttk.Label(result, text="Risk Scores", style="Header.TLabel").grid(row=0, column=0, sticky="w")

        self.risk_vars = {
            "Flood probability": tk.DoubleVar(value=0),
            "Snowmelt risk": tk.DoubleVar(value=0),
            "Freezing risk": tk.DoubleVar(value=0),
            "Storm probability": tk.DoubleVar(value=0),
            "Wildfire spread risk": tk.DoubleVar(value=0),
            "Heatwave probability": tk.DoubleVar(value=0),
            "Confidence": tk.DoubleVar(value=0),
        }
        self.risk_bars: Dict[str, ttk.Progressbar] = {}
        for i, (label, var) in enumerate(self.risk_vars.items(), start=1):
            ttk.Label(result, text=label).grid(row=i, column=0, sticky="w", pady=4)
            bar = ttk.Progressbar(result, variable=var, maximum=100, style="Glass.Horizontal.TProgressbar")
            bar.grid(row=i, column=1, sticky="ew", pady=4)
            self.risk_bars[label] = bar
        result.grid_columnconfigure(1, weight=1)
        return frame

    def _create_alerts_view(self, *_args) -> ttk.Frame:
        frame = ttk.Frame(self.view_container, padding=12, style="Glass.TFrame")
        ttk.Label(frame, text="Alerts Archive", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        self.alerts_listbox = tk.Listbox(frame, bg=COLOR_PANEL, fg=COLOR_TEXT, highlightthickness=0, borderwidth=0, height=22)
        self.alerts_listbox.grid(row=1, column=0, sticky="nsew", pady=6)
        frame.grid_rowconfigure(1, weight=1)
        return frame

    def _create_reports_view(self, *_args) -> ttk.Frame:
        frame = ttk.Frame(self.view_container, padding=12, style="Glass.TFrame")
        frame.grid_columnconfigure(0, weight=1)
        ttk.Label(frame, text="Reports", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        self.report_box = tk.Listbox(frame, bg=COLOR_PANEL, fg=COLOR_TEXT, highlightthickness=0, borderwidth=0, height=20)
        self.report_box.grid(row=1, column=0, sticky="nsew", pady=6)
        ttk.Button(frame, text="Generate Summary", style="Glass.TButton", command=self._refresh_reports).grid(row=2, column=0, sticky="w")
        frame.grid_rowconfigure(1, weight=1)
        return frame

    def _create_logs_view(self, *_args) -> ttk.Frame:
        frame = ttk.Frame(self.view_container, padding=12, style="Glass.TFrame")
        frame.grid_columnconfigure(0, weight=1)
        ttk.Label(frame, text="System Logs (recent hazard events)", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        self.logs_box = tk.Listbox(frame, bg=COLOR_PANEL, fg=COLOR_TEXT, highlightthickness=0, borderwidth=0, height=22)
        self.logs_box.grid(row=1, column=0, sticky="nsew", pady=6)
        ttk.Button(frame, text="Refresh Logs", style="Glass.TButton", command=self._refresh_logs).grid(row=2, column=0, sticky="w")
        frame.grid_rowconfigure(1, weight=1)
        return frame

    def _create_module_table_view(self, name: str, accepted_types: List[str]) -> ttk.Frame:
        frame = ttk.Frame(self.view_container, padding=12, style="Glass.TFrame")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        ttk.Label(frame, text=name, style="Header.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        tree = ttk.Treeview(
            frame,
            columns=("type", "location", "severity", "confidence", "time", "source"),
            show="headings",
            height=18,
            style="Glass.Treeview",
        )
        tree.tag_configure("odd", background="#141c30")
        tree.tag_configure("even", background="#121a2c")
        widths = [110, 170, 90, 100, 190, 110]
        for col, width in zip(("type", "location", "severity", "confidence", "time", "source"), widths):
            tree.heading(col, text=col.title(), anchor="w")
            tree.column(col, width=width, anchor="w")
        tree.grid(row=1, column=0, sticky="nsew")
        self.module_tables[name] = {"tree": tree, "types": accepted_types}
        return frame

    def _create_placeholder_view(self, name: str) -> ttk.Frame:
        frame = ttk.Frame(self.view_container, padding=12, style="Glass.TFrame")
        ttk.Label(frame, text=name, style="Header.TLabel").pack(anchor="w")
        ttk.Label(frame, text="Monitoring active. Detailed view coming soon.", background=COLOR_PANEL, foreground=COLOR_TEXT).pack(anchor="w", pady=6)
        return frame

    # ---------- Actions ----------
    def _open_map_external(self) -> None:
        path = build_map(self.events)
        abs_uri = Path(path).resolve().as_uri()
        webbrowser.open_new_tab(abs_uri)
        self.map_status.set(f"Opened map at {datetime.datetime.utcnow().strftime('%H:%M:%S')} UTC")

    def _set_status_light(self, color: str) -> None:
        self.status_indicator.delete("all")
        self.status_indicator.create_oval(2, 2, 18, 18, fill=color, outline="")

    def _enqueue_event(self, event: Dict) -> None:
        self.queue.put(("event", event))

    def _enqueue_alert(self, alert: Dict) -> None:
        self.queue.put(("alert", alert))

    def _process_queue(self) -> None:
        while not self.queue.empty():
            item_type, payload = self.queue.get()
            if item_type == "event":
                self._handle_event(payload)
            elif item_type == "alert":
                self._handle_alert(payload)
        self.root.after(1000, self._process_queue)

    def _handle_event(self, event: Dict) -> None:
        self.events.append(event)
        tag = "even" if len(self.events) % 2 == 0 else "odd"
        self.hazard_tree.insert(
            "",
            0,
            values=(
                event.get("type"),
                event.get("location"),
                f"{event.get('severity', 0):.1f}",
                f"{event.get('confidence', 0):.0f}%",
                event.get("timestamp"),
                event.get("source"),
            ),
            tags=(tag,),
        )
        self.hazard_count_var.set(f"Hazards: {len(self.events)}")
        self._set_status_light("#2ecc71")
        self.last_update_var.set(f"Last update: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        self._route_event_to_modules(event)
        self._update_bottom_charts_with_event(event)

    def _severity_color(self, severity: float) -> str:
        if severity < 40:
            return "#2ecc71"  # green
        if severity < 70:
            return "#f1c40f"  # yellow
        return "#e74c3c"  # red

    def _handle_alert(self, alert: Dict) -> None:
        self.alerts.append(alert)
        self._render_alert_cards()
        self.alerts_listbox.insert(0, f"{alert.get('timestamp')} | {alert.get('type')} - {alert.get('location')} (sev {alert.get('severity'):.1f})")

    def _handle_prediction(self) -> None:
        try:
            city = self.city_var.get().strip()
            lat = float(self.lat_var.get())
            lon = float(self.lon_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Latitude and longitude must be numeric.")
            return

        result, raw = self.prediction_engine.predict_for_location(city, lat, lon)
        mapping = {
            "Flood probability": result["flood_probability"],
            "Snowmelt risk": result["snowmelt_flood_risk"],
            "Freezing risk": result["freezing_probability"],
            "Storm probability": result["storm_severity_risk"],
            "Wildfire spread risk": result["wildfire_spread_risk"],
            "Heatwave probability": result["heatwave_probability"],
            "Confidence": result["confidence"],
        }
        for label, score in mapping.items():
            self.risk_vars[label].set(score)
        self.last_update_var.set(f"Prediction updated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        self._set_status_light("#f39c12")

    def _run_satellite_demo(self) -> None:
        raster = demo_raster()
        summary = self.analyzer.analyze(raster)
        text = (
            f"Wildfire clusters: {summary['wildfire_clusters']}, "
            f"Storm cloud mass: {summary['storm_cloud_mass']:.1f}, "
            f"Smoke plumes: {summary['smoke_plumes']:.1f}, "
            f"Large storm systems: {summary['large_storm_systems']}"
        )
        self.sat_summary.set(text)
        self.sat_summary_long.set(text)

    def _load_recent_alerts(self) -> None:
        records = self.db.fetch_recent("alerts", limit=20)
        self.alerts_listbox.delete(0, tk.END)
        for row in records:
            msg = f"{row['timestamp']} | {row['type']} – {row['location']} (sev {row['severity']:.1f})"
            self.alerts_listbox.insert(tk.END, msg)
        # Also refresh alerts pane cards with persistent data
        self.alerts.extend(records)
        self._render_alert_cards()

    def _render_alert_cards(self) -> None:
        for child in list(self.alert_frame.winfo_children()):
            child.destroy()
        for alert in reversed(self.alerts[-40:]):  # show recent
            card = ttk.Frame(self.alert_frame, padding=8, style="Glass.Card.TFrame")
            card.pack(fill="x", pady=4)
            color = self._severity_color(float(alert.get("severity", 0)))
            indicator = tk.Canvas(card, width=14, height=14, bg=COLOR_PANEL, highlightthickness=0)
            indicator.create_oval(2, 2, 12, 12, fill=color, outline=color)
            indicator.grid(row=0, column=0, rowspan=2, padx=(0, 8))
            ttk.Label(card, text=f"{alert.get('type')} – {alert.get('location')}", style="Header.TLabel").grid(row=0, column=1, sticky="w")
            ttk.Label(card, text=f"Severity {alert.get('severity', 0):.1f} | {alert.get('timestamp')}", background=COLOR_PANEL, foreground=COLOR_TEXT).grid(row=1, column=1, sticky="w")

    def _refresh_logs(self) -> None:
        rows = self.db.fetch_recent("hazard_events", limit=50)
        self.logs_box.delete(0, tk.END)
        for row in rows:
            msg = f"{row['timestamp']} | {row['type']} – {row['location']} (sev {row['severity']:.1f})"
            self.logs_box.insert(tk.END, msg)

    def _refresh_reports(self) -> None:
        hazards = self.db.fetch_recent("hazard_events", limit=200)
        totals = {}
        for h in hazards:
            t = h["type"]
            totals[t] = totals.get(t, 0) + 1
        self.report_box.delete(0, tk.END)
        self.report_box.insert(tk.END, f"Recent hazard count: {len(hazards)}")
        for t, c in sorted(totals.items(), key=lambda x: x[1], reverse=True):
            self.report_box.insert(tk.END, f"{t}: {c}")
    def _route_event_to_modules(self, event: Dict) -> None:
        etype = str(event.get("type", "")).lower()
        for name, cfg in self.module_tables.items():
            types = [t.lower() for t in cfg["types"]]
            if any(t.lower() in etype for t in types):
                tree = cfg["tree"]
                tag = "even" if len(tree.get_children()) % 2 == 0 else "odd"
                tree.insert(
                    "",
                    0,
                    values=(
                        event.get("type"),
                        event.get("location"),
                        f"{event.get('severity', 0):.1f}",
                        f"{event.get('confidence', 0):.0f}%",
                        event.get("timestamp"),
                        event.get("source"),
                    ),
                    tags=(tag,),
                )

    # ---------- Charts ----------
    def _update_bottom_charts_with_event(self, event: Dict) -> None:
        severity = float(event.get("severity", 0))
        wind = float(event.get("wind", severity))
        new_points = [
            event.get("temperature", random.uniform(10, 35)),
            severity,
            wind,
            event.get("confidence", random.uniform(40, 90)),
        ]
        for idx, value in enumerate(new_points):
            series = self.chart_series[idx]
            series.append(value)
            if len(series) > 30:
                series.pop(0)
        self._refresh_all_charts()

    def _update_charts(self) -> None:
        # passive drift to keep charts alive
        for idx in range(len(self.chart_series)):
            val = self.chart_series[idx][-1] + random.uniform(-2, 2)
            self.chart_series[idx].append(max(0, val))
            if len(self.chart_series[idx]) > 30:
                self.chart_series[idx].pop(0)
        self._refresh_all_charts()
        self.root.after(5000, self._update_charts)

    def _refresh_all_charts(self) -> None:
        for idx, series in enumerate(self.chart_series):
            ax = self.chart_axes[idx]
            ax.clear()
            ax.set_facecolor(COLOR_GLASS)
            ax.tick_params(colors=COLOR_TEXT, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(COLOR_GLASS)
            ax.plot(series, color=COLOR_ACCENT, linewidth=1.8)
            ax.set_title(self.chart_titles[idx], color=COLOR_TEXT, fontsize=10)
        for canvas in self.chart_canvases:
            canvas.draw()

    # ---------- Helpers ----------
    def _bootstrap_demo_data(self) -> None:
        demo = [
            {
                "type": "Wildfire",
                "location": "California",
                "lat": 36.77,
                "lon": -119.42,
                "severity": 72,
                "confidence": 88,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "source": "NASA FIRMS",
                "details": "Wildfire detected – California",
            },
            {
                "type": "Storm",
                "location": "Atlantic Ocean",
                "lat": 23.0,
                "lon": -45.0,
                "severity": 65,
                "confidence": 83,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "source": "NOAA",
                "details": "Storm formation – Atlantic Ocean",
            },
            {
                "type": "Heavy Snow",
                "location": "Canada",
                "lat": 53.0,
                "lon": -113.0,
                "severity": 55,
                "confidence": 70,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "source": "Open-Meteo",
                "details": "Heavy snowfall – Canada",
            },
        ]
        for event in demo:
            self._handle_event(event)
            self.alert_manager.create_alert(
                alert_type=event["type"],
                location=event["location"],
                severity=event["severity"],
                confidence=event["confidence"],
                message=event["details"],
            )

    def _on_close(self) -> None:
        try:
            self.monitor.stop()
        except Exception:
            pass
        try:
            self.db.close()
        except Exception:
            pass
        self.root.destroy()


def launch_app() -> None:
    root = tk.Tk()
    db = Database()
    DashboardApp(root, db)
    try:
        root.mainloop()
    finally:
        db.close()


if __name__ == "__main__":
    launch_app()
