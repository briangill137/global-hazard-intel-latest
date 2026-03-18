import datetime
import threading
import time
from typing import Callable, Dict, List, Optional

from alerts.alert_manager import AlertManager
from data_sources import api_client
from database.db import Database


class MonitoringEngine:
    """
    Periodic background monitor that pulls feeds and emits hazard events.
    """

    def __init__(
        self,
        db: Database,
        alert_manager: AlertManager,
        interval_seconds: int = 60,
        on_event: Optional[Callable[[Dict], None]] = None,
    ) -> None:
        self.db = db
        self.alert_manager = alert_manager
        self.interval = interval_seconds
        self.on_event = on_event
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="HazardMonitor")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._poll_once()
            except Exception as exc:  # noqa: BLE001
                # Keep running even if one poll fails
                print(f"[Monitor] error: {exc}")
            finally:
                self._stop_event.wait(self.interval)

    def _poll_once(self) -> None:
        feeds = api_client.aggregate_sources()
        events = self._detect_hazards(feeds)
        for event in events:
            self.db.insert_hazard(event)
            self.alert_manager.create_alert(
                alert_type=event["type"],
                location=event["location"],
                severity=event["severity"],
                confidence=event["confidence"],
                message=event["details"],
            )
            if self.on_event:
                self.on_event(event)

    def _detect_hazards(self, feeds: Dict[str, List[Dict]]) -> List[Dict]:
        now = datetime.datetime.utcnow().isoformat() + "Z"
        events: List[Dict] = []

        for fire in feeds.get("fires", []):
            if fire.get("confidence", 0) > 0.6 or fire.get("brightness", 0) > 300:
                events.append(
                    {
                        "type": "Wildfire",
                        "location": fire.get("location", "Unknown"),
                        "lat": fire.get("lat"),
                        "lon": fire.get("lon"),
                        "severity": min(100, fire.get("brightness", 300) / 4),
                        "confidence": fire.get("confidence", 0.7) * 100,
                        "timestamp": now,
                        "source": "NASA FIRMS",
                        "details": "Satellite hotspot detected",
                    }
                )

        for storm in feeds.get("storms", []):
            wind = storm.get("wind", 0)
            if wind > 60:
                label = "Hurricane" if wind >= 80 else "Storm"
                events.append(
                    {
                        "type": label,
                        "location": storm.get("location", "Unknown"),
                        "lat": storm.get("lat"),
                        "lon": storm.get("lon"),
                        "severity": min(100, wind),
                        "confidence": 85.0,
                        "timestamp": now,
                        "source": "NOAA",
                        "details": f"{label} system {storm.get('name', '')} with winds {wind} kt",
                    }
                )

        for quake in feeds.get("earthquakes", []):
            if quake.get("mag", 0) >= 5.0:
                events.append(
                    {
                        "type": "Earthquake",
                        "location": quake.get("place", "Unknown"),
                        "lat": quake.get("lat"),
                        "lon": quake.get("lon"),
                        "severity": quake.get("mag", 0) * 10,
                        "confidence": 70.0,
                        "timestamp": now,
                        "source": "USGS",
                        "details": f"M{quake.get('mag')} earthquake",
                    }
                )

        # Derived meteorological risks using Open-Meteo
        meteo = api_client.fetch_open_meteo(lat=0, lon=0)
        rain = meteo.get("rainfall", 0)
        snow = meteo.get("snowfall", 0)
        temp = meteo.get("temperature", 0)
        wind = meteo.get("wind", 0)
        humidity = meteo.get("humidity", 0)

        if rain > 20 or (rain > 10 and humidity > 80):
            events.append(
                {
                    "type": "Flood Risk",
                    "location": "Global",
                    "lat": None,
                    "lon": None,
                    "severity": min(100, rain * 3),
                    "confidence": 65.0,
                    "timestamp": now,
                    "source": "Open-Meteo",
                    "details": "Heavy rainfall detected",
                    "rainfall": rain,
                    "temperature": temp,
                    "wind": wind,
                }
            )
        if snow > 5:
            events.append(
                {
                    "type": "Heavy Snow",
                    "location": "Global",
                    "lat": None,
                    "lon": None,
                    "severity": min(100, snow * 10),
                    "confidence": 60.0,
                    "timestamp": now,
                    "source": "Open-Meteo",
                    "details": "Snow accumulation risk",
                    "snowfall": snow,
                    "temperature": temp,
                    "wind": wind,
                }
            )
        if temp > 35:
            events.append(
                {
                    "type": "Heatwave",
                    "location": "Global",
                    "lat": None,
                    "lon": None,
                    "severity": min(100, (temp - 25) * 4),
                    "confidence": 70.0,
                    "timestamp": now,
                    "source": "Open-Meteo",
                    "details": "High temperature anomaly",
                    "temperature": temp,
                    "wind": wind,
                    "humidity": humidity,
                }
            )
        if wind > 25:
            events.append(
                {
                    "type": "Extreme Wind",
                    "location": "Global",
                    "lat": None,
                    "lon": None,
                    "severity": min(100, wind * 3),
                    "confidence": 60.0,
                    "timestamp": now,
                    "source": "Open-Meteo",
                    "details": "Damaging wind gust potential",
                    "temperature": temp,
                    "wind": wind,
                    "humidity": humidity,
                }
            )
        return events
