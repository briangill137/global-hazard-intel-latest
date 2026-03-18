from __future__ import annotations

import datetime
from typing import Callable, Dict, Optional

from alerts.alert_manager import AlertManager
from database.db import Database


class GlacialAlertEngine:
    """Create alerts and hazard events for glacial fracture risks."""

    def __init__(
        self,
        db: Database,
        alert_manager: AlertManager,
        on_event: Optional[Callable[[Dict], None]] = None,
    ) -> None:
        self.db = db
        self.alert_manager = alert_manager
        self.on_event = on_event

    def handle_detection(
        self,
        location: str,
        fracture_prob: float,
        anomaly_score: float,
        time_to_fracture_sec: float,
        confidence: float,
        source: str = "Glacial Pulse",
    ) -> Dict:
        severity = min(100.0, fracture_prob * 70.0 + anomaly_score * 30.0)
        eta_minutes = max(1, int(time_to_fracture_sec / 60))
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        event = {
            "type": "Ice Shelf Fracture Risk",
            "location": location,
            "severity": severity,
            "confidence": confidence * 100.0,
            "timestamp": timestamp,
            "source": source,
            "details": f"Potential Ice Shelf Fracture Incoming | ETA {eta_minutes} min",
        }
        self.db.insert_hazard(event)
        alert = self.alert_manager.create_alert(
            alert_type="Ice Shelf Fracture",
            location=location,
            severity=severity,
            confidence=confidence * 100.0,
            message=event["details"],
        )
        if self.on_event:
            self.on_event(event)
        return alert
