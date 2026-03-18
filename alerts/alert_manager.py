import datetime
import threading
from collections import deque
from typing import Callable, Deque, Dict, List, Optional

from database.db import Database


class AlertManager:
    """Thread-safe alert dispatcher and persistence layer."""

    def __init__(self, db: Database, on_alert: Optional[Callable[[Dict], None]] = None) -> None:
        self.db = db
        self.on_alert = on_alert
        self._queue: Deque[Dict] = deque(maxlen=200)
        self._lock = threading.Lock()

    def create_alert(self, alert_type: str, location: str, severity: float, confidence: float, message: str) -> Dict:
        alert = {
            "type": alert_type,
            "location": location,
            "severity": float(severity),
            "confidence": float(confidence),
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "message": message,
        }
        with self._lock:
            self._queue.appendleft(alert)
        self.db.insert_alert(alert)
        if self.on_alert:
            self.on_alert(alert)
        return alert

    def fetch_recent(self, limit: int = 50) -> List[Dict]:
        with self._lock:
            return list(self._queue)[:limit]
