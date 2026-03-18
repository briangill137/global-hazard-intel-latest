from hazard_engine.monitor import MonitoringEngine
from database.db import Database
from alerts.alert_manager import AlertManager


def build_monitor(db: Database, alert_manager: AlertManager, interval: int = 60, on_event=None) -> MonitoringEngine:
    return MonitoringEngine(db=db, alert_manager=alert_manager, interval_seconds=interval, on_event=on_event)
