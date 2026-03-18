from alerts.alert_manager import AlertManager
from database.db import Database


def build_alert_engine(db: Database, on_alert=None) -> AlertManager:
    """Factory for alert manager to keep main.py tidy."""
    return AlertManager(db=db, on_alert=on_alert)
