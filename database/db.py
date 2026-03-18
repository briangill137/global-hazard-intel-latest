import sqlite3
import threading
from pathlib import Path
from typing import List, Dict, Any


class Database:
    """Lightweight SQLite helper with simple thread-safe inserts and fetches."""

    def __init__(self, db_path: str = "database/hazards.db") -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS hazard_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT,
                    location TEXT,
                    severity REAL,
                    confidence REAL,
                    timestamp TEXT,
                    source TEXT,
                    details TEXT
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    city TEXT,
                    latitude REAL,
                    longitude REAL,
                    flood REAL,
                    snowmelt REAL,
                    freezing REAL,
                    heatwave REAL,
                    wildfire REAL,
                    storm REAL,
                    confidence REAL,
                    location TEXT,
                    created_at TEXT
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT,
                    location TEXT,
                    severity REAL,
                    confidence REAL,
                    timestamp TEXT,
                    message TEXT
                )
                """
            )
            # Backfill missing columns for predictions table if they don't exist
            self._ensure_column("predictions", "confidence", "REAL")
            self._ensure_column("predictions", "location", "TEXT")

    def insert_hazard(self, event: Dict[str, Any]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO hazard_events(type, location, severity, confidence, timestamp, source, details)
                VALUES(?,?,?,?,?,?,?)
                """,
                (
                    event.get("type"),
                    event.get("location"),
                    float(event.get("severity", 0)),
                    float(event.get("confidence", 0)),
                    event.get("timestamp"),
                    event.get("source"),
                    event.get("details"),
                ),
            )

    def insert_prediction(self, record: Dict[str, Any]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO predictions(city, latitude, longitude, flood, snowmelt, freezing, heatwave, wildfire, storm, confidence, location, created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.get("city"),
                    record.get("latitude"),
                    record.get("longitude"),
                    record.get("flood"),
                    record.get("snowmelt"),
                    record.get("freezing"),
                    record.get("heatwave"),
                    record.get("wildfire"),
                    record.get("storm"),
                    record.get("confidence"),
                    record.get("location"),
                    record.get("created_at"),
                ),
            )

    def insert_alert(self, alert: Dict[str, Any]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO alerts(type, location, severity, confidence, timestamp, message)
                VALUES(?,?,?,?,?,?)
                """,
                (
                    alert.get("type"),
                    alert.get("location"),
                    float(alert.get("severity", 0)),
                    float(alert.get("confidence", 0)),
                    alert.get("timestamp"),
                    alert.get("message"),
                ),
            )

    def fetch_recent(self, table: str, limit: int = 50) -> List[Dict[str, Any]]:
        if table not in {"hazard_events", "predictions", "alerts"}:
            raise ValueError("Unsupported table.")
        with self._lock:
            cursor = self._conn.execute(
                f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (limit,)
            )
            columns = [col[0] for col in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def _ensure_column(self, table: str, column: str, coltype: str) -> None:
        cursor = self._conn.execute(f"PRAGMA table_info({table})")
        cols = [row[1] for row in cursor.fetchall()]
        if column not in cols:
            try:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
            except sqlite3.OperationalError:
                pass

    def close(self) -> None:
        with self._lock:
            self._conn.close()
