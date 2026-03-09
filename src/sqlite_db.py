from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
          user_id TEXT PRIMARY KEY
        );

        CREATE TABLE IF NOT EXISTS label_intervals (
          user_id TEXT NOT NULL,
          start_ts TEXT NOT NULL,
          end_ts TEXT NOT NULL,
          mode TEXT NOT NULL,
          PRIMARY KEY (user_id, start_ts, end_ts, mode),
          FOREIGN KEY (user_id) REFERENCES users(user_id)
        );

        CREATE TABLE IF NOT EXISTS segments (
          segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id TEXT NOT NULL,
          start_ts TEXT NOT NULL,
          end_ts TEXT NOT NULL,
          true_mode TEXT NOT NULL,
          n_points INTEGER NOT NULL,
          distance_m REAL NOT NULL,
          duration_s REAL NOT NULL,
          avg_speed_mps REAL NOT NULL,
          created_from TEXT NOT NULL,
          FOREIGN KEY (user_id) REFERENCES users(user_id)
        );

        CREATE TABLE IF NOT EXISTS segment_points (
          segment_id INTEGER NOT NULL,
          seq INTEGER NOT NULL,
          ts TEXT NOT NULL,
          lat REAL NOT NULL,
          lon REAL NOT NULL,
          PRIMARY KEY (segment_id, seq),
          FOREIGN KEY (segment_id) REFERENCES segments(segment_id)
        );

        CREATE TABLE IF NOT EXISTS segment_features (
          segment_id INTEGER PRIMARY KEY,
          user_id TEXT NOT NULL,
          true_mode TEXT NOT NULL,
          duration_s REAL NOT NULL,
          distance_m REAL NOT NULL,
          avg_speed_mps REAL NOT NULL,
          std_speed_mps REAL NOT NULL,
          p95_speed_mps REAL NOT NULL,
          max_speed_mps REAL NOT NULL,
          mean_accel_mps2 REAL NOT NULL,
          std_accel_mps2 REAL NOT NULL,
          stop_ratio REAL NOT NULL,
          mean_sampling_dt_s REAL NOT NULL,
          std_sampling_dt_s REAL NOT NULL,
          mean_abs_heading_change_deg REAL NOT NULL,
          FOREIGN KEY (segment_id) REFERENCES segments(segment_id),
          FOREIGN KEY (user_id) REFERENCES users(user_id)
        );

        CREATE INDEX IF NOT EXISTS idx_segments_user_mode ON segments(user_id, true_mode);
        CREATE INDEX IF NOT EXISTS idx_features_user_mode ON segment_features(user_id, true_mode);
        """
    )
    conn.commit()

