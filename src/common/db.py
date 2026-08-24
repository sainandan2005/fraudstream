from __future__ import annotations

import os

from psycopg_pool import ConnectionPool

from common.config import settings

_pool: ConnectionPool | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    alert_id    TEXT PRIMARY KEY,
    txn_id      TEXT NOT NULL,
    created_ms  BIGINT NOT NULL,
    card_id     TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    merchant_id TEXT NOT NULL,
    city        TEXT NOT NULL DEFAULT '',
    amount      NUMERIC(14, 2) NOT NULL,
    currency    TEXT NOT NULL DEFAULT 'INR',
    lat         DOUBLE PRECISION NOT NULL,
    lon         DOUBLE PRECISION NOT NULL,
    risk_score  INT NOT NULL,
    ml_score    REAL,
    rules       JSONB NOT NULL DEFAULT '[]',
    status      TEXT NOT NULL DEFAULT 'new',
    labeled_ms  BIGINT
);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts (created_ms DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_card ON alerts (card_id);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts (status);
"""

STATUSES = ("new", "acknowledged", "confirmed_fraud", "false_positive")

TRAINING_SCHEMA = """
CREATE TABLE IF NOT EXISTS training_examples (
    txn_id            TEXT PRIMARY KEY,
    ts_ms             BIGINT NOT NULL,
    card_id           TEXT NOT NULL,
    amount            DOUBLE PRECISION NOT NULL,
    window_count      INT NOT NULL,
    window_sum        DOUBLE PRECISION NOT NULL,
    user_window_count INT NOT NULL,
    new_merchant      BOOLEAN NOT NULL,
    blacklisted       BOOLEAN NOT NULL,
    implied_speed_kmh REAL,
    geo_distance_km   REAL,
    amount_z          REAL,
    amount_samples    INT,
    risk_score        INT NOT NULL,
    flagged           BOOLEAN NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_training_ts ON training_examples (ts_ms);
"""


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            os.environ.get("DATABASE_URL", settings.database_url),
            min_size=1,
            max_size=6,
            open=True,
        )
        with _pool.connection() as conn:
            conn.execute(SCHEMA)
            conn.execute(TRAINING_SCHEMA)
            conn.commit()
    return _pool
