"""
db.py — Stage 7: prediction logging to Postgres.

Small, self-contained database layer for the serving app. Kept separate from
serve.py so the HTTP logic and the storage logic don't tangle.

Design choices (from the Docker reference doc's DB section):
  * SQLAlchemy engine + psycopg2 driver.
  * CREATE TABLE IF NOT EXISTS at startup — safe to run every boot (created the
    first time, a no-op after). Fine for a project this size; production would use
    migrations (Alembic).
  * Parameterised inserts (values passed separately, never string-formatted into
    the SQL) — prevents SQL injection. A habit worth keeping even on toy projects.
  * with engine.begin() — opens a transaction that commits on success, rolls back
    on error. No manual commit/rollback.
  * Connection RETRY on init: depends_on guarantees the db CONTAINER started, not
    that Postgres inside it is READY to accept connections. Postgres takes a second
    or two to warm up, so we retry the first connection instead of crashing.

The DATABASE_URL is supplied via env var by compose.yaml:
    postgresql://iris:...@db:5432/predictions
where 'db' is the compose SERVICE NAME (resolves to the Postgres container).
"""

from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timezone

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

log = logging.getLogger("db")

DATABASE_URL = os.environ.get("DATABASE_URL")  # set by compose; None when running bare

_engine = None

_CREATE_TABLE = text("""
CREATE TABLE IF NOT EXISTS predictions (
    id            SERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL,
    probability   DOUBLE PRECISION NOT NULL,
    is_wake_word  BOOLEAN NOT NULL,
    threshold     DOUBLE PRECISION NOT NULL,
    model_version TEXT NOT NULL
)
""")

_INSERT = text("""
INSERT INTO predictions (created_at, probability, is_wake_word, threshold, model_version)
VALUES (:created_at, :probability, :is_wake_word, :threshold, :model_version)
""")


def init_db(retries: int = 10, delay: float = 2.0) -> bool:
    """Connect and ensure the table exists. Returns True on success, False if no
    DATABASE_URL is configured or the DB never came up.

    Retries the first connection to bridge the depends_on gap (container started
    vs. Postgres ready). Never raises — logging must never take down the service.
    """
    global _engine
    if not DATABASE_URL:
        log.info("db: no DATABASE_URL set — prediction logging disabled.")
        return False

    for attempt in range(1, retries + 1):
        try:
            engine = create_engine(DATABASE_URL, pool_pre_ping=True)
            with engine.begin() as conn:
                conn.execute(_CREATE_TABLE)
            _engine = engine
            log.info(f"db: connected and table ready (attempt {attempt}).")
            return True
        except OperationalError as e:
            log.warning(f"db: not ready (attempt {attempt}/{retries}): {type(e).__name__}. "
                  f"retrying in {delay}s...")
            time.sleep(delay)
        except Exception as e:  # noqa: BLE001 — never let DB init crash startup
            log.error(f"db: unexpected init error: {e}. logging disabled.")
            return False
    log.info("db: gave up connecting after retries — logging disabled.")
    return False


def log_prediction(probability: float, is_wake_word: bool,
                   threshold: float, model_version: str) -> bool:
    """Insert one prediction row. Returns True on success, False on any failure.

    Wrapped so a logging failure NEVER breaks the /predict response — resilience
    over consistency: a prediction can succeed and its log row silently not land,
    which is the accepted trade-off for a safety-net logger.
    """
    if _engine is None:
        return False
    try:
        with _engine.begin() as conn:
            conn.execute(_INSERT, {
                "created_at": datetime.now(timezone.utc),
                "probability": float(probability),
                "is_wake_word": bool(is_wake_word),
                "threshold": float(threshold),
                "model_version": model_version,
            })
        return True
    except Exception as e:  # noqa: BLE001 — logging must not raise into the request
        log.info(f"db: log_prediction failed (ignored): {e}")
        return False
