"""Database connection abstraction for SQLite (dev) and PostgreSQL/TimescaleDB (prod).

Auto-detects backend from DatabaseConfig.backend.
Provides a unified interface so schema.py and repository.py never import sqlite3 directly.

Placeholder style:
  SQLite    → ?
  PostgreSQL → %s
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import structlog

if TYPE_CHECKING:
    from collections.abc import Generator

    from ep2_crypto.config import DatabaseConfig

logger = structlog.get_logger(__name__)

Backend = Literal["sqlite", "timescaledb"]


class DatabaseConnection:
    """Thin wrapper that normalises SQLite and PostgreSQL connections.

    Usage (SQLite)::

        cfg = DatabaseConfig()  # backend="sqlite"
        db = DatabaseConnection(cfg)
        conn = db.get_connection()
        conn.execute(db.fmt("SELECT 1 WHERE 1 = {p}"), (1,))

    Usage (PostgreSQL)::

        cfg = DatabaseConfig(backend="timescaledb", timescaledb_url=SecretStr("postgresql://..."))
        db = DatabaseConnection(cfg)
        with db.cursor() as cur:
            cur.execute(db.fmt("SELECT 1 WHERE 1 = {p}"), (1,))
    """

    def __init__(self, config: DatabaseConfig) -> None:
        self._config = config
        self._backend: Backend = config.backend  # type: ignore[assignment]
        self._sqlite_conn: sqlite3.Connection | None = None
        self._pg_pool: Any = None  # psycopg2 pool — imported lazily

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def placeholder(self) -> str:
        """Return the correct parameter placeholder for the active backend."""
        return "?" if self._backend == "sqlite" else "%s"

    def fmt(self, sql: str) -> str:
        """Replace the portable token ``{p}`` with the correct placeholder.

        Write all SQL with ``{p}`` as the placeholder token.  Call ``fmt()``
        before executing so the correct style is substituted::

            db.fmt("SELECT * FROM ohlcv WHERE symbol = {p}")
            # SQLite  → "SELECT * FROM ohlcv WHERE symbol = ?"
            # PG      → "SELECT * FROM ohlcv WHERE symbol = %s"
        """
        return sql.replace("{p}", self.placeholder)

    def upsert_keyword(self) -> str:
        """Return the correct upsert keyword fragment for the active backend.

        SQLite    → "INSERT OR REPLACE INTO"
        PostgreSQL → "INSERT INTO"  (caller must append ON CONFLICT DO UPDATE)
        """
        if self._backend == "sqlite":
            return "INSERT OR REPLACE INTO"
        return "INSERT INTO"

    # ------------------------------------------------------------------
    # SQLite
    # ------------------------------------------------------------------

    def _get_sqlite_connection(self) -> sqlite3.Connection:
        """Return (or create) the single SQLite connection."""
        if self._sqlite_conn is None:
            db_path = self._config.sqlite_path
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._sqlite_conn = conn
            logger.debug("sqlite_connection_opened", path=str(db_path))
        return self._sqlite_conn

    # ------------------------------------------------------------------
    # PostgreSQL
    # ------------------------------------------------------------------

    def _get_pg_pool(self) -> Any:
        """Return (or create) the psycopg2 connection pool."""
        if self._pg_pool is None:
            try:
                from psycopg2 import pool as pg_pool  # type: ignore[import-untyped]
            except ImportError as exc:
                msg = (
                    "psycopg2 is required for TimescaleDB backend. "
                    "Install it with: uv add psycopg2-binary"
                )
                raise ImportError(msg) from exc

            url = self._config.timescaledb_url.get_secret_value()
            if not url:
                msg = "EP2_DB_TIMESCALEDB_URL must be set for timescaledb backend"
                raise ValueError(msg)

            self._pg_pool = pg_pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=url,
            )
            logger.info("pg_pool_created", minconn=1, maxconn=10)
        return self._pg_pool

    # ------------------------------------------------------------------
    # Unified connection access
    # ------------------------------------------------------------------

    def get_connection(self) -> sqlite3.Connection:
        """Return a raw SQLite connection.

        Only call this from schema.py during table creation or from code
        that only runs in the SQLite backend.  For generic code that must
        work on both backends use ``cursor()`` instead.

        Raises:
            RuntimeError: If called when the backend is not SQLite.
        """
        if self._backend != "sqlite":
            msg = "get_connection() is only available for the SQLite backend. Use cursor() instead."
            raise RuntimeError(msg)
        return self._get_sqlite_connection()

    @contextmanager
    def cursor(self) -> Generator[Any, None, None]:
        """Context manager that yields an execute-able cursor for either backend.

        For SQLite the cursor wraps the single persistent connection.
        For PostgreSQL a connection is borrowed from the pool and returned
        (or rolled back on error) when the context exits.

        The caller is responsible for committing by calling ``commit()``
        on the connection — or use the ``transaction()`` context manager.

        Yields:
            A DB-API 2.0 cursor.
        """
        if self._backend == "sqlite":
            conn = self._get_sqlite_connection()
            cur = conn.cursor()
            try:
                yield cur
            except Exception:
                conn.rollback()
                raise
        else:
            pool = self._get_pg_pool()
            pg_conn = pool.getconn()
            try:
                cur = pg_conn.cursor()
                yield cur
                pg_conn.commit()
            except Exception:
                pg_conn.rollback()
                raise
            finally:
                pool.putconn(pg_conn)

    @contextmanager
    def transaction(self) -> Generator[Any, None, None]:
        """Context manager that yields a cursor inside an explicit transaction.

        Commits on clean exit, rolls back on exception.

        Yields:
            A DB-API 2.0 cursor.
        """
        if self._backend == "sqlite":
            conn = self._get_sqlite_connection()
            cur = conn.cursor()
            try:
                yield cur
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        else:
            pool = self._get_pg_pool()
            pg_conn = pool.getconn()
            try:
                cur = pg_conn.cursor()
                yield cur
                pg_conn.commit()
            except Exception:
                pg_conn.rollback()
                raise
            finally:
                pool.putconn(pg_conn)

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> Any:
        """Execute a single statement and return the cursor.

        Uses ``fmt()`` automatically — write ``{p}`` in ``sql``.
        Commits after the statement for write operations.
        For read-heavy paths, use ``cursor()`` directly to avoid
        per-statement commit overhead.
        """
        formatted = self.fmt(sql)
        if self._backend == "sqlite":
            conn = self._get_sqlite_connection()
            cur = conn.execute(formatted, params)
            conn.commit()
            return cur
        else:
            pool = self._get_pg_pool()
            pg_conn = pool.getconn()
            try:
                cur = pg_conn.cursor()
                cur.execute(formatted, params)
                pg_conn.commit()
                return cur
            except Exception:
                pg_conn.rollback()
                raise
            finally:
                pool.putconn(pg_conn)

    def executemany(self, sql: str, params_seq: list[tuple[Any, ...]]) -> int:
        """Execute a statement for each param tuple and return the row count.

        Uses ``fmt()`` automatically.
        """
        formatted = self.fmt(sql)
        count = len(params_seq)
        if self._backend == "sqlite":
            conn = self._get_sqlite_connection()
            conn.executemany(formatted, params_seq)
            conn.commit()
        else:
            pool = self._get_pg_pool()
            pg_conn = pool.getconn()
            try:
                cur = pg_conn.cursor()
                cur.executemany(formatted, params_seq)
                pg_conn.commit()
            except Exception:
                pg_conn.rollback()
                raise
            finally:
                pool.putconn(pg_conn)
        return count

    def fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[Any]:
        """Execute a SELECT and return all rows.

        Uses ``fmt()`` automatically.
        """
        formatted = self.fmt(sql)
        if self._backend == "sqlite":
            conn = self._get_sqlite_connection()
            return conn.execute(formatted, params).fetchall()
        else:
            from psycopg2 import extras as pg_extras

            pool = self._get_pg_pool()
            pg_conn = pool.getconn()
            try:
                cur = pg_conn.cursor(cursor_factory=pg_extras.RealDictCursor)
                cur.execute(formatted, params)
                return cur.fetchall()
            finally:
                pool.putconn(pg_conn)

    def fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> Any | None:
        """Execute a SELECT and return one row (or None).

        Uses ``fmt()`` automatically.
        """
        formatted = self.fmt(sql)
        if self._backend == "sqlite":
            conn = self._get_sqlite_connection()
            return conn.execute(formatted, params).fetchone()
        else:
            from psycopg2 import extras as pg_extras

            pool = self._get_pg_pool()
            pg_conn = pool.getconn()
            try:
                cur = pg_conn.cursor(cursor_factory=pg_extras.RealDictCursor)
                cur.execute(formatted, params)
                return cur.fetchone()
            finally:
                pool.putconn(pg_conn)

    def close(self) -> None:
        """Close all open connections / pool."""
        if self._sqlite_conn is not None:
            self._sqlite_conn.close()
            self._sqlite_conn = None
            logger.debug("sqlite_connection_closed")
        if self._pg_pool is not None:
            self._pg_pool.closeall()
            self._pg_pool = None
            logger.debug("pg_pool_closed")
