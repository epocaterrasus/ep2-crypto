"""Unified database connection for SQLite (dev) and PostgreSQL/TimescaleDB (prod).

Provides a DBConnection wrapper with a consistent interface regardless of backend:
  - Translates ? placeholders to %s for PostgreSQL
  - Translates INSERT OR REPLACE INTO ... to INSERT INTO ... ON CONFLICT DO NOTHING
  - Wraps psycopg2 cursor management so callers use conn.execute() like sqlite3

Usage:
    conn = get_connection(os.environ.get("EP2_DB_URL", "data/history.db"))
    conn.execute("INSERT OR REPLACE INTO ohlcv VALUES (?, ?, ?)", (1, 2, 3))
    conn.commit()
    conn.close()
"""

from __future__ import annotations

import sqlite3
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def _is_postgres_url(url: str) -> bool:
    return url.startswith("postgresql://") or url.startswith("postgres://")


class _PGCursor:
    """Thin wrapper around a psycopg2 cursor to expose fetchone/fetchall like sqlite3."""

    def __init__(self, cur: Any) -> None:
        self._cur = cur

    def fetchone(self) -> Any:
        return self._cur.fetchone()

    def fetchall(self) -> list[Any]:
        return self._cur.fetchall()

    def __iter__(self) -> Any:
        return iter(self._cur)


class DBConnection:
    """Duck-typed connection wrapper normalising SQLite and PostgreSQL APIs.

    Both backends expose:
        execute(sql, params=None) -> cursor-like object with .fetchone() / .fetchall()
        executemany(sql, params_list) -> None
        commit() -> None
        close() -> None

    SQL dialect translation (only when is_postgres is True):
        ?                     ->  %s
        INSERT OR REPLACE INTO  ->  INSERT INTO ... ON CONFLICT DO NOTHING
    """

    def __init__(self, url: str) -> None:
        self._is_pg = _is_postgres_url(url)
        self._url = url
        if self._is_pg:
            try:
                import psycopg2  # type: ignore[import-untyped]

                self._raw: Any = psycopg2.connect(url)
                self._raw.autocommit = False
                logger.info("db_connected", backend="postgresql", url=_redact(url))
            except ImportError:
                msg = "psycopg2-binary is required for PostgreSQL. Run: uv sync"
                raise ImportError(msg) from None
        else:
            from ep2_crypto.db.schema import PRAGMA_SETTINGS

            self._raw = sqlite3.connect(url)
            self._raw.row_factory = sqlite3.Row
            for pragma in PRAGMA_SETTINGS:
                self._raw.execute(pragma)
            logger.info("db_connected", backend="sqlite", path=url)

    @property
    def is_postgres(self) -> bool:
        return self._is_pg

    @property
    def raw(self) -> Any:
        """Direct access to the underlying connection (use sparingly)."""
        return self._raw

    # ------------------------------------------------------------------
    # SQL dialect translation
    # ------------------------------------------------------------------

    def _adapt(self, sql: str) -> str:
        """Translate SQLite SQL to PostgreSQL SQL."""
        if not self._is_pg:
            return sql
        sql = sql.replace("?", "%s")
        if "INSERT OR REPLACE INTO" in sql:
            sql = sql.replace("INSERT OR REPLACE INTO", "INSERT INTO")
            sql = sql.rstrip() + " ON CONFLICT DO NOTHING"
        return sql

    # ------------------------------------------------------------------
    # Unified execute / executemany / commit / close
    # ------------------------------------------------------------------

    def execute(self, sql: str, params: Any = None) -> Any:
        """Execute a query and return a cursor with fetchone/fetchall."""
        sql = self._adapt(sql)
        if self._is_pg:
            cur = self._raw.cursor()
            cur.execute(sql, params)
            return _PGCursor(cur)
        return self._raw.execute(sql, params or ())

    def executemany(self, sql: str, params_list: list[Any]) -> None:
        """Execute a query with many parameter sets."""
        if not params_list:
            return
        sql = self._adapt(sql)
        if self._is_pg:
            cur = self._raw.cursor()
            cur.executemany(sql, params_list)
        else:
            self._raw.executemany(sql, params_list)

    def commit(self) -> None:
        self._raw.commit()

    def close(self) -> None:
        self._raw.close()


def get_connection(url: str) -> DBConnection:
    """Create a DBConnection from a SQLite path or PostgreSQL DSN."""
    return DBConnection(url)


def _redact(url: str) -> str:
    """Redact password from a PostgreSQL DSN for logging."""
    if "@" in url:
        prefix, rest = url.rsplit("@", 1)
        if ":" in prefix:
            scheme_user, _ = prefix.rsplit(":", 1)
            return f"{scheme_user}:***@{rest}"
    return url
