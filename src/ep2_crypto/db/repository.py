"""Parameterized query repository for all database operations.

All SQL uses the ``{p}`` portable placeholder token — never string interpolation.
DatabaseConnection.fmt() substitutes ``?`` (SQLite) or ``%s`` (PostgreSQL) at
runtime so the same query strings work transparently on both backends.

Upsert strategy:
  SQLite    → INSERT OR REPLACE INTO ...
  PostgreSQL → INSERT INTO ... ON CONFLICT (...) DO UPDATE SET ...

Each table has its own upsert suffix defined at the top of its section.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from ep2_crypto.db.connection import DatabaseConnection

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# ON CONFLICT clauses for PostgreSQL upserts (one per table)
# ---------------------------------------------------------------------------

_OHLCV_CONFLICT = (
    "ON CONFLICT (timestamp_ms, symbol, interval) DO UPDATE SET "
    "open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, close=EXCLUDED.close, "
    "volume=EXCLUDED.volume, quote_volume=EXCLUDED.quote_volume, trades_count=EXCLUDED.trades_count"
)

_ORDERBOOK_CONFLICT = (
    "ON CONFLICT (timestamp_ms, symbol) DO UPDATE SET "
    "bid_prices=EXCLUDED.bid_prices, bid_sizes=EXCLUDED.bid_sizes, "
    "ask_prices=EXCLUDED.ask_prices, ask_sizes=EXCLUDED.ask_sizes, "
    "mid_price=EXCLUDED.mid_price, spread=EXCLUDED.spread"
)

_TRADES_CONFLICT = (
    "ON CONFLICT (timestamp_ms, symbol, trade_id) DO UPDATE SET "
    "price=EXCLUDED.price, quantity=EXCLUDED.quantity, is_buyer_maker=EXCLUDED.is_buyer_maker"
)

_FUNDING_CONFLICT = (
    "ON CONFLICT (timestamp_ms, symbol) DO UPDATE SET "
    "funding_rate=EXCLUDED.funding_rate, mark_price=EXCLUDED.mark_price, "
    "index_price=EXCLUDED.index_price"
)

_OI_CONFLICT = (
    "ON CONFLICT (timestamp_ms, symbol) DO UPDATE SET "
    "open_interest=EXCLUDED.open_interest, oi_value_usd=EXCLUDED.oi_value_usd"
)

_LIQUIDATION_CONFLICT = (
    "ON CONFLICT (timestamp_ms, symbol, side, price) DO UPDATE SET "
    "quantity=EXCLUDED.quantity"
)

_CROSS_MARKET_CONFLICT = (
    "ON CONFLICT (timestamp_ms, symbol, source) DO UPDATE SET "
    "price=EXCLUDED.price"
)

_ONCHAIN_CONFLICT = (
    "ON CONFLICT (timestamp_ms, tx_hash) DO UPDATE SET "
    "value_btc=EXCLUDED.value_btc, fee_rate=EXCLUDED.fee_rate, "
    "is_exchange_flow=EXCLUDED.is_exchange_flow"
)

_REGIME_CONFLICT = (
    "ON CONFLICT (timestamp_ms, symbol) DO UPDATE SET "
    "regime=EXCLUDED.regime, hmm_state=EXCLUDED.hmm_state, hmm_prob=EXCLUDED.hmm_prob, "
    "bocpd_run_length=EXCLUDED.bocpd_run_length, garch_vol=EXCLUDED.garch_vol, "
    "efficiency_ratio=EXCLUDED.efficiency_ratio"
)

_PREDICTION_CONFLICT = (
    "ON CONFLICT (timestamp_ms, symbol) DO UPDATE SET "
    "direction=EXCLUDED.direction, confidence=EXCLUDED.confidence, "
    "calibrated_prob_up=EXCLUDED.calibrated_prob_up, "
    "calibrated_prob_down=EXCLUDED.calibrated_prob_down, "
    "position_size=EXCLUDED.position_size, regime=EXCLUDED.regime, "
    "model_version=EXCLUDED.model_version"
)

_FEATURE_SNAPSHOT_CONFLICT = (
    "ON CONFLICT (timestamp_ms, symbol) DO UPDATE SET "
    "features_json=EXCLUDED.features_json"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KNOWN_TABLES = frozenset(
    {
        "ohlcv",
        "orderbook_snapshot",
        "agg_trades",
        "funding_rate",
        "open_interest",
        "liquidation",
        "cross_market",
        "onchain_whale",
        "regime_label",
        "prediction",
        "feature_snapshot",
    }
)


def _upsert_sql(
    db: DatabaseConnection,
    table: str,
    columns: str,
    placeholders: str,
    conflict_clause: str,
) -> str:
    """Build a backend-correct upsert statement.

    Args:
        db: Active DatabaseConnection (provides backend + fmt).
        table: Table name.
        columns: Comma-separated column list string (e.g. ``"ts, symbol"``).
        placeholders: Comma-separated ``{p}`` tokens (e.g. ``"{p}, {p}"``).
        conflict_clause: ``ON CONFLICT ... DO UPDATE SET ...`` clause used for
            PostgreSQL. Ignored on SQLite (INSERT OR REPLACE handles it).

    Returns:
        Formatted SQL string with the correct placeholder style.
    """
    if db.backend == "sqlite":
        # table, columns, placeholders are all hardcoded constants — not user input.
        sql = f"INSERT OR REPLACE INTO {table} ({columns}) VALUES ({placeholders})"  # noqa: S608
    else:
        sql = (
            f"INSERT INTO {table} ({columns}) VALUES ({placeholders}) {conflict_clause}"  # noqa: S608
        )
    return db.fmt(sql)


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class Repository:
    """Database repository with parameterized queries for all tables.

    Works transparently with SQLite and PostgreSQL/TimescaleDB backends via
    the ``DatabaseConnection`` abstraction.
    """

    def __init__(self, db: DatabaseConnection) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------

    _OHLCV_COLS = "timestamp_ms, symbol, interval, open, high, low, close, volume, quote_volume, trades_count"
    _OHLCV_PH = ", ".join(["{p}"] * 10)

    def insert_ohlcv(
        self,
        timestamp_ms: int,
        symbol: str,
        interval: str,
        open_: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        quote_volume: float | None = None,
        trades_count: int | None = None,
    ) -> None:
        sql = _upsert_sql(self._db, "ohlcv", self._OHLCV_COLS, self._OHLCV_PH, _OHLCV_CONFLICT)
        self._db.execute(
            sql,
            (timestamp_ms, symbol, interval, open_, high, low, close, volume, quote_volume, trades_count),
        )

    def insert_ohlcv_batch(self, rows: list[tuple[Any, ...]]) -> int:
        sql = _upsert_sql(self._db, "ohlcv", self._OHLCV_COLS, self._OHLCV_PH, _OHLCV_CONFLICT)
        return self._db.executemany(sql, rows)

    def query_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[Any]:
        return self._db.fetchall(
            "SELECT * FROM ohlcv WHERE symbol = {p} AND interval = {p} "
            "AND timestamp_ms >= {p} AND timestamp_ms < {p} ORDER BY timestamp_ms",
            (symbol, interval, start_ms, end_ms),
        )

    def query_latest_ohlcv(self, symbol: str, interval: str, limit: int = 1) -> list[Any]:
        return self._db.fetchall(
            "SELECT * FROM ohlcv WHERE symbol = {p} AND interval = {p} "
            "ORDER BY timestamp_ms DESC LIMIT {p}",
            (symbol, interval, limit),
        )

    # ------------------------------------------------------------------
    # Order Book
    # ------------------------------------------------------------------

    _OB_COLS = "timestamp_ms, symbol, bid_prices, bid_sizes, ask_prices, ask_sizes, mid_price, spread"
    _OB_PH = ", ".join(["{p}"] * 8)

    def insert_orderbook(
        self,
        timestamp_ms: int,
        symbol: str,
        bid_prices: str,
        bid_sizes: str,
        ask_prices: str,
        ask_sizes: str,
        mid_price: float,
        spread: float,
    ) -> None:
        sql = _upsert_sql(
            self._db, "orderbook_snapshot", self._OB_COLS, self._OB_PH, _ORDERBOOK_CONFLICT
        )
        self._db.execute(
            sql,
            (timestamp_ms, symbol, bid_prices, bid_sizes, ask_prices, ask_sizes, mid_price, spread),
        )

    def query_orderbook(self, symbol: str, start_ms: int, end_ms: int) -> list[Any]:
        return self._db.fetchall(
            "SELECT * FROM orderbook_snapshot WHERE symbol = {p} "
            "AND timestamp_ms >= {p} AND timestamp_ms < {p} ORDER BY timestamp_ms",
            (symbol, start_ms, end_ms),
        )

    # ------------------------------------------------------------------
    # Aggregated Trades
    # ------------------------------------------------------------------

    _TRADE_COLS = "timestamp_ms, symbol, price, quantity, is_buyer_maker, trade_id"
    _TRADE_PH = ", ".join(["{p}"] * 6)

    def insert_trade(
        self,
        timestamp_ms: int,
        symbol: str,
        price: float,
        quantity: float,
        is_buyer_maker: bool,
        trade_id: str | None = None,
    ) -> None:
        sql = _upsert_sql(
            self._db, "agg_trades", self._TRADE_COLS, self._TRADE_PH, _TRADES_CONFLICT
        )
        self._db.execute(
            sql,
            (timestamp_ms, symbol, price, quantity, int(is_buyer_maker), trade_id),
        )

    def insert_trades_batch(self, rows: list[tuple[Any, ...]]) -> int:
        sql = _upsert_sql(
            self._db, "agg_trades", self._TRADE_COLS, self._TRADE_PH, _TRADES_CONFLICT
        )
        return self._db.executemany(sql, rows)

    def query_trades(self, symbol: str, start_ms: int, end_ms: int) -> list[Any]:
        return self._db.fetchall(
            "SELECT * FROM agg_trades WHERE symbol = {p} "
            "AND timestamp_ms >= {p} AND timestamp_ms < {p} ORDER BY timestamp_ms",
            (symbol, start_ms, end_ms),
        )

    # ------------------------------------------------------------------
    # Funding Rate
    # ------------------------------------------------------------------

    _FUND_COLS = "timestamp_ms, symbol, funding_rate, mark_price, index_price"
    _FUND_PH = ", ".join(["{p}"] * 5)

    def insert_funding_rate(
        self,
        timestamp_ms: int,
        symbol: str,
        funding_rate: float,
        mark_price: float | None = None,
        index_price: float | None = None,
    ) -> None:
        sql = _upsert_sql(
            self._db, "funding_rate", self._FUND_COLS, self._FUND_PH, _FUNDING_CONFLICT
        )
        self._db.execute(sql, (timestamp_ms, symbol, funding_rate, mark_price, index_price))

    def insert_funding_rate_batch(self, rows: list[tuple[Any, ...]]) -> int:
        sql = _upsert_sql(
            self._db, "funding_rate", self._FUND_COLS, self._FUND_PH, _FUNDING_CONFLICT
        )
        return self._db.executemany(sql, rows)

    def query_funding_rate(self, symbol: str, start_ms: int, end_ms: int) -> list[Any]:
        return self._db.fetchall(
            "SELECT * FROM funding_rate WHERE symbol = {p} "
            "AND timestamp_ms >= {p} AND timestamp_ms < {p} ORDER BY timestamp_ms",
            (symbol, start_ms, end_ms),
        )

    # ------------------------------------------------------------------
    # Open Interest
    # ------------------------------------------------------------------

    _OI_COLS = "timestamp_ms, symbol, open_interest, oi_value_usd"
    _OI_PH = ", ".join(["{p}"] * 4)

    def insert_open_interest(
        self,
        timestamp_ms: int,
        symbol: str,
        open_interest: float,
        oi_value_usd: float | None = None,
    ) -> None:
        sql = _upsert_sql(
            self._db, "open_interest", self._OI_COLS, self._OI_PH, _OI_CONFLICT
        )
        self._db.execute(sql, (timestamp_ms, symbol, open_interest, oi_value_usd))

    def insert_open_interest_batch(self, rows: list[tuple[Any, ...]]) -> int:
        sql = _upsert_sql(
            self._db, "open_interest", self._OI_COLS, self._OI_PH, _OI_CONFLICT
        )
        return self._db.executemany(sql, rows)

    def query_open_interest(self, symbol: str, start_ms: int, end_ms: int) -> list[Any]:
        return self._db.fetchall(
            "SELECT * FROM open_interest WHERE symbol = {p} "
            "AND timestamp_ms >= {p} AND timestamp_ms < {p} ORDER BY timestamp_ms",
            (symbol, start_ms, end_ms),
        )

    # ------------------------------------------------------------------
    # Liquidations
    # ------------------------------------------------------------------

    _LIQ_COLS = "timestamp_ms, symbol, side, price, quantity"
    _LIQ_PH = ", ".join(["{p}"] * 5)

    def insert_liquidation(
        self,
        timestamp_ms: int,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
    ) -> None:
        sql = _upsert_sql(
            self._db, "liquidation", self._LIQ_COLS, self._LIQ_PH, _LIQUIDATION_CONFLICT
        )
        self._db.execute(sql, (timestamp_ms, symbol, side, price, quantity))

    def query_liquidations(self, symbol: str, start_ms: int, end_ms: int) -> list[Any]:
        return self._db.fetchall(
            "SELECT * FROM liquidation WHERE symbol = {p} "
            "AND timestamp_ms >= {p} AND timestamp_ms < {p} ORDER BY timestamp_ms",
            (symbol, start_ms, end_ms),
        )

    # ------------------------------------------------------------------
    # Cross-Market
    # ------------------------------------------------------------------

    _CM_COLS = "timestamp_ms, symbol, price, source"
    _CM_PH = ", ".join(["{p}"] * 4)

    def insert_cross_market(
        self,
        timestamp_ms: int,
        symbol: str,
        price: float,
        source: str,
    ) -> None:
        sql = _upsert_sql(
            self._db, "cross_market", self._CM_COLS, self._CM_PH, _CROSS_MARKET_CONFLICT
        )
        self._db.execute(sql, (timestamp_ms, symbol, price, source))

    def insert_cross_market_batch(self, rows: list[tuple[Any, ...]]) -> int:
        sql = _upsert_sql(
            self._db, "cross_market", self._CM_COLS, self._CM_PH, _CROSS_MARKET_CONFLICT
        )
        return self._db.executemany(sql, rows)

    def query_cross_market(
        self, symbol: str, source: str, start_ms: int, end_ms: int
    ) -> list[Any]:
        return self._db.fetchall(
            "SELECT * FROM cross_market WHERE symbol = {p} AND source = {p} "
            "AND timestamp_ms >= {p} AND timestamp_ms < {p} ORDER BY timestamp_ms",
            (symbol, source, start_ms, end_ms),
        )

    # ------------------------------------------------------------------
    # On-Chain Whale
    # ------------------------------------------------------------------

    _ONCHAIN_COLS = "timestamp_ms, tx_hash, value_btc, fee_rate, is_exchange_flow"
    _ONCHAIN_PH = ", ".join(["{p}"] * 5)

    def insert_whale_tx(
        self,
        timestamp_ms: int,
        tx_hash: str,
        value_btc: float,
        fee_rate: float | None = None,
        is_exchange_flow: bool | None = None,
    ) -> None:
        sql = _upsert_sql(
            self._db, "onchain_whale", self._ONCHAIN_COLS, self._ONCHAIN_PH, _ONCHAIN_CONFLICT
        )
        self._db.execute(
            sql,
            (
                timestamp_ms,
                tx_hash,
                value_btc,
                fee_rate,
                int(is_exchange_flow) if is_exchange_flow is not None else None,
            ),
        )

    def query_whale_txs(self, start_ms: int, end_ms: int) -> list[Any]:
        return self._db.fetchall(
            "SELECT * FROM onchain_whale WHERE timestamp_ms >= {p} AND timestamp_ms < {p} "
            "ORDER BY timestamp_ms",
            (start_ms, end_ms),
        )

    # ------------------------------------------------------------------
    # Regime Labels
    # ------------------------------------------------------------------

    _REGIME_COLS = (
        "timestamp_ms, symbol, regime, hmm_state, hmm_prob, "
        "bocpd_run_length, garch_vol, efficiency_ratio"
    )
    _REGIME_PH = ", ".join(["{p}"] * 8)

    def insert_regime(
        self,
        timestamp_ms: int,
        symbol: str,
        regime: str,
        hmm_state: int | None = None,
        hmm_prob: float | None = None,
        bocpd_run_length: float | None = None,
        garch_vol: float | None = None,
        efficiency_ratio: float | None = None,
    ) -> None:
        sql = _upsert_sql(
            self._db, "regime_label", self._REGIME_COLS, self._REGIME_PH, _REGIME_CONFLICT
        )
        self._db.execute(
            sql,
            (
                timestamp_ms,
                symbol,
                regime,
                hmm_state,
                hmm_prob,
                bocpd_run_length,
                garch_vol,
                efficiency_ratio,
            ),
        )

    def query_regimes(self, symbol: str, start_ms: int, end_ms: int) -> list[Any]:
        return self._db.fetchall(
            "SELECT * FROM regime_label WHERE symbol = {p} "
            "AND timestamp_ms >= {p} AND timestamp_ms < {p} ORDER BY timestamp_ms",
            (symbol, start_ms, end_ms),
        )

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    _PRED_COLS = (
        "timestamp_ms, symbol, direction, confidence, calibrated_prob_up, "
        "calibrated_prob_down, position_size, regime, model_version"
    )
    _PRED_PH = ", ".join(["{p}"] * 9)

    def insert_prediction(
        self,
        timestamp_ms: int,
        symbol: str,
        direction: str,
        confidence: float,
        calibrated_prob_up: float | None = None,
        calibrated_prob_down: float | None = None,
        position_size: float | None = None,
        regime: str | None = None,
        model_version: str | None = None,
    ) -> None:
        sql = _upsert_sql(
            self._db, "prediction", self._PRED_COLS, self._PRED_PH, _PREDICTION_CONFLICT
        )
        self._db.execute(
            sql,
            (
                timestamp_ms,
                symbol,
                direction,
                confidence,
                calibrated_prob_up,
                calibrated_prob_down,
                position_size,
                regime,
                model_version,
            ),
        )

    def query_predictions(self, symbol: str, start_ms: int, end_ms: int) -> list[Any]:
        return self._db.fetchall(
            "SELECT * FROM prediction WHERE symbol = {p} "
            "AND timestamp_ms >= {p} AND timestamp_ms < {p} ORDER BY timestamp_ms",
            (symbol, start_ms, end_ms),
        )

    # ------------------------------------------------------------------
    # Feature Snapshots
    # ------------------------------------------------------------------

    _FS_COLS = "timestamp_ms, symbol, features_json"
    _FS_PH = ", ".join(["{p}"] * 3)

    def insert_feature_snapshot(
        self,
        timestamp_ms: int,
        symbol: str,
        features_json: str,
    ) -> None:
        sql = _upsert_sql(
            self._db,
            "feature_snapshot",
            self._FS_COLS,
            self._FS_PH,
            _FEATURE_SNAPSHOT_CONFLICT,
        )
        self._db.execute(sql, (timestamp_ms, symbol, features_json))

    def query_feature_snapshots(self, symbol: str, start_ms: int, end_ms: int) -> list[Any]:
        return self._db.fetchall(
            "SELECT * FROM feature_snapshot WHERE symbol = {p} "
            "AND timestamp_ms >= {p} AND timestamp_ms < {p} ORDER BY timestamp_ms",
            (symbol, start_ms, end_ms),
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_rows(self, table: str) -> int:
        """Return the row count for a table.

        The table name is validated against an explicit allowlist before use to
        prevent SQL injection (table names cannot be parameterized in SQL).
        """
        if table not in _KNOWN_TABLES:
            msg = f"Unknown table: {table!r}"
            raise ValueError(msg)
        # Table name is validated against the allowlist above — safe to interpolate.
        row = self._db.fetchone(f"SELECT count(*) FROM {table}")  # noqa: S608
        return int(row[0]) if row else 0
