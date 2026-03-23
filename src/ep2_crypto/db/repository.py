"""Parameterized query repository for all database operations.

All SQL uses ? placeholders — never string interpolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    import sqlite3

logger = structlog.get_logger(__name__)


class Repository:
    """Database repository with parameterized queries for all tables."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    # -- OHLCV ----------------------------------------------------------------

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
        self._conn.execute(
            "INSERT OR REPLACE INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                timestamp_ms,
                symbol,
                interval,
                open_,
                high,
                low,
                close,
                volume,
                quote_volume,
                trades_count,
            ),
        )
        self._conn.commit()

    def insert_ohlcv_batch(self, rows: list[tuple[Any, ...]]) -> int:
        self._conn.executemany(
            "INSERT OR REPLACE INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def query_ohlcv(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM ohlcv WHERE symbol = ? AND interval = ? "
            "AND timestamp_ms >= ? AND timestamp_ms < ? ORDER BY timestamp_ms",
            (symbol, interval, start_ms, end_ms),
        ).fetchall()

    def query_latest_ohlcv(self, symbol: str, interval: str, limit: int = 1) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM ohlcv WHERE symbol = ? AND interval = ? "
            "ORDER BY timestamp_ms DESC LIMIT ?",
            (symbol, interval, limit),
        ).fetchall()

    # -- Order Book -----------------------------------------------------------

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
        self._conn.execute(
            "INSERT OR REPLACE INTO orderbook_snapshot VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (timestamp_ms, symbol, bid_prices, bid_sizes, ask_prices, ask_sizes, mid_price, spread),
        )
        self._conn.commit()

    def query_orderbook(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
    ) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM orderbook_snapshot WHERE symbol = ? "
            "AND timestamp_ms >= ? AND timestamp_ms < ? ORDER BY timestamp_ms",
            (symbol, start_ms, end_ms),
        ).fetchall()

    # -- Aggregated Trades ----------------------------------------------------

    def insert_trade(
        self,
        timestamp_ms: int,
        symbol: str,
        price: float,
        quantity: float,
        is_buyer_maker: bool,
        trade_id: str | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO agg_trades VALUES (?, ?, ?, ?, ?, ?)",
            (timestamp_ms, symbol, price, quantity, int(is_buyer_maker), trade_id),
        )
        self._conn.commit()

    def insert_trades_batch(self, rows: list[tuple[Any, ...]]) -> int:
        self._conn.executemany(
            "INSERT OR REPLACE INTO agg_trades VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def query_trades(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
    ) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM agg_trades WHERE symbol = ? "
            "AND timestamp_ms >= ? AND timestamp_ms < ? ORDER BY timestamp_ms",
            (symbol, start_ms, end_ms),
        ).fetchall()

    # -- Funding Rate ---------------------------------------------------------

    def insert_funding_rate(
        self,
        timestamp_ms: int,
        symbol: str,
        funding_rate: float,
        mark_price: float | None = None,
        index_price: float | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO funding_rate VALUES (?, ?, ?, ?, ?)",
            (timestamp_ms, symbol, funding_rate, mark_price, index_price),
        )
        self._conn.commit()

    def query_funding_rate(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
    ) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM funding_rate WHERE symbol = ? "
            "AND timestamp_ms >= ? AND timestamp_ms < ? ORDER BY timestamp_ms",
            (symbol, start_ms, end_ms),
        ).fetchall()

    # -- Open Interest --------------------------------------------------------

    def insert_open_interest(
        self,
        timestamp_ms: int,
        symbol: str,
        open_interest: float,
        oi_value_usd: float | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO open_interest VALUES (?, ?, ?, ?)",
            (timestamp_ms, symbol, open_interest, oi_value_usd),
        )
        self._conn.commit()

    def query_open_interest(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
    ) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM open_interest WHERE symbol = ? "
            "AND timestamp_ms >= ? AND timestamp_ms < ? ORDER BY timestamp_ms",
            (symbol, start_ms, end_ms),
        ).fetchall()

    # -- Liquidations ---------------------------------------------------------

    def insert_liquidation(
        self,
        timestamp_ms: int,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO liquidation VALUES (?, ?, ?, ?, ?)",
            (timestamp_ms, symbol, side, price, quantity),
        )
        self._conn.commit()

    def query_liquidations(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
    ) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM liquidation WHERE symbol = ? "
            "AND timestamp_ms >= ? AND timestamp_ms < ? ORDER BY timestamp_ms",
            (symbol, start_ms, end_ms),
        ).fetchall()

    # -- Cross-Market ---------------------------------------------------------

    def insert_cross_market(
        self,
        timestamp_ms: int,
        symbol: str,
        price: float,
        source: str,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO cross_market VALUES (?, ?, ?, ?)",
            (timestamp_ms, symbol, price, source),
        )
        self._conn.commit()

    def query_cross_market(
        self,
        symbol: str,
        source: str,
        start_ms: int,
        end_ms: int,
    ) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM cross_market WHERE symbol = ? AND source = ? "
            "AND timestamp_ms >= ? AND timestamp_ms < ? ORDER BY timestamp_ms",
            (symbol, source, start_ms, end_ms),
        ).fetchall()

    # -- On-Chain Whale -------------------------------------------------------

    def insert_whale_tx(
        self,
        timestamp_ms: int,
        tx_hash: str,
        value_btc: float,
        fee_rate: float | None = None,
        is_exchange_flow: bool | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO onchain_whale VALUES (?, ?, ?, ?, ?)",
            (
                timestamp_ms,
                tx_hash,
                value_btc,
                fee_rate,
                int(is_exchange_flow) if is_exchange_flow is not None else None,
            ),
        )
        self._conn.commit()

    def query_whale_txs(
        self,
        start_ms: int,
        end_ms: int,
    ) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM onchain_whale WHERE timestamp_ms >= ? AND timestamp_ms < ? "
            "ORDER BY timestamp_ms",
            (start_ms, end_ms),
        ).fetchall()

    # -- Regime Labels --------------------------------------------------------

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
        self._conn.execute(
            "INSERT OR REPLACE INTO regime_label VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
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
        self._conn.commit()

    def query_regimes(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
    ) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM regime_label WHERE symbol = ? "
            "AND timestamp_ms >= ? AND timestamp_ms < ? ORDER BY timestamp_ms",
            (symbol, start_ms, end_ms),
        ).fetchall()

    # -- Predictions ----------------------------------------------------------

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
        self._conn.execute(
            "INSERT OR REPLACE INTO prediction VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
        self._conn.commit()

    def query_predictions(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
    ) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM prediction WHERE symbol = ? "
            "AND timestamp_ms >= ? AND timestamp_ms < ? ORDER BY timestamp_ms",
            (symbol, start_ms, end_ms),
        ).fetchall()

    # -- Feature Snapshots ----------------------------------------------------

    def insert_feature_snapshot(
        self,
        timestamp_ms: int,
        symbol: str,
        features_json: str,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO feature_snapshot VALUES (?, ?, ?)",
            (timestamp_ms, symbol, features_json),
        )
        self._conn.commit()

    def query_feature_snapshots(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
    ) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM feature_snapshot WHERE symbol = ? "
            "AND timestamp_ms >= ? AND timestamp_ms < ? ORDER BY timestamp_ms",
            (symbol, start_ms, end_ms),
        ).fetchall()

    # -- Utility --------------------------------------------------------------

    def count_rows(self, table: str) -> int:
        """Count rows in a table. Table name is validated against known tables."""
        allowed = {
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
        if table not in allowed:
            msg = f"Unknown table: {table}"
            raise ValueError(msg)
        # Table name is validated against allowlist — safe to interpolate
        return self._conn.execute(
            f"SELECT count(*) FROM {table}"  # noqa: S608
        ).fetchone()[0]
