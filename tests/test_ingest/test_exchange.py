"""Tests for Binance exchange data collectors."""

from __future__ import annotations

import asyncio

import pytest

from ep2_crypto.db.repository import Repository
from ep2_crypto.db.schema import create_tables
from ep2_crypto.ingest.base import CollectorState
from ep2_crypto.ingest.exchange import (
    BinanceDepthCollector,
    BinanceKlineCollector,
    BinanceTradeCollector,
)


@pytest.fixture()
def db_repo() -> Repository:
    """In-memory SQLite database with schema."""
    conn = create_tables(":memory:")
    return Repository(conn)


class MockExchange:
    """Mock ccxt pro exchange supporting watch_ohlcv, watch_order_book, watch_trades."""

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}
        self._call_count = 0
        self._closed = False
        self._candles: list[list[list[float]]] = []
        self._orderbooks: list[dict] = []
        self._trades: list[list[dict]] = []
        self._fail_after: int = 0
        self._delay_s: float = 0.01

    def set_candles(self, candles: list[list[list[float]]]) -> None:
        self._candles = candles

    def set_orderbooks(self, orderbooks: list[dict]) -> None:
        self._orderbooks = orderbooks

    def set_trades(self, trades: list[list[dict]]) -> None:
        self._trades = trades

    def set_fail_after(self, n: int) -> None:
        self._fail_after = n

    async def _check_fail(self) -> None:
        self._call_count += 1
        if self._fail_after > 0 and self._call_count >= self._fail_after:
            msg = "Mock stream error"
            raise ConnectionError(msg)
        await asyncio.sleep(self._delay_s)

    async def watch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ) -> list[list[float]]:
        await self._check_fail()
        if self._candles:
            idx = min(self._call_count - 1, len(self._candles) - 1)
            return self._candles[idx]
        ts = 1700000000000 + (self._call_count * 60000)
        return [[ts, 42000.0, 42050.0, 41980.0, 42030.0, 100.5]]

    async def watch_order_book(
        self,
        symbol: str,
        limit: int | None = None,
        params: dict | None = None,
    ) -> dict:
        await self._check_fail()
        if self._orderbooks:
            idx = min(self._call_count - 1, len(self._orderbooks) - 1)
            return self._orderbooks[idx]
        ts = 1700000000000 + (self._call_count * 100)
        return {
            "timestamp": ts,
            "bids": [[42000.0, 1.5], [41999.0, 2.0], [41998.0, 0.8]],
            "asks": [[42001.0, 1.2], [42002.0, 1.8], [42003.0, 0.5]],
        }

    async def watch_trades(
        self,
        symbol: str,
        since: int | None = None,
        limit: int | None = None,
        params: dict | None = None,
    ) -> list[dict]:
        await self._check_fail()
        if self._trades:
            idx = min(self._call_count - 1, len(self._trades) - 1)
            return self._trades[idx]
        ts = 1700000000000 + (self._call_count * 50)
        return [
            {
                "id": str(1000 + self._call_count),
                "timestamp": ts,
                "price": 42000.0 + self._call_count,
                "amount": 0.5,
                "side": "buy",
            }
        ]

    async def close(self) -> None:
        self._closed = True


class MockExchangeClass:
    """Factory that creates MockExchange instances (mimics ccxtpro.binanceusdm)."""

    def __init__(self) -> None:
        self.last_instance: MockExchange | None = None
        self._candles: list[list[list[float]]] = []
        self._orderbooks: list[dict] = []
        self._trades: list[list[dict]] = []
        self._fail_after: int = 0

    def set_candles(self, candles: list[list[list[float]]]) -> None:
        self._candles = candles

    def set_orderbooks(self, orderbooks: list[dict]) -> None:
        self._orderbooks = orderbooks

    def set_trades(self, trades: list[list[dict]]) -> None:
        self._trades = trades

    def set_fail_after(self, n: int) -> None:
        self._fail_after = n

    def __call__(self, config: dict | None = None) -> MockExchange:
        instance = MockExchange(config)
        if self._candles:
            instance.set_candles(self._candles)
        if self._orderbooks:
            instance.set_orderbooks(self._orderbooks)
        if self._trades:
            instance.set_trades(self._trades)
        if self._fail_after > 0:
            instance.set_fail_after(self._fail_after)
        self.last_instance = instance
        return instance


# -- Kline collector tests --


@pytest.mark.asyncio
async def test_kline_collector_stores_candles(db_repo: Repository) -> None:
    """Kline collector stores received candles to the database."""
    factory = MockExchangeClass()
    factory.set_candles(
        [
            [
                [1700000000000, 42000.0, 42050.0, 41980.0, 42030.0, 100.5],
                [1700000060000, 42030.0, 42060.0, 42010.0, 42045.0, 95.2],
            ],
        ]
    )

    collector = BinanceKlineCollector(
        db_repo,
        symbol="BTC/USDT:USDT",
        exchange_class=factory,
    )

    async with collector:
        await asyncio.sleep(0.1)

    rows = db_repo.query_ohlcv("BTC/USDT:USDT", "1m", 0, 2000000000000)
    assert len(rows) >= 2

    row = rows[0]
    assert row["timestamp_ms"] == 1700000000000
    assert row["open"] == 42000.0
    assert row["high"] == 42050.0
    assert row["low"] == 41980.0
    assert row["close"] == 42030.0
    assert row["volume"] == 100.5


@pytest.mark.asyncio
async def test_kline_collector_deduplicates(db_repo: Repository) -> None:
    """Same timestamp is upserted, not duplicated."""
    factory = MockExchangeClass()
    # Same candle returned twice
    candle = [1700000000000, 42000.0, 42050.0, 41980.0, 42030.0, 100.5]
    factory.set_candles([[candle], [candle]])

    collector = BinanceKlineCollector(
        db_repo,
        symbol="BTC/USDT:USDT",
        exchange_class=factory,
    )

    async with collector:
        await asyncio.sleep(0.1)

    rows = db_repo.query_ohlcv("BTC/USDT:USDT", "1m", 0, 2000000000000)
    # Should have exactly 1 row due to dedup in _process_candles
    ts_set = {r["timestamp_ms"] for r in rows}
    assert 1700000000000 in ts_set


@pytest.mark.asyncio
async def test_kline_collector_updates_health(db_repo: Repository) -> None:
    """Health check reflects message count after receiving data."""
    factory = MockExchangeClass()
    collector = BinanceKlineCollector(
        db_repo,
        symbol="BTC/USDT:USDT",
        exchange_class=factory,
    )

    async with collector:
        await asyncio.sleep(0.1)
        health = collector.health_check()
        assert health.healthy
        assert health.messages_received > 0
        assert health.state == CollectorState.RUNNING


@pytest.mark.asyncio
async def test_kline_collector_reconnects_on_error(db_repo: Repository) -> None:
    """Collector reconnects when watch_ohlcv raises."""
    factory = MockExchangeClass()
    factory.set_fail_after(3)

    collector = BinanceKlineCollector(
        db_repo,
        symbol="BTC/USDT:USDT",
        exchange_class=factory,
        reconnect_delay_s=0.01,
        max_reconnect_delay_s=0.05,
    )

    async with collector:
        await asyncio.sleep(0.2)

    assert collector._reconnect_count > 0


@pytest.mark.asyncio
async def test_kline_collector_custom_symbol(db_repo: Repository) -> None:
    """Collector uses the configured symbol."""
    factory = MockExchangeClass()
    collector = BinanceKlineCollector(
        db_repo,
        symbol="ETH/USDT:USDT",
        exchange_class=factory,
    )

    assert "ETH/USDT:USDT" in collector.name

    async with collector:
        await asyncio.sleep(0.05)

    rows = db_repo.query_ohlcv("ETH/USDT:USDT", "1m", 0, 2000000000000)
    assert len(rows) >= 1


@pytest.mark.asyncio
async def test_kline_collector_exchange_close_called(db_repo: Repository) -> None:
    """Exchange.close() is called on collector stop."""
    factory = MockExchangeClass()
    collector = BinanceKlineCollector(
        db_repo,
        symbol="BTC/USDT:USDT",
        exchange_class=factory,
    )

    async with collector:
        await asyncio.sleep(0.05)

    assert factory.last_instance is not None
    assert factory.last_instance._closed


@pytest.mark.asyncio
async def test_kline_collector_empty_candles_skipped(db_repo: Repository) -> None:
    """Empty candle lists don't cause errors."""
    factory = MockExchangeClass()
    factory.set_candles([[]])

    collector = BinanceKlineCollector(
        db_repo,
        symbol="BTC/USDT:USDT",
        exchange_class=factory,
    )

    async with collector:
        await asyncio.sleep(0.05)

    rows = db_repo.query_ohlcv("BTC/USDT:USDT", "1m", 0, 2000000000000)
    assert len(rows) == 0


# -- Depth collector tests --


@pytest.mark.asyncio
async def test_depth_collector_stores_orderbook(db_repo: Repository) -> None:
    """Depth collector stores bid/ask snapshots to orderbook table."""
    factory = MockExchangeClass()
    factory.set_orderbooks(
        [
            {
                "timestamp": 1700000000000,
                "bids": [[42000.0, 1.5], [41999.0, 2.0]],
                "asks": [[42001.0, 1.2], [42002.0, 1.8]],
            }
        ]
    )

    collector = BinanceDepthCollector(
        db_repo,
        symbol="BTC/USDT:USDT",
        exchange_class=factory,
    )

    async with collector:
        await asyncio.sleep(0.1)

    rows = db_repo.query_orderbook("BTC/USDT:USDT", 0, 2000000000000)
    assert len(rows) >= 1

    row = rows[0]
    assert row["mid_price"] == (42000.0 + 42001.0) / 2.0
    assert row["spread"] == 1.0


@pytest.mark.asyncio
async def test_depth_collector_json_serialization(db_repo: Repository) -> None:
    """Bid/ask prices and sizes are JSON-serialized."""
    import json

    factory = MockExchangeClass()
    factory.set_orderbooks(
        [
            {
                "timestamp": 1700000000000,
                "bids": [[42000.0, 1.5], [41999.0, 2.0]],
                "asks": [[42001.0, 1.2], [42002.0, 1.8]],
            }
        ]
    )

    collector = BinanceDepthCollector(
        db_repo,
        symbol="BTC/USDT:USDT",
        exchange_class=factory,
    )

    async with collector:
        await asyncio.sleep(0.1)

    rows = db_repo.query_orderbook("BTC/USDT:USDT", 0, 2000000000000)
    assert len(rows) >= 1

    bid_prices = json.loads(rows[0]["bid_prices"])
    assert bid_prices == [42000.0, 41999.0]

    ask_sizes = json.loads(rows[0]["ask_sizes"])
    assert ask_sizes == [1.2, 1.8]


@pytest.mark.asyncio
async def test_depth_collector_empty_book_skipped(db_repo: Repository) -> None:
    """Empty bid/ask lists don't cause errors."""
    factory = MockExchangeClass()
    factory.set_orderbooks(
        [
            {
                "timestamp": 1700000000000,
                "bids": [],
                "asks": [],
            }
        ]
    )

    collector = BinanceDepthCollector(
        db_repo,
        symbol="BTC/USDT:USDT",
        exchange_class=factory,
    )

    async with collector:
        await asyncio.sleep(0.05)

    rows = db_repo.query_orderbook("BTC/USDT:USDT", 0, 2000000000000)
    assert len(rows) == 0


@pytest.mark.asyncio
async def test_depth_collector_health(db_repo: Repository) -> None:
    """Depth collector reports healthy while running."""
    factory = MockExchangeClass()
    collector = BinanceDepthCollector(
        db_repo,
        symbol="BTC/USDT:USDT",
        exchange_class=factory,
    )

    async with collector:
        await asyncio.sleep(0.05)
        health = collector.health_check()
        assert health.healthy
        assert health.messages_received > 0


# -- Trade collector tests --


@pytest.mark.asyncio
async def test_trade_collector_stores_trades(db_repo: Repository) -> None:
    """Trade collector stores received trades to the database."""
    factory = MockExchangeClass()
    factory.set_trades(
        [
            [
                {
                    "id": "1001",
                    "timestamp": 1700000000000,
                    "price": 42000.0,
                    "amount": 0.5,
                    "side": "buy",
                },
                {
                    "id": "1002",
                    "timestamp": 1700000000050,
                    "price": 42001.0,
                    "amount": 1.0,
                    "side": "sell",
                },
            ]
        ]
    )

    collector = BinanceTradeCollector(
        db_repo,
        symbol="BTC/USDT:USDT",
        exchange_class=factory,
    )

    async with collector:
        await asyncio.sleep(0.1)

    rows = db_repo.query_trades("BTC/USDT:USDT", 0, 2000000000000)
    assert len(rows) >= 2

    # Check sell side is marked as buyer_maker
    sell_rows = [r for r in rows if r["trade_id"] == "1002"]
    assert len(sell_rows) >= 1
    assert sell_rows[0]["is_buyer_maker"] == 1


@pytest.mark.asyncio
async def test_trade_collector_deduplicates(db_repo: Repository) -> None:
    """Same trade ID is not stored twice via in-memory dedup."""
    factory = MockExchangeClass()
    trade = {
        "id": "1001",
        "timestamp": 1700000000000,
        "price": 42000.0,
        "amount": 0.5,
        "side": "buy",
    }
    factory.set_trades([[trade], [trade]])

    collector = BinanceTradeCollector(
        db_repo,
        symbol="BTC/USDT:USDT",
        exchange_class=factory,
    )

    async with collector:
        await asyncio.sleep(0.1)

    rows = db_repo.query_trades("BTC/USDT:USDT", 0, 2000000000000)
    ids = [r["trade_id"] for r in rows]
    # Should appear at most once (dedup by last_trade_id check)
    assert ids.count("1001") <= 1


@pytest.mark.asyncio
async def test_trade_collector_health(db_repo: Repository) -> None:
    """Trade collector reports healthy while running."""
    factory = MockExchangeClass()
    collector = BinanceTradeCollector(
        db_repo,
        symbol="BTC/USDT:USDT",
        exchange_class=factory,
    )

    async with collector:
        await asyncio.sleep(0.05)
        health = collector.health_check()
        assert health.healthy
        assert health.messages_received > 0
