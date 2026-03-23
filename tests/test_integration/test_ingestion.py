"""Integration test: full ingestion pipeline with mock exchange data.

Verifies that the orchestrator manages multiple collectors, data flows
to the database, health checks work, and shutdown is clean.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from ep2_crypto.db.repository import Repository
from ep2_crypto.db.schema import create_tables
from ep2_crypto.ingest.base import CollectorState
from ep2_crypto.ingest.derivatives import (
    BybitFundingCollector,
    BybitLiquidationCollector,
    BybitOICollector,
)
from ep2_crypto.ingest.exchange import (
    BinanceDepthCollector,
    BinanceKlineCollector,
    BinanceTradeCollector,
)
from ep2_crypto.ingest.orchestrator import Orchestrator


@pytest.fixture()
def db_repo() -> Repository:
    conn = create_tables(":memory:")
    return Repository(conn)


# -- Mock classes (simplified versions for integration) --


class MockCcxtExchange:
    """Mock ccxt pro exchange with all watch methods."""

    def __init__(self, config: dict | None = None) -> None:
        self._call_count = 0

    async def watch_ohlcv(self, symbol: str, timeframe: str = "1m", **kwargs: object) -> list:
        self._call_count += 1
        await asyncio.sleep(0.01)
        ts = 1700000000000 + (self._call_count * 60000)
        return [[ts, 42000.0, 42050.0, 41980.0, 42030.0, 100.5]]

    async def watch_order_book(self, symbol: str, **kwargs: object) -> dict:
        self._call_count += 1
        await asyncio.sleep(0.01)
        ts = 1700000000000 + (self._call_count * 100)
        return {
            "timestamp": ts,
            "bids": [[42000.0, 1.5], [41999.0, 2.0]],
            "asks": [[42001.0, 1.2], [42002.0, 1.8]],
        }

    async def watch_trades(self, symbol: str, **kwargs: object) -> list:
        self._call_count += 1
        await asyncio.sleep(0.01)
        ts = 1700000000000 + (self._call_count * 50)
        return [
            {
                "id": str(1000 + self._call_count),
                "timestamp": ts,
                "price": 42000.0,
                "amount": 0.5,
                "side": "buy",
            }
        ]

    async def close(self) -> None:
        pass


class MockCcxtExchangeClass:
    def __call__(self, config: dict | None = None) -> MockCcxtExchange:
        return MockCcxtExchange(config)


class MockBybitRest:
    """Mock Bybit REST exchange for OI and funding."""

    def __init__(self, config: dict | None = None) -> None:
        self._oi_count = 0
        self._funding_count = 0

    def fetch_open_interest(self, symbol: str) -> dict:
        self._oi_count += 1
        return {
            "timestamp": 1700000000000 + (self._oi_count * 300000),
            "openInterestAmount": 50000.0,
            "openInterestValue": 2100000000.0,
        }

    def fetch_funding_rate(self, symbol: str) -> dict:
        self._funding_count += 1
        return {
            "timestamp": 1700000000000 + (self._funding_count * 300000),
            "fundingRate": 0.0001,
            "markPrice": 42000.0,
            "indexPrice": 41999.0,
        }


class MockBybitRestClass:
    def __call__(self, config: dict | None = None) -> MockBybitRest:
        return MockBybitRest(config)


class MockLiqWs:
    """Mock WebSocket for liquidation stream."""

    def __init__(self) -> None:
        self._sent: list[str] = []
        self._msg_index = 0
        self._messages = [
            json.dumps({"op": "subscribe", "success": True}),
            json.dumps(
                {
                    "topic": "allLiquidation.BTCUSDT",
                    "data": {
                        "updatedTime": 1700000000000,
                        "symbol": "BTCUSDT",
                        "side": "Buy",
                        "price": "42000.0",
                        "size": "1.5",
                    },
                }
            ),
        ]

    async def send(self, msg: str) -> None:
        self._sent.append(msg)

    async def recv(self) -> str:
        await asyncio.sleep(0.01)
        if self._msg_index < len(self._messages):
            result = self._messages[self._msg_index]
            self._msg_index += 1
            return result
        await asyncio.sleep(1.0)
        return json.dumps({"op": "pong"})

    async def close(self) -> None:
        pass


# -- Integration tests --


@pytest.mark.asyncio
async def test_full_ingestion_pipeline(db_repo: Repository) -> None:
    """All collectors run together via orchestrator, data reaches DB."""
    ccxt_factory = MockCcxtExchangeClass()
    bybit_factory = MockBybitRestClass()

    async def liq_ws_connector(url: str) -> MockLiqWs:
        return MockLiqWs()

    # Create all collectors
    kline = BinanceKlineCollector(
        db_repo,
        symbol="BTC/USDT:USDT",
        exchange_class=ccxt_factory,
    )
    depth = BinanceDepthCollector(
        db_repo,
        symbol="BTC/USDT:USDT",
        exchange_class=ccxt_factory,
    )
    trades = BinanceTradeCollector(
        db_repo,
        symbol="BTC/USDT:USDT",
        exchange_class=ccxt_factory,
    )
    oi = BybitOICollector(
        db_repo,
        symbol="BTCUSDT",
        poll_interval_s=0.02,
        exchange_class=bybit_factory,
    )
    funding = BybitFundingCollector(
        db_repo,
        symbol="BTCUSDT",
        poll_interval_s=0.02,
        exchange_class=bybit_factory,
    )
    liquidation = BybitLiquidationCollector(
        db_repo,
        symbol="BTCUSDT",
        ws_connector=liq_ws_connector,
    )

    # Wire up orchestrator
    orch = Orchestrator()
    for collector in [kline, depth, trades, oi, funding, liquidation]:
        orch.add_collector(collector)

    assert len(orch.collector_names) == 6

    # Run for a short period
    async def shutdown_after_delay() -> None:
        await asyncio.sleep(0.2)
        await orch.shutdown()

    _task = asyncio.create_task(shutdown_after_delay())
    await orch.run()

    # Verify data reached the database
    ohlcv_rows = db_repo.query_ohlcv("BTC/USDT:USDT", "1m", 0, 2000000000000)
    assert len(ohlcv_rows) >= 1, "No OHLCV data stored"

    orderbook_rows = db_repo.query_orderbook("BTC/USDT:USDT", 0, 2000000000000)
    assert len(orderbook_rows) >= 1, "No orderbook data stored"

    trade_rows = db_repo.query_trades("BTC/USDT:USDT", 0, 2000000000000)
    assert len(trade_rows) >= 1, "No trade data stored"

    oi_rows = db_repo.query_open_interest("BTC/USDT:USDT", 0, 2000000000000)
    assert len(oi_rows) >= 1, "No OI data stored"

    funding_rows = db_repo.query_funding_rate("BTC/USDT:USDT", 0, 2000000000000)
    assert len(funding_rows) >= 1, "No funding rate data stored"

    liq_rows = db_repo.query_liquidations("BTC/USDT:USDT", 0, 2000000000000)
    assert len(liq_rows) >= 1, "No liquidation data stored"


@pytest.mark.asyncio
async def test_health_check_all_collectors(db_repo: Repository) -> None:
    """Health check aggregates all collector states during run."""
    ccxt_factory = MockCcxtExchangeClass()
    bybit_factory = MockBybitRestClass()

    kline = BinanceKlineCollector(
        db_repo,
        symbol="BTC/USDT:USDT",
        exchange_class=ccxt_factory,
    )
    oi = BybitOICollector(
        db_repo,
        symbol="BTCUSDT",
        poll_interval_s=0.02,
        exchange_class=bybit_factory,
    )

    orch = Orchestrator()
    orch.add_collector(kline)
    orch.add_collector(oi)

    async def check_and_shutdown() -> None:
        await asyncio.sleep(0.1)

        health = orch.health_check()
        assert health.healthy
        assert len(health.collectors) == 2
        assert health.unhealthy_count == 0
        assert health.total_messages > 0

        # Verify to_dict serialization
        d = health.to_dict()
        assert len(d["collectors"]) == 2
        assert all(c["healthy"] for c in d["collectors"])

        await orch.shutdown()

    _task = asyncio.create_task(check_and_shutdown())
    await orch.run()


@pytest.mark.asyncio
async def test_clean_shutdown_all_stopped(db_repo: Repository) -> None:
    """All collectors reach STOPPED state after orchestrator shutdown."""
    ccxt_factory = MockCcxtExchangeClass()

    collectors = [
        BinanceKlineCollector(
            db_repo,
            symbol="BTC/USDT:USDT",
            exchange_class=ccxt_factory,
        ),
        BinanceDepthCollector(
            db_repo,
            symbol="BTC/USDT:USDT",
            exchange_class=ccxt_factory,
        ),
        BinanceTradeCollector(
            db_repo,
            symbol="BTC/USDT:USDT",
            exchange_class=ccxt_factory,
        ),
    ]

    orch = Orchestrator()
    for c in collectors:
        orch.add_collector(c)

    async def shutdown_soon() -> None:
        await asyncio.sleep(0.05)
        await orch.shutdown()

    _task = asyncio.create_task(shutdown_soon())
    await orch.run()

    for c in collectors:
        assert c.state == CollectorState.STOPPED, f"{c.name} not stopped"

    assert not orch.running


@pytest.mark.asyncio
async def test_no_duplicate_records(db_repo: Repository) -> None:
    """Data deduplication works across the pipeline."""
    ccxt_factory = MockCcxtExchangeClass()

    kline = BinanceKlineCollector(
        db_repo,
        symbol="BTC/USDT:USDT",
        exchange_class=ccxt_factory,
    )

    orch = Orchestrator()
    orch.add_collector(kline)

    async def shutdown_delayed() -> None:
        await asyncio.sleep(0.15)
        await orch.shutdown()

    _task = asyncio.create_task(shutdown_delayed())
    await orch.run()

    rows = db_repo.query_ohlcv("BTC/USDT:USDT", "1m", 0, 2000000000000)
    timestamps = [r["timestamp_ms"] for r in rows]

    # No duplicate timestamps
    assert len(timestamps) == len(set(timestamps)), "Duplicate timestamps found"
