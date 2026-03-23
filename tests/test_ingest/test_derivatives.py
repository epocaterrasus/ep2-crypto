"""Tests for Bybit derivatives collectors (OI, funding rate, liquidations)."""

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


@pytest.fixture()
def db_repo() -> Repository:
    """In-memory SQLite database with schema."""
    conn = create_tables(":memory:")
    return Repository(conn)


class MockBybitExchange:
    """Mock Bybit REST exchange for OI and funding rate."""

    def __init__(self, config: dict | None = None) -> None:
        self._config = config or {}
        self._oi_responses: list[dict] = []
        self._funding_responses: list[dict] = []
        self._oi_call_count = 0
        self._funding_call_count = 0
        self._fail_oi_after: int = 0
        self._fail_funding_after: int = 0

    def set_oi_responses(self, responses: list[dict]) -> None:
        self._oi_responses = responses

    def set_funding_responses(self, responses: list[dict]) -> None:
        self._funding_responses = responses

    def set_fail_oi_after(self, n: int) -> None:
        self._fail_oi_after = n

    def set_fail_funding_after(self, n: int) -> None:
        self._fail_funding_after = n

    def fetch_open_interest(self, symbol: str) -> dict:
        self._oi_call_count += 1
        if self._fail_oi_after > 0 and self._oi_call_count >= self._fail_oi_after:
            msg = "API error"
            raise ConnectionError(msg)
        if self._oi_responses:
            idx = min(self._oi_call_count - 1, len(self._oi_responses) - 1)
            return self._oi_responses[idx]
        return {
            "timestamp": 1700000000000 + (self._oi_call_count * 300000),
            "openInterestAmount": 50000.0 + self._oi_call_count * 100,
            "openInterestValue": 2100000000.0,
        }

    def fetch_funding_rate(self, symbol: str) -> dict:
        self._funding_call_count += 1
        if self._fail_funding_after > 0 and self._funding_call_count >= self._fail_funding_after:
            msg = "API error"
            raise ConnectionError(msg)
        if self._funding_responses:
            idx = min(self._funding_call_count - 1, len(self._funding_responses) - 1)
            return self._funding_responses[idx]
        return {
            "timestamp": 1700000000000 + (self._funding_call_count * 300000),
            "fundingRate": 0.0001 * self._funding_call_count,
            "markPrice": 42000.0,
            "indexPrice": 41999.0,
        }


class MockBybitExchangeClass:
    """Factory for MockBybitExchange instances."""

    def __init__(self) -> None:
        self.last_instance: MockBybitExchange | None = None
        self._oi_responses: list[dict] = []
        self._funding_responses: list[dict] = []
        self._fail_oi_after: int = 0
        self._fail_funding_after: int = 0

    def set_oi_responses(self, responses: list[dict]) -> None:
        self._oi_responses = responses

    def set_funding_responses(self, responses: list[dict]) -> None:
        self._funding_responses = responses

    def set_fail_oi_after(self, n: int) -> None:
        self._fail_oi_after = n

    def set_fail_funding_after(self, n: int) -> None:
        self._fail_funding_after = n

    def __call__(self, config: dict | None = None) -> MockBybitExchange:
        instance = MockBybitExchange(config)
        if self._oi_responses:
            instance.set_oi_responses(self._oi_responses)
        if self._funding_responses:
            instance.set_funding_responses(self._funding_responses)
        if self._fail_oi_after > 0:
            instance.set_fail_oi_after(self._fail_oi_after)
        if self._fail_funding_after > 0:
            instance.set_fail_funding_after(self._fail_funding_after)
        self.last_instance = instance
        return instance


# -- OI collector tests --


@pytest.mark.asyncio
async def test_oi_collector_stores_data(db_repo: Repository) -> None:
    """OI collector stores fetched open interest to database."""
    factory = MockBybitExchangeClass()
    factory.set_oi_responses(
        [
            {
                "timestamp": 1700000000000,
                "openInterestAmount": 50000.0,
                "openInterestValue": 2100000000.0,
            }
        ]
    )

    collector = BybitOICollector(
        db_repo,
        symbol="BTCUSDT",
        poll_interval_s=0.01,
        exchange_class=factory,
    )

    async with collector:
        await asyncio.sleep(0.1)

    rows = db_repo.query_open_interest("BTC/USDT:USDT", 0, 2000000000000)
    assert len(rows) >= 1

    row = rows[0]
    assert row["open_interest"] == 50000.0
    assert row["oi_value_usd"] == 2100000000.0


@pytest.mark.asyncio
async def test_oi_collector_polls_repeatedly(db_repo: Repository) -> None:
    """OI collector polls at the configured interval."""
    factory = MockBybitExchangeClass()

    collector = BybitOICollector(
        db_repo,
        symbol="BTCUSDT",
        poll_interval_s=0.02,
        exchange_class=factory,
    )

    async with collector:
        await asyncio.sleep(0.30)

    health = collector.health_check()
    assert health.messages_received >= 2


@pytest.mark.asyncio
async def test_oi_collector_reconnects_on_error(db_repo: Repository) -> None:
    """OI collector reconnects when fetch raises."""
    factory = MockBybitExchangeClass()
    factory.set_fail_oi_after(2)

    collector = BybitOICollector(
        db_repo,
        symbol="BTCUSDT",
        poll_interval_s=0.01,
        exchange_class=factory,
        reconnect_delay_s=0.01,
    )

    async with collector:
        await asyncio.sleep(0.15)

    assert collector._reconnect_count > 0


@pytest.mark.asyncio
async def test_oi_collector_health(db_repo: Repository) -> None:
    """OI collector reports healthy state."""
    factory = MockBybitExchangeClass()

    collector = BybitOICollector(
        db_repo,
        symbol="BTCUSDT",
        poll_interval_s=0.01,
        exchange_class=factory,
    )

    async with collector:
        await asyncio.sleep(0.05)
        health = collector.health_check()
        assert health.healthy
        assert health.state == CollectorState.RUNNING


# -- Funding collector tests --


@pytest.mark.asyncio
async def test_funding_collector_stores_data(db_repo: Repository) -> None:
    """Funding collector stores fetched funding rate to database."""
    factory = MockBybitExchangeClass()
    factory.set_funding_responses(
        [
            {
                "timestamp": 1700000000000,
                "fundingRate": 0.0001,
                "markPrice": 42000.0,
                "indexPrice": 41999.0,
            }
        ]
    )

    collector = BybitFundingCollector(
        db_repo,
        symbol="BTCUSDT",
        poll_interval_s=0.01,
        exchange_class=factory,
    )

    async with collector:
        await asyncio.sleep(0.1)

    rows = db_repo.query_funding_rate("BTC/USDT:USDT", 0, 2000000000000)
    assert len(rows) >= 1

    row = rows[0]
    assert row["funding_rate"] == 0.0001
    assert row["mark_price"] == 42000.0
    assert row["index_price"] == 41999.0


@pytest.mark.asyncio
async def test_funding_collector_handles_missing_prices(db_repo: Repository) -> None:
    """Funding collector handles missing mark/index prices."""
    factory = MockBybitExchangeClass()
    factory.set_funding_responses(
        [
            {
                "timestamp": 1700000000000,
                "fundingRate": 0.0002,
            }
        ]
    )

    collector = BybitFundingCollector(
        db_repo,
        symbol="BTCUSDT",
        poll_interval_s=0.01,
        exchange_class=factory,
    )

    async with collector:
        await asyncio.sleep(0.1)

    rows = db_repo.query_funding_rate("BTC/USDT:USDT", 0, 2000000000000)
    assert len(rows) >= 1

    row = rows[0]
    assert row["funding_rate"] == 0.0002
    assert row["mark_price"] is None
    assert row["index_price"] is None


@pytest.mark.asyncio
async def test_funding_collector_reconnects_on_error(db_repo: Repository) -> None:
    """Funding collector reconnects when fetch raises."""
    factory = MockBybitExchangeClass()
    factory.set_fail_funding_after(2)

    collector = BybitFundingCollector(
        db_repo,
        symbol="BTCUSDT",
        poll_interval_s=0.01,
        exchange_class=factory,
        reconnect_delay_s=0.01,
    )

    async with collector:
        await asyncio.sleep(0.15)

    assert collector._reconnect_count > 0


@pytest.mark.asyncio
async def test_funding_collector_health(db_repo: Repository) -> None:
    """Funding collector reports healthy state."""
    factory = MockBybitExchangeClass()

    collector = BybitFundingCollector(
        db_repo,
        symbol="BTCUSDT",
        poll_interval_s=0.01,
        exchange_class=factory,
    )

    async with collector:
        await asyncio.sleep(0.05)
        health = collector.health_check()
        assert health.healthy


# -- Mock WebSocket for liquidation tests --


class MockWebSocket:
    """Mock WebSocket connection for testing."""

    def __init__(self) -> None:
        self._messages: list[str] = []
        self._sent: list[str] = []
        self._recv_index = 0
        self._closed = False
        self._fail_recv_after: int = 0
        self._recv_count = 0

    def set_messages(self, messages: list[str]) -> None:
        self._messages = messages

    def set_fail_recv_after(self, n: int) -> None:
        self._fail_recv_after = n

    async def send(self, msg: str) -> None:
        self._sent.append(msg)

    async def recv(self) -> str:
        self._recv_count += 1
        if self._fail_recv_after > 0 and self._recv_count >= self._fail_recv_after:
            msg = "Connection closed"
            raise ConnectionError(msg)

        await asyncio.sleep(0.01)

        if self._recv_index < len(self._messages):
            result = self._messages[self._recv_index]
            self._recv_index += 1
            return result

        # If no more messages, keep returning pong to stay alive
        await asyncio.sleep(0.5)
        return json.dumps({"op": "pong"})

    async def close(self) -> None:
        self._closed = True


def make_liquidation_msg(
    side: str = "Buy",
    price: float = 42000.0,
    size: float = 1.5,
    updated_time: int = 1700000000000,
) -> str:
    """Create a mock liquidation WebSocket message."""
    return json.dumps(
        {
            "topic": "allLiquidation.BTCUSDT",
            "data": {
                "updatedTime": updated_time,
                "symbol": "BTCUSDT",
                "side": side,
                "price": str(price),
                "size": str(size),
            },
        }
    )


# -- Liquidation collector tests --


@pytest.mark.asyncio
async def test_liquidation_collector_stores_events(db_repo: Repository) -> None:
    """Liquidation collector stores liquidation events."""
    ws = MockWebSocket()
    ws.set_messages(
        [
            json.dumps({"op": "subscribe", "success": True}),
            make_liquidation_msg(side="Buy", price=42000.0, size=1.5),
            make_liquidation_msg(
                side="Sell",
                price=41500.0,
                size=2.0,
                updated_time=1700000001000,
            ),
        ]
    )

    async def ws_connector(url: str) -> MockWebSocket:
        return ws

    collector = BybitLiquidationCollector(
        db_repo,
        symbol="BTCUSDT",
        ws_connector=ws_connector,
    )

    async with collector:
        await asyncio.sleep(0.15)

    rows = db_repo.query_liquidations("BTC/USDT:USDT", 0, 2000000000000)
    assert len(rows) >= 2

    # Buy side = short liquidated
    short_rows = [r for r in rows if r["side"] == "short"]
    assert len(short_rows) >= 1
    assert float(short_rows[0]["price"]) == 42000.0

    # Sell side = long liquidated
    long_rows = [r for r in rows if r["side"] == "long"]
    assert len(long_rows) >= 1
    assert float(long_rows[0]["price"]) == 41500.0


@pytest.mark.asyncio
async def test_liquidation_collector_sends_subscription(
    db_repo: Repository,
) -> None:
    """Collector sends subscription message on connect."""
    ws = MockWebSocket()
    ws.set_messages([json.dumps({"op": "subscribe", "success": True})])

    async def ws_connector(url: str) -> MockWebSocket:
        return ws

    collector = BybitLiquidationCollector(
        db_repo,
        symbol="BTCUSDT",
        ws_connector=ws_connector,
    )

    async with collector:
        await asyncio.sleep(0.05)

    # Check subscription was sent
    assert len(ws._sent) >= 1
    sub_msg = json.loads(ws._sent[0])
    assert sub_msg["op"] == "subscribe"
    assert "allLiquidation.BTCUSDT" in sub_msg["args"]


@pytest.mark.asyncio
async def test_liquidation_collector_skips_non_data_messages(
    db_repo: Repository,
) -> None:
    """Non-data messages (pong, subscribe) are ignored."""
    ws = MockWebSocket()
    ws.set_messages(
        [
            json.dumps({"op": "pong"}),
            json.dumps({"op": "subscribe", "success": True}),
            json.dumps({"topic": "other.topic", "data": {}}),
        ]
    )

    async def ws_connector(url: str) -> MockWebSocket:
        return ws

    collector = BybitLiquidationCollector(
        db_repo,
        symbol="BTCUSDT",
        ws_connector=ws_connector,
    )

    async with collector:
        await asyncio.sleep(0.1)

    rows = db_repo.query_liquidations("BTC/USDT:USDT", 0, 2000000000000)
    assert len(rows) == 0


@pytest.mark.asyncio
async def test_liquidation_collector_reconnects_on_error(
    db_repo: Repository,
) -> None:
    """Liquidation collector reconnects on WebSocket errors."""

    async def ws_connector(url: str) -> MockWebSocket:
        new_ws = MockWebSocket()
        # Fail on first recv so the run_loop raises quickly
        new_ws.set_fail_recv_after(1)
        return new_ws

    collector = BybitLiquidationCollector(
        db_repo,
        symbol="BTCUSDT",
        ws_connector=ws_connector,
        reconnect_delay_s=0.01,
    )

    async with collector:
        await asyncio.sleep(0.3)

    assert collector._reconnect_count > 0


@pytest.mark.asyncio
async def test_liquidation_collector_health(db_repo: Repository) -> None:
    """Liquidation collector reports healthy state."""
    ws = MockWebSocket()
    ws.set_messages(
        [
            make_liquidation_msg(),
        ]
    )

    async def ws_connector(url: str) -> MockWebSocket:
        return ws

    collector = BybitLiquidationCollector(
        db_repo,
        symbol="BTCUSDT",
        ws_connector=ws_connector,
    )

    async with collector:
        await asyncio.sleep(0.1)
        health = collector.health_check()
        assert health.healthy
        assert health.messages_received > 0
