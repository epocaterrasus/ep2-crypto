"""Tests for PolymarketAdapter and PolymarketConfig.

All py-clob-client SDK calls are mocked so the test suite runs without
the SDK installed (added in S17-T6). Tests cover:
- Config validation
- connect/disconnect lifecycle
- Market discovery and parsing
- Order placement (BUY/SELL, limit/market)
- Order cancellation and status polling
- Position and balance queries
- Resolution tracking
- Heartbeat loop
- Error handling (not connected, no active market, SDK exceptions)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ep2_crypto.execution.polymarket import (
    BinaryMarket,
    PolymarketAdapter,
    ResolutionEvent,
)
from ep2_crypto.execution.polymarket_config import PolymarketConfig
from ep2_crypto.execution.venue import (
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    VenueType,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> PolymarketConfig:
    return PolymarketConfig(
        private_key="0xdeadbeef" * 8,  # 64 hex chars
        funder_address="0xabc123",
        heartbeat_interval_s=0.05,  # Very fast for tests
    )


@pytest.fixture
def mock_client() -> MagicMock:
    """Fake ClobClient with common methods pre-configured."""
    client = MagicMock()
    client.derive_api_key.return_value = {"key": "test_key"}
    client.get_ok.return_value = {"status": "ok"}
    client.get_balance_allowance.return_value = {"balance": 500.0}
    client.post_order.return_value = {"orderID": "ord-001", "status": "LIVE"}
    client.cancel.return_value = {"status": "CANCELLED"}
    client.get_order.return_value = {
        "status": "FILLED",
        "size_matched": "10.0",
        "price": "0.62",
    }
    client.get_order_book.return_value = {
        "bids": [{"price": "0.60", "size": "100"}, {"price": "0.58", "size": "200"}],
        "asks": [{"price": "0.62", "size": "150"}, {"price": "0.64", "size": "100"}],
    }
    client.create_order.return_value = {"signed": "order_data"}
    return client


@pytest.fixture
def market() -> BinaryMarket:
    return BinaryMarket(
        condition_id="0xcond001",
        question_id="0xq001",
        slug="will-btc-be-higher-in-5-minutes-20260323",
        question="Will BTC be higher in 5 minutes?",
        end_date_iso="2026-03-23T17:00:00Z",
        active=True,
        yes_token_id="0xyes001",
        no_token_id="0xno001",
        current_yes_price=0.62,
        current_no_price=0.38,
    )


@pytest.fixture
def adapter(config: PolymarketConfig, mock_client: MagicMock) -> PolymarketAdapter:
    """Pre-connected adapter with mocked client."""
    a = PolymarketAdapter(config)
    a._client = mock_client
    a._connected = True
    return a


@pytest.fixture
def adapter_with_market(adapter: PolymarketAdapter, market: BinaryMarket) -> PolymarketAdapter:
    adapter.set_active_market(market)
    return adapter


# ---------------------------------------------------------------------------
# PolymarketConfig tests
# ---------------------------------------------------------------------------


class TestPolymarketConfig:
    def test_default_host(self) -> None:
        cfg = PolymarketConfig()
        assert cfg.host == "https://clob.polymarket.com"

    def test_default_chain_id(self) -> None:
        cfg = PolymarketConfig()
        assert cfg.chain_id == 137

    def test_invalid_chain_id(self) -> None:
        with pytest.raises(ValueError, match="chain_id"):
            PolymarketConfig(chain_id=1)  # Ethereum mainnet, not supported

    def test_valid_testnet_chain_id(self) -> None:
        cfg = PolymarketConfig(chain_id=80002)
        assert cfg.chain_id == 80002

    def test_invalid_heartbeat(self) -> None:
        with pytest.raises(ValueError, match="heartbeat"):
            PolymarketConfig(heartbeat_interval_s=0.0)

    def test_has_credentials_false_by_default(self) -> None:
        cfg = PolymarketConfig(private_key="", funder_address="")
        assert not cfg.has_credentials

    def test_has_credentials_true(self, config: PolymarketConfig) -> None:
        assert config.has_credentials

    def test_partial_credentials(self) -> None:
        cfg = PolymarketConfig(private_key="0xkey", funder_address="")
        assert not cfg.has_credentials


# ---------------------------------------------------------------------------
# Connection lifecycle
# ---------------------------------------------------------------------------


class TestConnectionLifecycle:
    @pytest.mark.asyncio
    async def test_connect_success(
        self, config: PolymarketConfig, mock_client: MagicMock
    ) -> None:
        adapter = PolymarketAdapter(config)
        with patch.object(adapter, "_build_client", return_value=mock_client):
            await adapter.connect()

        assert adapter.is_connected
        assert adapter._heartbeat_task is not None
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_connect_no_credentials_raises(self) -> None:
        adapter = PolymarketAdapter(PolymarketConfig(private_key="", funder_address=""))
        with pytest.raises(ConnectionError, match="credentials"):
            await adapter.connect()

    @pytest.mark.asyncio
    async def test_connect_sdk_error_raises(
        self, config: PolymarketConfig, mock_client: MagicMock
    ) -> None:
        mock_client.derive_api_key.side_effect = RuntimeError("auth_failed")
        adapter = PolymarketAdapter(config)
        with patch.object(adapter, "_build_client", return_value=mock_client):
            with pytest.raises(RuntimeError):
                await adapter.connect()
        assert not adapter.is_connected

    @pytest.mark.asyncio
    async def test_connect_already_connected_noop(
        self, adapter: PolymarketAdapter
    ) -> None:
        # Second connect on already-connected adapter should not raise
        with patch.object(adapter, "_build_client") as mock_build:
            await adapter.connect()
            mock_build.assert_not_called()

    @pytest.mark.asyncio
    async def test_disconnect_clears_state(self, adapter: PolymarketAdapter) -> None:
        # Start a heartbeat task
        adapter._heartbeat_task = asyncio.create_task(asyncio.sleep(100))
        await adapter.disconnect()

        assert not adapter.is_connected
        assert adapter._client is None
        assert adapter._heartbeat_task is None

    @pytest.mark.asyncio
    async def test_venue_type(self, adapter: PolymarketAdapter) -> None:
        assert adapter.venue_type == VenueType.POLYMARKET_BINARY


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


class TestHeartbeat:
    @pytest.mark.asyncio
    async def test_heartbeat_calls_ping(
        self, config: PolymarketConfig, mock_client: MagicMock
    ) -> None:
        adapter = PolymarketAdapter(config)
        with patch.object(adapter, "_build_client", return_value=mock_client):
            await adapter.connect()

        # Wait for at least one heartbeat cycle (interval is 0.05s)
        await asyncio.sleep(0.12)
        mock_client.get_ok.assert_called()
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_heartbeat_survives_ping_error(
        self, config: PolymarketConfig, mock_client: MagicMock
    ) -> None:
        mock_client.get_ok.side_effect = RuntimeError("network_error")
        adapter = PolymarketAdapter(config)
        with patch.object(adapter, "_build_client", return_value=mock_client):
            await adapter.connect()
        # Should not raise — heartbeat errors are logged and swallowed
        await asyncio.sleep(0.12)
        await adapter.disconnect()


# ---------------------------------------------------------------------------
# Market discovery
# ---------------------------------------------------------------------------


class TestMarketDiscovery:
    def test_parse_gamma_market(self, adapter: PolymarketAdapter) -> None:
        raw = {
            "conditionId": "0xcond",
            "questionId": "0xq",
            "slug": "will-btc-be-higher-in-5-minutes",
            "question": "Will BTC be higher?",
            "endDateIso": "2026-03-23T17:00:00Z",
            "active": True,
            "tokens": [
                {"outcome": "Yes", "token_id": "0xyes", "price": "0.65"},
                {"outcome": "No", "token_id": "0xno", "price": "0.35"},
            ],
        }
        market = adapter._parse_gamma_market(raw)

        assert market.condition_id == "0xcond"
        assert market.yes_token_id == "0xyes"
        assert market.no_token_id == "0xno"
        assert market.current_yes_price == pytest.approx(0.65)
        assert market.current_no_price == pytest.approx(0.35)
        assert market.active is True

    def test_parse_gamma_market_snake_case_keys(self, adapter: PolymarketAdapter) -> None:
        raw = {
            "condition_id": "0xcond2",
            "question_id": "0xq2",
            "slug": "btc-market",
            "question": "BTC up?",
            "end_date_iso": "2026-03-23T17:05:00Z",
            "active": True,
            "tokens": [],
        }
        market = adapter._parse_gamma_market(raw)
        assert market.condition_id == "0xcond2"

    def test_parse_gamma_market_missing_tokens_defaults(
        self, adapter: PolymarketAdapter
    ) -> None:
        raw = {
            "conditionId": "0xcond3",
            "questionId": "0xq3",
            "slug": "test",
            "question": "Test?",
            "endDateIso": "2026-03-23T17:00:00Z",
            "active": True,
            "tokens": [],
        }
        market = adapter._parse_gamma_market(raw)
        assert market.yes_token_id == ""
        assert market.current_yes_price == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_discover_market_no_results_raises(
        self, adapter: PolymarketAdapter
    ) -> None:
        with patch.object(adapter, "_fetch_gamma_markets", return_value=[]):
            with pytest.raises(LookupError, match="No active"):
                await adapter.discover_market("btc-5min")

    @pytest.mark.asyncio
    async def test_discover_market_returns_first(
        self, adapter: PolymarketAdapter, market: BinaryMarket
    ) -> None:
        with patch.object(adapter, "_fetch_gamma_markets", return_value=[market]):
            result = await adapter.discover_market()
        assert result.condition_id == market.condition_id

    def test_set_active_market(
        self, adapter: PolymarketAdapter, market: BinaryMarket
    ) -> None:
        adapter.set_active_market(market)
        assert adapter._active_market == market


# ---------------------------------------------------------------------------
# Order placement
# ---------------------------------------------------------------------------


class TestOrderPlacement:
    @pytest.mark.asyncio
    async def test_place_buy_order_success(
        self, adapter_with_market: PolymarketAdapter
    ) -> None:
        request = OrderRequest(
            side=OrderSide.BUY,
            size=50.0,
            order_type=OrderType.LIMIT,
            price=0.62,
        )
        result = await adapter_with_market.place_order(request)

        assert result.order_id == "ord-001"
        assert result.status == OrderStatus.OPEN
        assert result.venue == VenueType.POLYMARKET_BINARY

    @pytest.mark.asyncio
    async def test_place_sell_order_uses_no_token(
        self, adapter_with_market: PolymarketAdapter, mock_client: MagicMock
    ) -> None:
        request = OrderRequest(side=OrderSide.SELL, size=30.0, price=0.38)
        await adapter_with_market.place_order(request)

        # create_order should be called with no_token_id
        call_args = mock_client.create_order.call_args[0][0]
        assert call_args["token_id"] == "0xno001"

    @pytest.mark.asyncio
    async def test_place_market_order_uses_mid_price(
        self, adapter_with_market: PolymarketAdapter, mock_client: MagicMock
    ) -> None:
        request = OrderRequest(side=OrderSide.BUY, size=10.0, order_type=OrderType.MARKET)
        result = await adapter_with_market.place_order(request)

        call_args = mock_client.create_order.call_args[0][0]
        assert call_args["price"] == pytest.approx(0.62)  # current_yes_price

    @pytest.mark.asyncio
    async def test_place_order_not_connected_returns_rejected(self) -> None:
        adapter = PolymarketAdapter()
        request = OrderRequest(side=OrderSide.BUY, size=10.0)
        result = await adapter.place_order(request)
        assert result.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_place_order_no_active_market_returns_rejected(
        self, adapter: PolymarketAdapter
    ) -> None:
        request = OrderRequest(side=OrderSide.BUY, size=10.0)
        result = await adapter.place_order(request)
        assert result.status == OrderStatus.REJECTED
        assert "no_active_market" in str(result.raw_response)

    @pytest.mark.asyncio
    async def test_place_order_sdk_exception_returns_rejected(
        self, adapter_with_market: PolymarketAdapter, mock_client: MagicMock
    ) -> None:
        mock_client.post_order.side_effect = RuntimeError("sdk_error")
        request = OrderRequest(side=OrderSide.BUY, size=10.0, price=0.60)
        result = await adapter_with_market.place_order(request)
        assert result.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_place_order_filled_status(
        self, adapter_with_market: PolymarketAdapter, mock_client: MagicMock
    ) -> None:
        mock_client.post_order.return_value = {"orderID": "ord-filled", "status": "MATCHED"}
        request = OrderRequest(side=OrderSide.BUY, size=20.0, price=0.62)
        result = await adapter_with_market.place_order(request)
        assert result.status == OrderStatus.FILLED
        assert result.fill_quantity == pytest.approx(20.0)

    def test_fee_estimation(self, adapter: PolymarketAdapter) -> None:
        # 2% of notional: 100 shares * $0.60 * 0.02 = $1.20
        fee = adapter._estimate_fee(size=100.0, price=0.60)
        assert fee == pytest.approx(1.20, abs=0.001)

    def test_order_status_mapping(self) -> None:
        cases = [
            ("LIVE", OrderStatus.OPEN),
            ("OPEN", OrderStatus.OPEN),
            ("FILLED", OrderStatus.FILLED),
            ("MATCHED", OrderStatus.FILLED),
            ("CANCELLED", OrderStatus.CANCELLED),
            ("CANCELED", OrderStatus.CANCELLED),
            ("EXPIRED", OrderStatus.EXPIRED),
            ("PARTIALLY_FILLED", OrderStatus.PARTIALLY_FILLED),
            ("UNKNOWN_STATUS", OrderStatus.PENDING),
        ]
        for raw, expected in cases:
            assert PolymarketAdapter._map_order_status(raw) == expected


# ---------------------------------------------------------------------------
# Order management
# ---------------------------------------------------------------------------


class TestOrderManagement:
    @pytest.mark.asyncio
    async def test_cancel_order_success(
        self, adapter_with_market: PolymarketAdapter
    ) -> None:
        result = await adapter_with_market.cancel_order("ord-001")
        assert result.status == OrderStatus.CANCELLED
        assert result.order_id == "ord-001"

    @pytest.mark.asyncio
    async def test_cancel_order_not_connected(self) -> None:
        adapter = PolymarketAdapter()
        result = await adapter.cancel_order("ord-001")
        assert result.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_cancel_sdk_error_returns_rejected(
        self, adapter_with_market: PolymarketAdapter, mock_client: MagicMock
    ) -> None:
        mock_client.cancel.side_effect = RuntimeError("cancel_failed")
        result = await adapter_with_market.cancel_order("ord-bad")
        assert result.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_get_order_filled(
        self, adapter_with_market: PolymarketAdapter
    ) -> None:
        result = await adapter_with_market.get_order("ord-001")
        assert result.status == OrderStatus.FILLED
        assert result.fill_quantity == pytest.approx(10.0)
        assert result.fill_price == pytest.approx(0.62)

    @pytest.mark.asyncio
    async def test_get_order_not_connected(self) -> None:
        adapter = PolymarketAdapter()
        result = await adapter.get_order("ord-001")
        assert result.status == OrderStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_order_sdk_error(
        self, adapter_with_market: PolymarketAdapter, mock_client: MagicMock
    ) -> None:
        mock_client.get_order.side_effect = RuntimeError("network_error")
        result = await adapter_with_market.get_order("ord-bad")
        assert result.status == OrderStatus.PENDING

    def test_get_pending_orders_excludes_terminal(
        self, adapter_with_market: PolymarketAdapter
    ) -> None:
        from ep2_crypto.execution.polymarket import _PendingOrder

        adapter_with_market._pending_orders["o1"] = _PendingOrder(
            order_id="o1",
            condition_id="0xcond001",
            side=OrderSide.BUY,
            token_id="0xyes",
            size=10.0,
            price=0.60,
            status=OrderStatus.OPEN,
        )
        adapter_with_market._pending_orders["o2"] = _PendingOrder(
            order_id="o2",
            condition_id="0xcond001",
            side=OrderSide.BUY,
            token_id="0xyes",
            size=10.0,
            price=0.60,
            status=OrderStatus.FILLED,
        )
        pending = adapter_with_market.get_pending_orders()
        assert len(pending) == 1
        assert pending[0].order_id == "o1"


# ---------------------------------------------------------------------------
# Position and balance
# ---------------------------------------------------------------------------


class TestPositionAndBalance:
    @pytest.mark.asyncio
    async def test_get_position_no_active_market(
        self, adapter: PolymarketAdapter
    ) -> None:
        position = await adapter.get_position()
        assert position.side is None
        assert position.size == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_get_position_with_filled_buy(
        self, adapter_with_market: PolymarketAdapter
    ) -> None:
        from ep2_crypto.execution.polymarket import _PendingOrder

        adapter_with_market._pending_orders["o1"] = _PendingOrder(
            order_id="o1",
            condition_id="0xcond001",
            side=OrderSide.BUY,
            token_id="0xyes001",
            size=50.0,
            price=0.62,
            status=OrderStatus.FILLED,
        )
        position = await adapter_with_market.get_position()
        assert position.side == OrderSide.BUY
        assert position.size == pytest.approx(50.0)
        assert position.cost_basis == pytest.approx(31.0)  # 50 * 0.62

    @pytest.mark.asyncio
    async def test_get_position_net_zero(
        self, adapter_with_market: PolymarketAdapter
    ) -> None:
        from ep2_crypto.execution.polymarket import _PendingOrder

        for oid, side in [("o1", OrderSide.BUY), ("o2", OrderSide.SELL)]:
            adapter_with_market._pending_orders[oid] = _PendingOrder(
                order_id=oid,
                condition_id="0xcond001",
                side=side,
                token_id="tok",
                size=25.0,
                price=0.50,
                status=OrderStatus.FILLED,
            )
        position = await adapter_with_market.get_position()
        assert position.side is None
        assert position.size == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_get_balance_success(
        self, adapter_with_market: PolymarketAdapter
    ) -> None:
        balance = await adapter_with_market.get_balance()
        assert balance == pytest.approx(500.0)

    @pytest.mark.asyncio
    async def test_get_balance_not_connected(self) -> None:
        adapter = PolymarketAdapter()
        balance = await adapter.get_balance()
        assert balance == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_get_balance_sdk_error(
        self, adapter_with_market: PolymarketAdapter, mock_client: MagicMock
    ) -> None:
        mock_client.get_balance_allowance.side_effect = RuntimeError("err")
        balance = await adapter_with_market.get_balance()
        assert balance == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Order book
# ---------------------------------------------------------------------------


class TestOrderBook:
    @pytest.mark.asyncio
    async def test_get_orderbook_success(
        self, adapter_with_market: PolymarketAdapter
    ) -> None:
        snapshot = await adapter_with_market.get_orderbook()
        assert snapshot.venue == VenueType.POLYMARKET_BINARY
        assert len(snapshot.bids) == 2
        assert len(snapshot.asks) == 2
        # Bids sorted highest first
        assert snapshot.bids[0].price > snapshot.bids[1].price
        # Asks sorted lowest first
        assert snapshot.asks[0].price < snapshot.asks[1].price

    @pytest.mark.asyncio
    async def test_get_orderbook_not_connected(self) -> None:
        adapter = PolymarketAdapter()
        snapshot = await adapter.get_orderbook()
        assert snapshot.bids == []
        assert snapshot.asks == []

    @pytest.mark.asyncio
    async def test_get_orderbook_sdk_error(
        self, adapter_with_market: PolymarketAdapter, mock_client: MagicMock
    ) -> None:
        mock_client.get_order_book.side_effect = RuntimeError("err")
        snapshot = await adapter_with_market.get_orderbook()
        assert snapshot.bids == []

    @pytest.mark.asyncio
    async def test_orderbook_mid_price(
        self, adapter_with_market: PolymarketAdapter
    ) -> None:
        snapshot = await adapter_with_market.get_orderbook()
        # best bid=0.60, best ask=0.62
        assert snapshot.best_bid == pytest.approx(0.60)
        assert snapshot.best_ask == pytest.approx(0.62)
        assert snapshot.mid_price == pytest.approx(0.61)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_success(
        self, adapter_with_market: PolymarketAdapter
    ) -> None:
        assert await adapter_with_market.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_not_connected(self) -> None:
        adapter = PolymarketAdapter()
        assert await adapter.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_sdk_error(
        self, adapter_with_market: PolymarketAdapter, mock_client: MagicMock
    ) -> None:
        mock_client.get_ok.side_effect = RuntimeError("unhealthy")
        assert await adapter_with_market.health_check() is False


# ---------------------------------------------------------------------------
# Resolution tracking
# ---------------------------------------------------------------------------


class TestResolutionTracking:
    @pytest.mark.asyncio
    async def test_poll_resolution_unresolved_returns_none(
        self, adapter_with_market: PolymarketAdapter, mock_client: MagicMock
    ) -> None:
        mock_client.get_market.return_value = {"closed": False, "resolved": False}
        result = await adapter_with_market.poll_resolution("0xcond001")
        assert result is None

    @pytest.mark.asyncio
    async def test_poll_resolution_resolved_returns_event(
        self, adapter_with_market: PolymarketAdapter, mock_client: MagicMock
    ) -> None:
        mock_client.get_market.return_value = {
            "closed": True,
            "winner": "Yes",
            "tokens": [
                {"outcome": "Yes", "token_id": "0xyes001"},
                {"outcome": "No", "token_id": "0xno001"},
            ],
        }
        event = await adapter_with_market.poll_resolution("0xcond001")
        assert event is not None
        assert event.resolved_outcome == "yes"
        assert event.winning_token_id == "0xyes001"

    @pytest.mark.asyncio
    async def test_poll_resolution_fires_callbacks(
        self, adapter_with_market: PolymarketAdapter, mock_client: MagicMock
    ) -> None:
        mock_client.get_market.return_value = {
            "closed": True,
            "winner": "No",
            "tokens": [{"outcome": "No", "token_id": "0xno001"}],
        }
        received: list[ResolutionEvent] = []
        adapter_with_market.add_resolution_callback(received.append)

        await adapter_with_market.poll_resolution("0xcond001")
        assert len(received) == 1
        assert received[0].resolved_outcome == "no"

    @pytest.mark.asyncio
    async def test_poll_resolution_not_connected(self) -> None:
        adapter = PolymarketAdapter()
        result = await adapter.poll_resolution("0xcond001")
        assert result is None

    @pytest.mark.asyncio
    async def test_poll_resolution_sdk_error(
        self, adapter_with_market: PolymarketAdapter, mock_client: MagicMock
    ) -> None:
        mock_client.get_market.side_effect = RuntimeError("rpc_error")
        result = await adapter_with_market.poll_resolution("0xcond001")
        assert result is None

    def test_resolution_callback_exception_does_not_propagate(
        self, adapter_with_market: PolymarketAdapter
    ) -> None:
        def bad_callback(event: ResolutionEvent) -> None:
            raise RuntimeError("callback_error")

        adapter_with_market.add_resolution_callback(bad_callback)
        event = ResolutionEvent(
            condition_id="0xcond",
            resolved_outcome="yes",
            resolved_at_ms=int(time.time() * 1000),
            winning_token_id="0xyes",
        )
        # Should not raise
        adapter_with_market._fire_resolution(event)


# ---------------------------------------------------------------------------
# Token ID resolution
# ---------------------------------------------------------------------------


class TestTokenResolution:
    def test_buy_side_maps_to_yes_token(
        self, adapter_with_market: PolymarketAdapter
    ) -> None:
        request = OrderRequest(side=OrderSide.BUY, size=10.0)
        token_id = adapter_with_market._resolve_token_id(request)
        assert token_id == "0xyes001"

    def test_sell_side_maps_to_no_token(
        self, adapter_with_market: PolymarketAdapter
    ) -> None:
        request = OrderRequest(side=OrderSide.SELL, size=10.0)
        token_id = adapter_with_market._resolve_token_id(request)
        assert token_id == "0xno001"
