"""Integration tests for Sprint 17 execution layer.

Tests the full pipeline:
    VenueRegistry → PolymarketAdapter → BinaryMarket → place_order
    PolymarketRiskAdapter → BinarySignal → trade decision → order
    Live loop venue selection

All external SDK calls are mocked.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from ep2_crypto.execution import (
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    PolymarketAdapter,
    PolymarketConfig,
    VenueRegistry,
    VenueType,
)
from ep2_crypto.execution.polymarket import BinaryMarket


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.derive_api_key.return_value = {"key": "test_key"}
    client.get_ok.return_value = {"status": "ok"}
    client.get_balance_allowance.return_value = {"balance": 1_000.0}
    client.post_order.return_value = {"orderID": "int-ord-001", "status": "LIVE"}
    client.create_order.return_value = {"signed": "data"}
    client.cancel.return_value = {"status": "CANCELLED"}
    client.get_order.return_value = {
        "status": "FILLED",
        "size_matched": "50.0",
        "price": "0.62",
    }
    return client


@pytest.fixture
def market() -> BinaryMarket:
    return BinaryMarket(
        condition_id="0xcond_int",
        question_id="0xq_int",
        slug="will-btc-be-higher-in-5-minutes",
        question="Will BTC be higher in 5 minutes?",
        end_date_iso="2026-03-23T17:05:00Z",
        active=True,
        yes_token_id="0xyes_int",
        no_token_id="0xno_int",
        current_yes_price=0.62,
        current_no_price=0.38,
    )


@pytest.fixture
def connected_adapter(mock_client: MagicMock, market: BinaryMarket) -> PolymarketAdapter:
    config = PolymarketConfig(
        private_key="0x" + "a" * 64,
        funder_address="0xfunder",
        heartbeat_interval_s=0.05,
    )
    adapter = PolymarketAdapter(config)
    adapter._client = mock_client
    adapter._connected = True
    adapter.set_active_market(market)
    return adapter


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_register_and_retrieve_polymarket_adapter(
        self, connected_adapter: PolymarketAdapter
    ) -> None:
        registry = VenueRegistry()
        registry.register(connected_adapter)

        retrieved = registry.get(VenueType.POLYMARKET_BINARY)
        assert retrieved is connected_adapter

    def test_registry_list_venues(
        self, connected_adapter: PolymarketAdapter
    ) -> None:
        registry = VenueRegistry()
        registry.register(connected_adapter)
        venues = registry.list_venues()
        assert VenueType.POLYMARKET_BINARY in venues

    def test_registry_unregistered_venue_raises(self) -> None:
        registry = VenueRegistry()
        with pytest.raises(KeyError):
            registry.get(VenueType.POLYMARKET_BINARY)

    def test_registry_replace_adapter(
        self, connected_adapter: PolymarketAdapter, mock_client: MagicMock
    ) -> None:
        registry = VenueRegistry()
        registry.register(connected_adapter)

        # Replace with a new adapter
        config2 = PolymarketConfig(
            private_key="0x" + "b" * 64,
            funder_address="0xfunder2",
        )
        adapter2 = PolymarketAdapter(config2)
        adapter2._client = mock_client
        adapter2._connected = True
        registry.register(adapter2)

        assert registry.count == 1
        assert registry.get(VenueType.POLYMARKET_BINARY) is adapter2


# ---------------------------------------------------------------------------
# Full order pipeline
# ---------------------------------------------------------------------------


class TestOrderPipeline:
    @pytest.mark.asyncio
    async def test_full_buy_flow(
        self, connected_adapter: PolymarketAdapter
    ) -> None:
        """Place → query → cancel lifecycle."""
        # Place
        request = OrderRequest(side=OrderSide.BUY, size=100.0, price=0.62)
        place_result = await connected_adapter.place_order(request)
        assert place_result.status == OrderStatus.OPEN
        order_id = place_result.order_id

        # Query
        query_result = await connected_adapter.get_order(order_id)
        assert query_result.status == OrderStatus.FILLED

        # Cancel (should succeed even on filled order — CLOB may reject but we handle gracefully)
        cancel_result = await connected_adapter.cancel_order(order_id)
        assert cancel_result.order_id == order_id

    @pytest.mark.asyncio
    async def test_full_sell_flow(
        self, connected_adapter: PolymarketAdapter, mock_client: MagicMock
    ) -> None:
        request = OrderRequest(side=OrderSide.SELL, size=50.0, price=0.38)
        result = await connected_adapter.place_order(request)
        assert result.status == OrderStatus.OPEN

        # Verify NO token used
        call_args = mock_client.create_order.call_args[0][0]
        assert call_args["token_id"] == "0xno_int"

    @pytest.mark.asyncio
    async def test_order_fee_populated(
        self, connected_adapter: PolymarketAdapter
    ) -> None:
        request = OrderRequest(side=OrderSide.BUY, size=100.0, price=0.62)
        result = await connected_adapter.place_order(request)
        assert result.fee > 0
        assert result.fee_currency == "USDC"


# ---------------------------------------------------------------------------
# Balance and position pipeline
# ---------------------------------------------------------------------------


class TestBalanceAndPosition:
    @pytest.mark.asyncio
    async def test_balance_and_place_flow(
        self, connected_adapter: PolymarketAdapter
    ) -> None:
        balance = await connected_adapter.get_balance()
        assert balance == pytest.approx(1_000.0)

        # Can place order within balance
        request = OrderRequest(side=OrderSide.BUY, size=10.0, price=0.62)
        result = await connected_adapter.place_order(request)
        assert result.status not in (OrderStatus.REJECTED,)

    @pytest.mark.asyncio
    async def test_health_check_before_order(
        self, connected_adapter: PolymarketAdapter
    ) -> None:
        healthy = await connected_adapter.health_check()
        assert healthy

        request = OrderRequest(side=OrderSide.BUY, size=10.0, price=0.62)
        result = await connected_adapter.place_order(request)
        assert result.order_id != ""


# ---------------------------------------------------------------------------
# Live loop venue selection
# ---------------------------------------------------------------------------


class TestLiveLoopVenueSelection:
    def test_live_loop_accepts_binance_venue(self) -> None:
        import sys
        sys.path.insert(0, "/Users/edgarpocaterra/ep2-crypto/scripts")
        from live import LivePredictionLoop  # type: ignore[import]

        loop = LivePredictionLoop(venue_type=VenueType.BINANCE_PERPS)
        assert loop._venue_type == VenueType.BINANCE_PERPS

    def test_live_loop_accepts_polymarket_venue(self) -> None:
        import sys
        sys.path.insert(0, "/Users/edgarpocaterra/ep2-crypto/scripts")
        from live import LivePredictionLoop  # type: ignore[import]

        loop = LivePredictionLoop(venue_type=VenueType.POLYMARKET_BINARY)
        assert loop._venue_type == VenueType.POLYMARKET_BINARY

    def test_live_loop_default_venue_is_binance(self) -> None:
        import sys
        sys.path.insert(0, "/Users/edgarpocaterra/ep2-crypto/scripts")
        from live import LivePredictionLoop  # type: ignore[import]

        loop = LivePredictionLoop()
        assert loop._venue_type == VenueType.BINANCE_PERPS


# ---------------------------------------------------------------------------
# __init__.py clean exports
# ---------------------------------------------------------------------------


class TestCleanExports:
    def test_all_public_names_importable(self) -> None:
        from ep2_crypto.execution import __all__ as public_api

        import ep2_crypto.execution as execution_module

        for name in public_api:
            assert hasattr(execution_module, name), f"{name} missing from execution package"

    def test_polymarket_adapter_importable_from_package(self) -> None:
        from ep2_crypto.execution import PolymarketAdapter
        assert PolymarketAdapter is not None

    def test_polymarket_config_importable_from_package(self) -> None:
        from ep2_crypto.execution import PolymarketConfig
        assert PolymarketConfig is not None

    def test_venue_registry_importable_from_package(self) -> None:
        from ep2_crypto.execution import VenueRegistry
        assert VenueRegistry is not None
