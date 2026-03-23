"""Tests for VenueAdapter ABC and VenueRegistry."""

from __future__ import annotations

import asyncio

import pytest

from ep2_crypto.execution.registry import VenueRegistry
from ep2_crypto.execution.venue import (
    OrderBookLevel,
    OrderBookSnapshot,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionInfo,
    VenueAdapter,
    VenueType,
)

# ---------------------------------------------------------------------------
# Concrete test adapter (minimal implementation)
# ---------------------------------------------------------------------------


class FakeAdapter(VenueAdapter):
    """Minimal concrete adapter for testing the ABC contract."""

    def __init__(self, venue: VenueType = VenueType.POLYMARKET_BINARY) -> None:
        self._venue = venue
        self._connected = False

    @property
    def venue_type(self) -> VenueType:
        return self._venue

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def place_order(self, request: OrderRequest) -> OrderResult:
        return OrderResult(
            order_id="fake-001",
            status=OrderStatus.FILLED,
            fill_price=0.52,
            fill_quantity=request.size,
            fee=0.0,
            venue=self._venue,
        )

    async def cancel_order(self, order_id: str) -> OrderResult:
        return OrderResult(
            order_id=order_id,
            status=OrderStatus.CANCELLED,
            venue=self._venue,
        )

    async def get_order(self, order_id: str) -> OrderResult:
        return OrderResult(
            order_id=order_id,
            status=OrderStatus.FILLED,
            fill_price=0.52,
            fill_quantity=10.0,
            venue=self._venue,
        )

    async def get_position(self) -> PositionInfo:
        return PositionInfo(
            venue=self._venue,
            symbol="BTC-UP-5M",
            side=OrderSide.BUY,
            size=10.0,
            entry_price=0.52,
            cost_basis=5.2,
        )

    async def get_orderbook(self) -> OrderBookSnapshot:
        return OrderBookSnapshot(
            venue=self._venue,
            symbol="BTC-UP-5M",
            bids=[OrderBookLevel(0.48, 100.0), OrderBookLevel(0.47, 200.0)],
            asks=[OrderBookLevel(0.52, 100.0), OrderBookLevel(0.53, 200.0)],
            timestamp_ms=1700000000000,
        )

    async def get_balance(self) -> float:
        return 1000.0

    async def health_check(self) -> bool:
        return self._connected


# ---------------------------------------------------------------------------
# ABC enforcement
# ---------------------------------------------------------------------------


class TestVenueAdapterABC:
    def test_cannot_instantiate_abc_directly(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            VenueAdapter()  # type: ignore[abstract]

    def test_incomplete_subclass_raises(self) -> None:
        class IncompleteAdapter(VenueAdapter):
            @property
            def venue_type(self) -> VenueType:
                return VenueType.BINANCE_PERPS

            @property
            def is_connected(self) -> bool:
                return False

            # Missing all other abstract methods

        with pytest.raises(TypeError, match="abstract"):
            IncompleteAdapter()  # type: ignore[abstract]

    def test_complete_subclass_instantiates(self) -> None:
        adapter = FakeAdapter()
        assert isinstance(adapter, VenueAdapter)
        assert adapter.venue_type == VenueType.POLYMARKET_BINARY


# ---------------------------------------------------------------------------
# FakeAdapter contract tests
# ---------------------------------------------------------------------------


class TestFakeAdapterContract:
    def test_connect_disconnect(self) -> None:
        adapter = FakeAdapter()
        assert not adapter.is_connected

        asyncio.get_event_loop().run_until_complete(adapter.connect())
        assert adapter.is_connected

        asyncio.get_event_loop().run_until_complete(adapter.disconnect())
        assert not adapter.is_connected

    def test_place_order(self) -> None:
        adapter = FakeAdapter()
        request = OrderRequest(
            side=OrderSide.BUY,
            size=10.0,
            order_type=OrderType.LIMIT_POST_ONLY,
            price=0.50,
        )
        result = asyncio.get_event_loop().run_until_complete(adapter.place_order(request))
        assert result.is_filled
        assert result.is_terminal
        assert result.order_id == "fake-001"
        assert result.fill_quantity == 10.0

    def test_cancel_order(self) -> None:
        adapter = FakeAdapter()
        result = asyncio.get_event_loop().run_until_complete(adapter.cancel_order("fake-001"))
        assert result.status == OrderStatus.CANCELLED
        assert result.is_terminal

    def test_get_position(self) -> None:
        adapter = FakeAdapter()
        pos = asyncio.get_event_loop().run_until_complete(adapter.get_position())
        assert pos.is_open
        assert pos.side == OrderSide.BUY
        assert pos.size == 10.0

    def test_get_orderbook(self) -> None:
        adapter = FakeAdapter()
        book = asyncio.get_event_loop().run_until_complete(adapter.get_orderbook())
        assert book.best_bid == 0.48
        assert book.best_ask == 0.52
        assert book.mid_price == pytest.approx(0.50)
        assert book.spread == pytest.approx(0.04)

    def test_get_balance(self) -> None:
        adapter = FakeAdapter()
        balance = asyncio.get_event_loop().run_until_complete(adapter.get_balance())
        assert balance == 1000.0

    def test_health_check(self) -> None:
        adapter = FakeAdapter()
        healthy = asyncio.get_event_loop().run_until_complete(adapter.health_check())
        assert not healthy  # Not connected

        asyncio.get_event_loop().run_until_complete(adapter.connect())
        healthy = asyncio.get_event_loop().run_until_complete(adapter.health_check())
        assert healthy


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------


class TestOrderResult:
    def test_filled_is_terminal(self) -> None:
        result = OrderResult(order_id="x", status=OrderStatus.FILLED)
        assert result.is_filled
        assert result.is_terminal

    def test_pending_is_not_terminal(self) -> None:
        result = OrderResult(order_id="x", status=OrderStatus.PENDING)
        assert not result.is_filled
        assert not result.is_terminal

    def test_rejected_is_terminal(self) -> None:
        result = OrderResult(order_id="x", status=OrderStatus.REJECTED)
        assert not result.is_filled
        assert result.is_terminal


class TestPositionInfo:
    def test_no_position(self) -> None:
        pos = PositionInfo(
            venue=VenueType.POLYMARKET_BINARY,
            symbol="BTC-UP-5M",
            side=None,
            size=0.0,
        )
        assert not pos.is_open

    def test_open_position(self) -> None:
        pos = PositionInfo(
            venue=VenueType.POLYMARKET_BINARY,
            symbol="BTC-UP-5M",
            side=OrderSide.BUY,
            size=10.0,
            entry_price=0.50,
        )
        assert pos.is_open


class TestOrderBookSnapshot:
    def test_empty_book(self) -> None:
        book = OrderBookSnapshot(
            venue=VenueType.POLYMARKET_BINARY,
            symbol="BTC-UP-5M",
            bids=[],
            asks=[],
        )
        assert book.best_bid is None
        assert book.best_ask is None
        assert book.mid_price is None
        assert book.spread is None

    def test_one_sided_book(self) -> None:
        book = OrderBookSnapshot(
            venue=VenueType.POLYMARKET_BINARY,
            symbol="BTC-UP-5M",
            bids=[OrderBookLevel(0.48, 100.0)],
            asks=[],
        )
        assert book.best_bid == 0.48
        assert book.best_ask is None
        assert book.mid_price is None

    def test_full_book(self) -> None:
        book = OrderBookSnapshot(
            venue=VenueType.POLYMARKET_BINARY,
            symbol="BTC-UP-5M",
            bids=[OrderBookLevel(0.49, 50.0)],
            asks=[OrderBookLevel(0.51, 50.0)],
        )
        assert book.mid_price == pytest.approx(0.50)
        assert book.spread == pytest.approx(0.02)


class TestOrderRequest:
    def test_market_order(self) -> None:
        req = OrderRequest(side=OrderSide.BUY, size=100.0)
        assert req.order_type == OrderType.MARKET
        assert req.price is None

    def test_limit_order(self) -> None:
        req = OrderRequest(
            side=OrderSide.BUY,
            size=100.0,
            order_type=OrderType.LIMIT,
            price=0.48,
        )
        assert req.price == 0.48


# ---------------------------------------------------------------------------
# VenueType enum
# ---------------------------------------------------------------------------


class TestVenueType:
    def test_values(self) -> None:
        assert VenueType.BINANCE_PERPS == "binance_perps"
        assert VenueType.POLYMARKET_BINARY == "polymarket_binary"


# ---------------------------------------------------------------------------
# VenueRegistry
# ---------------------------------------------------------------------------


class TestVenueRegistry:
    def test_register_and_get(self) -> None:
        registry = VenueRegistry()
        adapter = FakeAdapter(VenueType.POLYMARKET_BINARY)
        registry.register(adapter)

        retrieved = registry.get(VenueType.POLYMARKET_BINARY)
        assert retrieved is adapter

    def test_get_missing_raises(self) -> None:
        registry = VenueRegistry()
        with pytest.raises(KeyError, match="No adapter registered"):
            registry.get(VenueType.BINANCE_PERPS)

    def test_list_venues(self) -> None:
        registry = VenueRegistry()
        assert registry.list_venues() == []

        registry.register(FakeAdapter(VenueType.POLYMARKET_BINARY))
        assert VenueType.POLYMARKET_BINARY in registry.list_venues()

    def test_count(self) -> None:
        registry = VenueRegistry()
        assert registry.count == 0

        registry.register(FakeAdapter(VenueType.POLYMARKET_BINARY))
        assert registry.count == 1

    def test_replace_adapter(self) -> None:
        registry = VenueRegistry()
        adapter1 = FakeAdapter(VenueType.POLYMARKET_BINARY)
        adapter2 = FakeAdapter(VenueType.POLYMARKET_BINARY)

        registry.register(adapter1)
        registry.register(adapter2)

        assert registry.get(VenueType.POLYMARKET_BINARY) is adapter2
        assert registry.count == 1

    def test_multiple_venues(self) -> None:
        registry = VenueRegistry()
        pm = FakeAdapter(VenueType.POLYMARKET_BINARY)
        bn = FakeAdapter(VenueType.BINANCE_PERPS)

        registry.register(pm)
        registry.register(bn)

        assert registry.count == 2
        assert registry.get(VenueType.POLYMARKET_BINARY) is pm
        assert registry.get(VenueType.BINANCE_PERPS) is bn
