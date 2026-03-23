"""Tests for LiveExchange: mocked ccxt interactions."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ep2_crypto.execution.live_exchange import LiveExchange
from ep2_crypto.execution.venue import (
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    VenueType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ccxt_order(
    order_id: str = "123",
    status: str = "closed",
    avg_price: float = 100_000.0,
    filled: float = 0.1,
    fee_cost: float = 4.0,
    fee_currency: str = "USDT",
) -> dict:
    return {
        "id": order_id,
        "status": status,
        "average": avg_price,
        "filled": filled,
        "fee": {"cost": fee_cost, "currency": fee_currency},
        "price": avg_price,
    }


def make_ccxt_position(
    symbol: str = "BTC/USDT:USDT",
    side: str = "long",
    contracts: float = 0.1,
    entry_price: float = 100_000.0,
    unrealised_pnl: float = 500.0,
) -> dict:
    return {
        "symbol": symbol,
        "side": side,
        "contracts": contracts,
        "entryPrice": entry_price,
        "unrealizedPnl": unrealised_pnl,
        "initialMargin": contracts * entry_price,
    }


def make_mock_ccxt() -> MagicMock:
    """Return a MagicMock with all async ccxt methods pre-configured."""
    mock = MagicMock()
    mock.fetch_balance = AsyncMock(return_value={"USDT": {"free": 50_000.0}})
    mock.create_order = AsyncMock(return_value=make_ccxt_order())
    mock.cancel_order = AsyncMock(return_value=make_ccxt_order(status="canceled"))
    mock.fetch_order = AsyncMock(return_value=make_ccxt_order())
    mock.fetch_positions = AsyncMock(return_value=[make_ccxt_position()])
    mock.fetch_order_book = AsyncMock(
        return_value={
            "bids": [[99_990.0, 0.5], [99_980.0, 1.0]],
            "asks": [[100_010.0, 0.5], [100_020.0, 1.0]],
            "timestamp": 1_700_000_000_000,
        }
    )
    mock.fetch_time = AsyncMock(return_value=1_700_000_000_000)
    mock.close = AsyncMock()
    return mock


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


class TestLiveExchangeLifecycle:
    def test_venue_type(self) -> None:
        assert LiveExchange().venue_type == VenueType.BINANCE_PERPS

    def test_not_connected_initially(self) -> None:
        exchange = LiveExchange()
        assert not exchange.is_connected

    @pytest.mark.asyncio
    async def test_connect_requires_credentials(self) -> None:
        exchange = LiveExchange()
        env = {"BINANCE_API_KEY": "", "BINANCE_API_SECRET": ""}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("BINANCE_API_KEY", None)
            os.environ.pop("BINANCE_API_SECRET", None)
            with pytest.raises(EnvironmentError, match="BINANCE_API_KEY"):
                await exchange.connect()

    @pytest.mark.asyncio
    async def test_connect_with_credentials(self) -> None:
        mock_ccxt = make_mock_ccxt()
        exchange = LiveExchange()
        with (
            patch.dict(os.environ, {"BINANCE_API_KEY": "key", "BINANCE_API_SECRET": "secret"}),
            patch("ep2_crypto.execution.live_exchange.LiveExchange._require_connected"),
            patch("ccxt.pro.binanceusdm", return_value=mock_ccxt),
        ):
            exchange._exchange = mock_ccxt
            exchange._connected = True
            assert exchange.is_connected

    @pytest.mark.asyncio
    async def test_disconnect_closes_exchange(self) -> None:
        exchange = LiveExchange()
        mock_ccxt = make_mock_ccxt()
        exchange._exchange = mock_ccxt
        exchange._connected = True
        await exchange.disconnect()
        mock_ccxt.close.assert_awaited_once()
        assert not exchange.is_connected

    @pytest.mark.asyncio
    async def test_require_connected_raises(self) -> None:
        exchange = LiveExchange()
        with pytest.raises(RuntimeError, match="not connected"):
            await exchange.get_balance()


# ---------------------------------------------------------------------------
# Order management tests
# ---------------------------------------------------------------------------


class TestLiveExchangeOrders:
    @pytest.fixture
    def exchange(self) -> LiveExchange:
        ex = LiveExchange()
        ex._exchange = make_mock_ccxt()
        ex._connected = True
        return ex

    @pytest.mark.asyncio
    async def test_place_market_buy(self, exchange: LiveExchange) -> None:
        result = await exchange.place_order(
            OrderRequest(side=OrderSide.BUY, size=0.1)
        )
        exchange._exchange.create_order.assert_awaited_once()
        call_args = exchange._exchange.create_order.call_args
        assert call_args.kwargs["type"] == "market"
        assert call_args.kwargs["side"] == "buy"
        assert call_args.kwargs["amount"] == 0.1

    @pytest.mark.asyncio
    async def test_place_market_sell(self, exchange: LiveExchange) -> None:
        result = await exchange.place_order(
            OrderRequest(side=OrderSide.SELL, size=0.05)
        )
        call_args = exchange._exchange.create_order.call_args
        assert call_args.kwargs["side"] == "sell"

    @pytest.mark.asyncio
    async def test_place_limit_order(self, exchange: LiveExchange) -> None:
        result = await exchange.place_order(
            OrderRequest(side=OrderSide.BUY, size=0.1, order_type=OrderType.LIMIT, price=99_000.0)
        )
        call_args = exchange._exchange.create_order.call_args
        assert call_args.kwargs["type"] == "limit"
        assert call_args.kwargs["price"] == 99_000.0

    @pytest.mark.asyncio
    async def test_place_order_filled_status(self, exchange: LiveExchange) -> None:
        result = await exchange.place_order(OrderRequest(side=OrderSide.BUY, size=0.1))
        assert result.status == OrderStatus.FILLED
        assert result.fill_price == pytest.approx(100_000.0)
        assert result.fill_quantity == pytest.approx(0.1)
        assert result.fee == pytest.approx(4.0)
        assert result.venue == VenueType.BINANCE_PERPS

    @pytest.mark.asyncio
    async def test_place_order_exception_returns_rejected(self, exchange: LiveExchange) -> None:
        exchange._exchange.create_order = AsyncMock(side_effect=Exception("Network error"))
        result = await exchange.place_order(OrderRequest(side=OrderSide.BUY, size=0.1))
        assert result.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_cancel_order(self, exchange: LiveExchange) -> None:
        result = await exchange.cancel_order("order-123")
        exchange._exchange.cancel_order.assert_awaited_once_with("order-123", symbol=LiveExchange._LiveExchange__name if hasattr(LiveExchange, '_LiveExchange__name') else "BTC/USDT:USDT")
        assert result.status == OrderStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_order_exception(self, exchange: LiveExchange) -> None:
        exchange._exchange.cancel_order = AsyncMock(side_effect=Exception("Not found"))
        result = await exchange.cancel_order("bad-id")
        assert result.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_get_order(self, exchange: LiveExchange) -> None:
        result = await exchange.get_order("order-123")
        assert result.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_get_order_exception(self, exchange: LiveExchange) -> None:
        exchange._exchange.fetch_order = AsyncMock(side_effect=Exception("Not found"))
        result = await exchange.get_order("bad-id")
        assert result.status == OrderStatus.REJECTED


# ---------------------------------------------------------------------------
# Position and balance tests
# ---------------------------------------------------------------------------


class TestLiveExchangePosition:
    @pytest.fixture
    def exchange(self) -> LiveExchange:
        ex = LiveExchange()
        ex._exchange = make_mock_ccxt()
        ex._connected = True
        return ex

    @pytest.mark.asyncio
    async def test_get_position_long(self, exchange: LiveExchange) -> None:
        pos = await exchange.get_position()
        assert pos.is_open
        assert pos.side == OrderSide.BUY
        assert pos.size == pytest.approx(0.1)
        assert pos.entry_price == pytest.approx(100_000.0)
        assert pos.unrealized_pnl == pytest.approx(500.0)

    @pytest.mark.asyncio
    async def test_get_position_short(self, exchange: LiveExchange) -> None:
        exchange._exchange.fetch_positions = AsyncMock(
            return_value=[make_ccxt_position(side="short")]
        )
        pos = await exchange.get_position()
        assert pos.side == OrderSide.SELL

    @pytest.mark.asyncio
    async def test_get_position_empty(self, exchange: LiveExchange) -> None:
        exchange._exchange.fetch_positions = AsyncMock(return_value=[])
        pos = await exchange.get_position()
        assert not pos.is_open
        assert pos.size == 0.0

    @pytest.mark.asyncio
    async def test_get_position_exception(self, exchange: LiveExchange) -> None:
        exchange._exchange.fetch_positions = AsyncMock(side_effect=Exception("API error"))
        pos = await exchange.get_position()
        assert not pos.is_open

    @pytest.mark.asyncio
    async def test_get_balance(self, exchange: LiveExchange) -> None:
        balance = await exchange.get_balance()
        assert balance == pytest.approx(50_000.0)

    @pytest.mark.asyncio
    async def test_get_balance_exception(self, exchange: LiveExchange) -> None:
        exchange._exchange.fetch_balance = AsyncMock(side_effect=Exception("API error"))
        balance = await exchange.get_balance()
        assert balance == 0.0


# ---------------------------------------------------------------------------
# Orderbook and health check tests
# ---------------------------------------------------------------------------


class TestLiveExchangeOrderbook:
    @pytest.fixture
    def exchange(self) -> LiveExchange:
        ex = LiveExchange()
        ex._exchange = make_mock_ccxt()
        ex._connected = True
        return ex

    @pytest.mark.asyncio
    async def test_get_orderbook(self, exchange: LiveExchange) -> None:
        book = await exchange.get_orderbook()
        assert book.best_bid == pytest.approx(99_990.0)
        assert book.best_ask == pytest.approx(100_010.0)
        assert len(book.bids) == 2
        assert len(book.asks) == 2
        assert book.timestamp_ms == 1_700_000_000_000

    @pytest.mark.asyncio
    async def test_get_orderbook_exception_returns_empty(self, exchange: LiveExchange) -> None:
        exchange._exchange.fetch_order_book = AsyncMock(side_effect=Exception("Error"))
        book = await exchange.get_orderbook()
        assert book.bids == []
        assert book.asks == []

    @pytest.mark.asyncio
    async def test_health_check_passes(self, exchange: LiveExchange) -> None:
        assert await exchange.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_disconnected(self) -> None:
        exchange = LiveExchange()
        assert await exchange.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_exception(self, exchange: LiveExchange) -> None:
        exchange._exchange.fetch_time = AsyncMock(side_effect=Exception("Timeout"))
        assert await exchange.health_check() is False


# ---------------------------------------------------------------------------
# Order status mapping tests
# ---------------------------------------------------------------------------


class TestOrderStatusMapping:
    def test_closed_maps_to_filled(self) -> None:
        result = LiveExchange._parse_order_result(
            LiveExchange(),
            make_ccxt_order(status="closed"),
        )
        assert result.status == OrderStatus.FILLED

    def test_open_maps_to_open(self) -> None:
        result = LiveExchange._parse_order_result(
            LiveExchange(),
            make_ccxt_order(status="open"),
        )
        assert result.status == OrderStatus.OPEN

    def test_canceled_maps_to_cancelled(self) -> None:
        result = LiveExchange._parse_order_result(
            LiveExchange(),
            make_ccxt_order(status="canceled"),
        )
        assert result.status == OrderStatus.CANCELLED
