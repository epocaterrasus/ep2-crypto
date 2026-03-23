"""Tests for PaperExchange: fill simulation, balance, position lifecycle."""

from __future__ import annotations

import pytest

from ep2_crypto.execution.paper_exchange import FillSimulator, PaperExchange, PaperPosition
from ep2_crypto.execution.venue import (
    OrderBookLevel,
    OrderBookSnapshot,
    OrderRequest,
    OrderSide,
    OrderStatus,
    VenueType,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_snapshot(
    bid: float = 99_990.0, ask: float = 100_010.0, depth: int = 5
) -> OrderBookSnapshot:
    """Create a realistic BTC orderbook snapshot."""
    bids = [OrderBookLevel(price=bid - i * 10, size=0.5) for i in range(depth)]
    asks = [OrderBookLevel(price=ask + i * 10, size=0.5) for i in range(depth)]
    return OrderBookSnapshot(
        venue=VenueType.PAPER,
        symbol="BTC/USDT",
        bids=bids,
        asks=asks,
        timestamp_ms=1_700_000_000_000,
    )


def make_exchange(balance: float = 200_000.0) -> PaperExchange:
    exchange = PaperExchange(initial_balance_usd=balance, seed=0)
    exchange.update_price(100_000.0)
    exchange.update_orderbook(make_snapshot())
    return exchange


# ---------------------------------------------------------------------------
# FillSimulator tests
# ---------------------------------------------------------------------------


class TestFillSimulator:
    def test_buy_walks_asks(self) -> None:
        sim = FillSimulator(slippage_noise_bps=0.0)
        snap = make_snapshot(ask=100_010.0)
        price, filled, unfilled = sim.simulate(
            OrderRequest(side=OrderSide.BUY, size=0.3),
            snap,
        )
        assert filled == pytest.approx(0.3, rel=1e-6)
        assert unfilled == pytest.approx(0.0, abs=1e-9)
        assert price >= 100_010.0  # at or above best ask

    def test_sell_walks_bids(self) -> None:
        sim = FillSimulator(slippage_noise_bps=0.0)
        snap = make_snapshot(bid=99_990.0)
        price, filled, unfilled = sim.simulate(
            OrderRequest(side=OrderSide.SELL, size=0.3),
            snap,
        )
        assert filled == pytest.approx(0.3, rel=1e-6)
        assert unfilled == pytest.approx(0.0, abs=1e-9)
        assert price <= 99_990.0  # at or below best bid

    def test_partial_fill_when_order_exceeds_book(self) -> None:
        sim = FillSimulator(slippage_noise_bps=0.0)
        # Book only has 5 * 0.5 = 2.5 BTC on ask side
        snap = make_snapshot(depth=5)
        _price, filled, unfilled = sim.simulate(
            OrderRequest(side=OrderSide.BUY, size=10.0),
            snap,
        )
        assert filled == pytest.approx(2.5, rel=1e-6)
        assert unfilled == pytest.approx(7.5, rel=1e-6)

    def test_empty_book_returns_zero_fill(self) -> None:
        sim = FillSimulator()
        snap = OrderBookSnapshot(
            venue=VenueType.PAPER, symbol="BTC/USDT", bids=[], asks=[], timestamp_ms=0
        )
        _price, filled, unfilled = sim.simulate(OrderRequest(side=OrderSide.BUY, size=0.1), snap)
        assert filled == 0.0
        assert unfilled == 0.1

    def test_slippage_noise_applied(self) -> None:
        sim = FillSimulator(slippage_noise_bps=10.0)
        snap = make_snapshot(ask=100_000.0, depth=1)
        price, _filled, _ = sim.simulate(OrderRequest(side=OrderSide.BUY, size=0.5), snap)
        # Should be ≥ best ask due to noise
        assert price >= 100_000.0

    def test_avg_price_weighted_across_levels(self) -> None:
        sim = FillSimulator(slippage_noise_bps=0.0)
        asks = [
            OrderBookLevel(price=100_000.0, size=1.0),
            OrderBookLevel(price=101_000.0, size=1.0),
        ]
        snap = OrderBookSnapshot(
            venue=VenueType.PAPER,
            symbol="BTC/USDT",
            bids=[OrderBookLevel(price=99_000.0, size=2.0)],
            asks=asks,
            timestamp_ms=0,
        )
        price, filled, _unfilled = sim.simulate(OrderRequest(side=OrderSide.BUY, size=2.0), snap)
        assert filled == pytest.approx(2.0)
        assert price == pytest.approx(100_500.0, rel=1e-6)  # (100k + 101k) / 2


# ---------------------------------------------------------------------------
# PaperPosition tests
# ---------------------------------------------------------------------------


class TestPaperPosition:
    def test_unrealised_pnl_long(self) -> None:
        pos = PaperPosition(side=OrderSide.BUY, size=1.0, entry_price=100_000.0)
        pos.update_unrealised(101_000.0)
        assert pos.unrealised_pnl == pytest.approx(1_000.0)

    def test_unrealised_pnl_short(self) -> None:
        pos = PaperPosition(side=OrderSide.SELL, size=1.0, entry_price=100_000.0)
        pos.update_unrealised(99_000.0)
        assert pos.unrealised_pnl == pytest.approx(1_000.0)

    def test_no_position_zero_pnl(self) -> None:
        pos = PaperPosition()
        pos.update_unrealised(100_000.0)
        assert pos.unrealised_pnl == 0.0

    def test_is_open(self) -> None:
        pos = PaperPosition(side=OrderSide.BUY, size=1.0, entry_price=100_000.0)
        assert pos.is_open is True
        pos.size = 0.0
        assert pos.is_open is False


# ---------------------------------------------------------------------------
# PaperExchange lifecycle tests
# ---------------------------------------------------------------------------


class TestPaperExchangeLifecycle:
    @pytest.mark.asyncio
    async def test_connect_disconnect(self) -> None:
        exchange = PaperExchange()
        assert not exchange.is_connected
        await exchange.connect()
        assert exchange.is_connected
        await exchange.disconnect()
        assert not exchange.is_connected

    @pytest.mark.asyncio
    async def test_initial_balance(self) -> None:
        exchange = PaperExchange(initial_balance_usd=50_000.0)
        await exchange.connect()
        balance = await exchange.get_balance()
        assert balance == pytest.approx(50_000.0)

    @pytest.mark.asyncio
    async def test_reject_when_not_connected(self) -> None:
        exchange = PaperExchange()
        exchange.update_price(100_000.0)
        exchange.update_orderbook(make_snapshot())
        result = await exchange.place_order(OrderRequest(side=OrderSide.BUY, size=0.1))
        assert result.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_reject_no_price_data(self) -> None:
        exchange = PaperExchange()
        await exchange.connect()
        result = await exchange.place_order(OrderRequest(side=OrderSide.BUY, size=0.1))
        assert result.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_health_check(self) -> None:
        exchange = PaperExchange()
        await exchange.connect()
        assert await exchange.health_check() is True
        await exchange.disconnect()
        assert await exchange.health_check() is False

    @pytest.mark.asyncio
    async def test_venue_type(self) -> None:
        assert PaperExchange().venue_type == VenueType.PAPER


# ---------------------------------------------------------------------------
# Order placement and fill tests
# ---------------------------------------------------------------------------


class TestPaperExchangeOrders:
    @pytest.mark.asyncio
    async def test_buy_order_fills(self) -> None:
        exchange = make_exchange(balance=10_000.0)
        await exchange.connect()
        result = await exchange.place_order(OrderRequest(side=OrderSide.BUY, size=0.05))
        assert result.status == OrderStatus.FILLED
        assert result.fill_quantity == pytest.approx(0.05, rel=1e-6)
        assert result.fill_price > 0
        assert result.fee > 0

    @pytest.mark.asyncio
    async def test_sell_order_fills(self) -> None:
        exchange = make_exchange(balance=10_000.0)
        await exchange.connect()
        # First open a long
        await exchange.place_order(OrderRequest(side=OrderSide.BUY, size=0.05))
        # Then sell
        result = await exchange.place_order(OrderRequest(side=OrderSide.SELL, size=0.05))
        assert result.status == OrderStatus.FILLED
        assert result.fill_price > 0

    @pytest.mark.asyncio
    async def test_balance_decreases_on_buy(self) -> None:
        exchange = make_exchange(balance=10_000.0)
        await exchange.connect()
        before = await exchange.get_balance()
        await exchange.place_order(OrderRequest(side=OrderSide.BUY, size=0.05))
        after = await exchange.get_balance()
        assert after < before

    @pytest.mark.asyncio
    async def test_reject_insufficient_balance(self) -> None:
        exchange = PaperExchange(initial_balance_usd=100.0, seed=0)  # Only $100
        exchange.update_price(100_000.0)
        exchange.update_orderbook(make_snapshot())
        await exchange.connect()
        result = await exchange.place_order(
            OrderRequest(side=OrderSide.BUY, size=10.0)  # $1M order
        )
        assert result.status == OrderStatus.REJECTED

    @pytest.mark.asyncio
    async def test_fee_applied_on_buy(self) -> None:
        exchange = PaperExchange(
            initial_balance_usd=200_000.0, taker_fee_bps=4.0, slippage_noise_bps=0.0
        )
        exchange.update_price(100_000.0)
        exchange.update_orderbook(make_snapshot(ask=100_010.0, depth=10))
        await exchange.connect()
        result = await exchange.place_order(OrderRequest(side=OrderSide.BUY, size=0.1))
        assert result.status == OrderStatus.FILLED
        expected_notional = result.fill_price * result.fill_quantity
        expected_fee = expected_notional * 4.0 * 1e-4
        assert result.fee == pytest.approx(expected_fee, rel=1e-4)

    @pytest.mark.asyncio
    async def test_cancel_filled_order(self) -> None:
        exchange = make_exchange()
        await exchange.connect()
        result = await exchange.place_order(OrderRequest(side=OrderSide.BUY, size=0.05))
        cancelled = await exchange.cancel_order(result.order_id)
        # Paper orders fill instantly — can't cancel a filled order
        assert cancelled.status == OrderStatus.FILLED

    @pytest.mark.asyncio
    async def test_get_order_returns_result(self) -> None:
        exchange = make_exchange()
        await exchange.connect()
        result = await exchange.place_order(OrderRequest(side=OrderSide.BUY, size=0.05))
        fetched = await exchange.get_order(result.order_id)
        assert fetched.order_id == result.order_id
        assert fetched.status == result.status

    @pytest.mark.asyncio
    async def test_get_order_unknown_id(self) -> None:
        exchange = make_exchange()
        await exchange.connect()
        result = await exchange.get_order("nonexistent-id")
        assert result.status == OrderStatus.REJECTED


# ---------------------------------------------------------------------------
# Position tracking tests
# ---------------------------------------------------------------------------


class TestPaperExchangePosition:
    @pytest.mark.asyncio
    async def test_no_position_initially(self) -> None:
        exchange = make_exchange()
        await exchange.connect()
        pos = await exchange.get_position()
        assert not pos.is_open
        assert pos.size == 0.0

    @pytest.mark.asyncio
    async def test_long_position_after_buy(self) -> None:
        exchange = make_exchange()
        await exchange.connect()
        await exchange.place_order(OrderRequest(side=OrderSide.BUY, size=0.1))
        pos = await exchange.get_position()
        assert pos.is_open
        assert pos.side == OrderSide.BUY
        assert pos.size == pytest.approx(0.1, rel=1e-6)

    @pytest.mark.asyncio
    async def test_position_closes_after_sell(self) -> None:
        exchange = make_exchange()
        await exchange.connect()
        await exchange.place_order(OrderRequest(side=OrderSide.BUY, size=0.1))
        await exchange.place_order(OrderRequest(side=OrderSide.SELL, size=0.1))
        pos = await exchange.get_position()
        assert not pos.is_open

    @pytest.mark.asyncio
    async def test_short_position(self) -> None:
        exchange = make_exchange()
        await exchange.connect()
        await exchange.place_order(OrderRequest(side=OrderSide.SELL, size=0.1))
        pos = await exchange.get_position()
        assert pos.is_open
        assert pos.side == OrderSide.SELL

    @pytest.mark.asyncio
    async def test_position_flip_long_to_short(self) -> None:
        exchange = make_exchange()
        await exchange.connect()
        await exchange.place_order(OrderRequest(side=OrderSide.BUY, size=0.1))
        await exchange.place_order(OrderRequest(side=OrderSide.SELL, size=0.2))
        pos = await exchange.get_position()
        assert pos.side == OrderSide.SELL
        assert pos.size == pytest.approx(0.1, rel=1e-3)

    @pytest.mark.asyncio
    async def test_unrealised_pnl_updates(self) -> None:
        exchange = make_exchange(balance=200_000.0)
        await exchange.connect()
        await exchange.place_order(OrderRequest(side=OrderSide.BUY, size=1.0))
        exchange.update_price(101_000.0)
        pos = await exchange.get_position()
        assert pos.unrealized_pnl > 0


# ---------------------------------------------------------------------------
# PnL and summary tests
# ---------------------------------------------------------------------------


class TestPaperExchangeSummary:
    @pytest.mark.asyncio
    async def test_summary_after_round_trip(self) -> None:
        exchange = make_exchange(balance=10_000.0)
        await exchange.connect()
        await exchange.place_order(OrderRequest(side=OrderSide.BUY, size=0.05))
        await exchange.place_order(OrderRequest(side=OrderSide.SELL, size=0.05))
        summary = exchange.get_summary()
        assert summary["total_trades"] == 2
        assert summary["balance_usd"] > 0

    @pytest.mark.asyncio
    async def test_trades_list_grows(self) -> None:
        exchange = make_exchange()
        await exchange.connect()
        await exchange.place_order(OrderRequest(side=OrderSide.BUY, size=0.05))
        await exchange.place_order(OrderRequest(side=OrderSide.BUY, size=0.03))
        assert len(exchange.trades) == 2

    @pytest.mark.asyncio
    async def test_fees_are_positive(self) -> None:
        exchange = make_exchange()
        await exchange.connect()
        await exchange.place_order(OrderRequest(side=OrderSide.BUY, size=0.05))
        assert exchange.total_fees_usd > 0

    @pytest.mark.asyncio
    async def test_win_rate_in_summary(self) -> None:
        exchange = make_exchange()
        await exchange.connect()
        summary = exchange.get_summary()
        assert 0.0 <= summary["win_rate"] <= 1.0

    @pytest.mark.asyncio
    async def test_orderbook_update_reflected(self) -> None:
        exchange = PaperExchange()
        await exchange.connect()
        snap = make_snapshot(bid=95_000.0, ask=95_010.0)
        exchange.update_orderbook(snap)
        book = await exchange.get_orderbook()
        assert book.best_bid == pytest.approx(95_000.0)
        assert book.best_ask == pytest.approx(95_010.0)

    @pytest.mark.asyncio
    async def test_synthetic_book_used_when_no_snapshot(self) -> None:
        """When no orderbook snapshot is available, uses synthetic book from last_price."""
        exchange = PaperExchange(initial_balance_usd=100_000.0)
        await exchange.connect()
        exchange.update_price(100_000.0)
        # No snapshot fed — should still fill using synthetic book
        result = await exchange.place_order(OrderRequest(side=OrderSide.BUY, size=0.01))
        assert result.status == OrderStatus.FILLED
