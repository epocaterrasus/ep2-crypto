"""Tests for PaperRunner: signal routing, confidence gating, position lifecycle."""

from __future__ import annotations

import pytest

from ep2_crypto.execution.paper_exchange import PaperExchange
from ep2_crypto.execution.paper_runner import PaperRunner, TradeSignal
from ep2_crypto.execution.venue import OrderBookLevel, OrderBookSnapshot, OrderSide, VenueType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_exchange(balance: float = 200_000.0) -> PaperExchange:
    exchange = PaperExchange(initial_balance_usd=balance, seed=0)
    exchange.update_price(100_000.0)
    snapshot = OrderBookSnapshot(
        venue=VenueType.PAPER,
        symbol="BTC/USDT",
        bids=[OrderBookLevel(price=99_990.0, size=10.0)],
        asks=[OrderBookLevel(price=100_010.0, size=10.0)],
        timestamp_ms=0,
    )
    exchange.update_orderbook(snapshot)
    return exchange


def make_runner(threshold: float = 0.60) -> PaperRunner:
    exchange = make_exchange()
    return PaperRunner(exchange=exchange, confidence_threshold=threshold)


def make_signal(
    direction: str = "up",
    confidence: float = 0.75,
    regime: str = "trending",
    size: float = 0.01,
) -> TradeSignal:
    return TradeSignal(
        direction=direction,
        confidence=confidence,
        regime=regime,
        position_size_btc=size,
    )


# ---------------------------------------------------------------------------
# TradeSignal tests
# ---------------------------------------------------------------------------


class TestTradeSignal:
    def test_up_is_tradeable(self) -> None:
        assert TradeSignal("up", 0.8, "trending", 0.01).is_tradeable

    def test_down_is_tradeable(self) -> None:
        assert TradeSignal("down", 0.8, "trending", 0.01).is_tradeable

    def test_flat_not_tradeable(self) -> None:
        assert not TradeSignal("flat", 0.9, "ranging", 0.01).is_tradeable

    def test_timestamp_auto_populated(self) -> None:
        signal = TradeSignal("up", 0.8, "trending", 0.01)
        assert signal.timestamp_ms > 0


# ---------------------------------------------------------------------------
# Confidence gating tests
# ---------------------------------------------------------------------------


class TestConfidenceGating:
    @pytest.mark.asyncio
    async def test_signal_below_threshold_skipped(self) -> None:
        runner = make_runner(threshold=0.70)
        result = await runner.on_signal(make_signal(confidence=0.65))
        assert result is None
        assert runner.get_summary()["signals_skipped"] == 1

    @pytest.mark.asyncio
    async def test_signal_at_threshold_traded(self) -> None:
        runner = make_runner(threshold=0.60)
        await runner.exchange.connect()
        result = await runner.on_signal(make_signal(confidence=0.60))
        assert result is not None

    @pytest.mark.asyncio
    async def test_flat_signal_skipped(self) -> None:
        runner = make_runner()
        result = await runner.on_signal(make_signal(direction="flat", confidence=0.95))
        assert result is None
        assert runner.get_summary()["signals_skipped"] == 1

    @pytest.mark.asyncio
    async def test_same_direction_skipped(self) -> None:
        runner = make_runner()
        await runner.exchange.connect()
        # First signal trades
        await runner.on_signal(make_signal(direction="up"))
        # Second signal in same direction should be skipped
        result = await runner.on_signal(make_signal(direction="up"))
        assert result is None


# ---------------------------------------------------------------------------
# Signal routing tests
# ---------------------------------------------------------------------------


class TestSignalRouting:
    @pytest.mark.asyncio
    async def test_up_signal_places_buy(self) -> None:
        runner = make_runner()
        await runner.exchange.connect()
        result = await runner.on_signal(make_signal(direction="up"))
        assert result is not None
        assert result.is_filled
        # Position should be long
        pos = await runner.exchange.get_position()
        assert pos.side == OrderSide.BUY

    @pytest.mark.asyncio
    async def test_down_signal_places_sell(self) -> None:
        runner = make_runner()
        await runner.exchange.connect()
        result = await runner.on_signal(make_signal(direction="down"))
        assert result is not None
        assert result.is_filled
        pos = await runner.exchange.get_position()
        assert pos.side == OrderSide.SELL

    @pytest.mark.asyncio
    async def test_size_capped_at_max(self) -> None:
        runner = PaperRunner(
            exchange=make_exchange(),
            confidence_threshold=0.5,
            max_position_btc=0.05,
        )
        await runner.exchange.connect()
        # Request 1.0 BTC but max is 0.05
        result = await runner.on_signal(make_signal(size=1.0))
        assert result is not None
        assert result.fill_quantity == pytest.approx(0.05, rel=1e-3)

    @pytest.mark.asyncio
    async def test_order_history_recorded(self) -> None:
        runner = make_runner()
        await runner.exchange.connect()
        await runner.on_signal(make_signal(direction="up"))
        history = runner.order_history
        assert len(history) == 1
        assert history[0]["direction"] == "up"
        assert "fill_price" in history[0]
        assert "confidence" in history[0]


# ---------------------------------------------------------------------------
# Position close tests
# ---------------------------------------------------------------------------


class TestClosePosition:
    @pytest.mark.asyncio
    async def test_close_long_position(self) -> None:
        runner = make_runner()
        await runner.exchange.connect()
        await runner.on_signal(make_signal(direction="up"))
        result = await runner.close_position()
        assert result is not None
        assert result.is_filled
        pos = await runner.exchange.get_position()
        assert not pos.is_open

    @pytest.mark.asyncio
    async def test_close_when_no_position(self) -> None:
        runner = make_runner()
        await runner.exchange.connect()
        result = await runner.close_position()
        assert result is None  # No position to close

    @pytest.mark.asyncio
    async def test_close_resets_direction(self) -> None:
        runner = make_runner()
        await runner.exchange.connect()
        await runner.on_signal(make_signal(direction="up"))
        await runner.close_position()
        # After close, same direction should be tradeable again
        result = await runner.on_signal(make_signal(direction="up"))
        assert result is not None


# ---------------------------------------------------------------------------
# Summary tests
# ---------------------------------------------------------------------------


class TestRunnerSummary:
    @pytest.mark.asyncio
    async def test_summary_counts(self) -> None:
        runner = make_runner()
        await runner.exchange.connect()
        await runner.on_signal(make_signal(confidence=0.50))  # skipped (below 0.60)
        await runner.on_signal(make_signal(confidence=0.80))  # traded
        await runner.on_signal(make_signal(direction="flat"))  # skipped (flat)
        summary = runner.get_summary()
        assert summary["signals_received"] == 3
        assert summary["signals_traded"] == 1
        assert summary["signals_skipped"] == 2

    @pytest.mark.asyncio
    async def test_summary_includes_exchange_data(self) -> None:
        """PaperExchange summary should be merged into runner summary."""
        runner = make_runner()
        await runner.exchange.connect()
        await runner.on_signal(make_signal())
        summary = runner.get_summary()
        # These keys come from PaperExchange.get_summary()
        assert "balance_usd" in summary
        assert "total_trades" in summary
        assert "win_rate" in summary

    @pytest.mark.asyncio
    async def test_trade_rate_calculation(self) -> None:
        runner = make_runner()
        await runner.exchange.connect()
        for _ in range(3):
            await runner.on_signal(make_signal(confidence=0.80, direction="up"))
            await runner.close_position()
        summary = runner.get_summary()
        assert 0.0 < summary["trade_rate"] <= 1.0
