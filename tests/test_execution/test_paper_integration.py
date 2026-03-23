"""Integration tests: full paper trading pipeline from signal to report.

Tests the complete flow:
  1. PaperExchange connects
  2. PaperRunner receives prediction signals
  3. Trades execute via PaperExchange
  4. DailyReportGenerator produces a valid report
  5. VenueRegistry manages exchange registration
"""

from __future__ import annotations

import pytest

from ep2_crypto.execution.paper_exchange import PaperExchange
from ep2_crypto.execution.paper_runner import PaperRunner, TradeSignal
from ep2_crypto.execution.registry import VenueRegistry
from ep2_crypto.execution.venue import OrderBookLevel, OrderBookSnapshot, VenueType
from ep2_crypto.monitoring.daily_report import DailyReportGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_funded_exchange(balance: float = 500_000.0) -> PaperExchange:
    """Create a PaperExchange with orderbook and price pre-loaded."""
    exchange = PaperExchange(initial_balance_usd=balance, seed=0)
    exchange.update_price(100_000.0)
    snapshot = OrderBookSnapshot(
        venue=VenueType.PAPER,
        symbol="BTC/USDT",
        bids=[OrderBookLevel(price=99_990.0, size=100.0)],
        asks=[OrderBookLevel(price=100_010.0, size=100.0)],
        timestamp_ms=0,
    )
    exchange.update_orderbook(snapshot)
    return exchange


# ---------------------------------------------------------------------------
# Full signal → fill → report pipeline
# ---------------------------------------------------------------------------


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_signal_to_fill_to_report(self) -> None:
        """Complete happy path: signal fires, fills, report generated."""
        exchange = make_funded_exchange()
        await exchange.connect()
        runner = PaperRunner(exchange=exchange, confidence_threshold=0.60)

        # Simulate 5 bar closes with alternating up/down signals
        directions = ["up", "down", "up", "down", "up"]
        for direction in directions:
            await runner.close_position()  # Close any open position first
            signal = TradeSignal(
                direction=direction,
                confidence=0.75,
                regime="trending",
                position_size_btc=0.01,
            )
            result = await runner.on_signal(signal)
            assert result is not None
            assert result.is_filled

        # Generate report
        gen = DailyReportGenerator(initial_balance=500_000.0)
        report = gen.generate(trades=exchange.trades, final_balance=await exchange.get_balance())

        assert report.total_trades >= 5
        assert report.total_fees_usd > 0
        assert report.report_date != ""
        assert report.go_nogo_verdict in ("GO", "NO_GO", "INSUFFICIENT_DATA")

    @pytest.mark.asyncio
    async def test_balance_consistent_with_trades(self) -> None:
        """Final balance must equal initial + sum of net trade PnLs."""
        initial = 500_000.0
        exchange = make_funded_exchange(balance=initial)
        await exchange.connect()
        runner = PaperRunner(exchange=exchange, confidence_threshold=0.5)

        # Place a buy and immediately close
        await runner.on_signal(TradeSignal("up", 0.80, "trending", 0.1))
        await runner.close_position()

        final_balance = await exchange.get_balance()
        total_pnl = exchange.total_pnl_usd
        total_fees = exchange.total_fees_usd

        # balance = initial + pnl - fees (approximately, slippage makes it inexact)
        expected = initial + total_pnl - total_fees
        assert abs(final_balance - expected) < 1.0  # within $1

    @pytest.mark.asyncio
    async def test_skipped_signals_not_in_trades(self) -> None:
        """Signals below confidence threshold should not create trades."""
        exchange = make_funded_exchange()
        await exchange.connect()
        runner = PaperRunner(exchange=exchange, confidence_threshold=0.80)

        # Low confidence signal — should be skipped
        await runner.on_signal(TradeSignal("up", 0.70, "trending", 0.01))
        assert len(exchange.trades) == 0

        # High confidence signal — should trade
        await runner.on_signal(TradeSignal("up", 0.85, "trending", 0.01))
        assert len(exchange.trades) == 1


# ---------------------------------------------------------------------------
# VenueRegistry integration
# ---------------------------------------------------------------------------


class TestVenueRegistryIntegration:
    @pytest.mark.asyncio
    async def test_paper_exchange_registered_and_retrieved(self) -> None:
        exchange = make_funded_exchange()
        await exchange.connect()
        registry = VenueRegistry()
        registry.register(exchange)
        retrieved = registry.get(VenueType.PAPER)
        assert retrieved is exchange

    @pytest.mark.asyncio
    async def test_registry_lists_venues(self) -> None:
        registry = VenueRegistry()
        exchange = make_funded_exchange()
        await exchange.connect()
        registry.register(exchange)
        assert VenueType.PAPER in registry.list_venues()

    def test_registry_raises_for_unknown_venue(self) -> None:
        registry = VenueRegistry()
        with pytest.raises(KeyError):
            registry.get(VenueType.BINANCE_PERPS)


# ---------------------------------------------------------------------------
# Multi-session simulation (paper trading acceptance criteria check)
# ---------------------------------------------------------------------------


class TestPaperTradingSession:
    @pytest.mark.asyncio
    async def test_session_produces_enough_data_for_report(self) -> None:
        """Simulate a paper trading session with enough trades for a report."""
        exchange = make_funded_exchange(balance=1_000_000.0)
        await exchange.connect()
        runner = PaperRunner(exchange=exchange, confidence_threshold=0.50)

        # Simulate 50 alternating bar closes
        for i in range(50):
            await runner.close_position()
            direction = "up" if i % 2 == 0 else "down"
            await runner.on_signal(
                TradeSignal(
                    direction=direction,
                    confidence=0.70,
                    regime="trending" if i % 3 == 0 else "ranging",
                    position_size_btc=0.01,
                )
            )
        await runner.close_position()

        assert len(exchange.trades) >= 50

        gen = DailyReportGenerator(initial_balance=1_000_000.0)
        report = gen.generate(
            trades=exchange.trades,
            final_balance=await exchange.get_balance(),
        )
        assert report.total_trades >= 50
        assert report.total_fees_usd > 0
        # For INSUFFICIENT_DATA we need 200+ trades — still verifiable
        assert report.go_nogo_verdict in ("GO", "NO_GO", "INSUFFICIENT_DATA")

    @pytest.mark.asyncio
    async def test_position_closed_on_shutdown(self) -> None:
        """Simulates the shutdown path where close_position is called."""
        exchange = make_funded_exchange()
        await exchange.connect()
        runner = PaperRunner(exchange=exchange, confidence_threshold=0.5)

        await runner.on_signal(TradeSignal("up", 0.80, "trending", 0.01))
        pos_before = await exchange.get_position()
        assert pos_before.is_open

        await runner.close_position()
        pos_after = await exchange.get_position()
        assert not pos_after.is_open

    @pytest.mark.asyncio
    async def test_summary_matches_exchange_state(self) -> None:
        """Runner summary should reflect PaperExchange state."""
        exchange = make_funded_exchange()
        await exchange.connect()
        runner = PaperRunner(exchange=exchange, confidence_threshold=0.5)

        await runner.on_signal(TradeSignal("up", 0.80, "trending", 0.02))
        await runner.close_position()

        summary = runner.get_summary()
        assert summary["total_trades"] == len(exchange.trades)
        assert summary["signals_traded"] == 1  # only on_signal() counts; close_position() doesn't
        assert "balance_usd" in summary
