"""Tests for execution quality tracker (A/B test)."""

from __future__ import annotations

import pytest

from ep2_crypto.execution.quality_tracker import (
    ExecutionQualityTracker,
    ExecutionRecord,
    ExecutionStrategy,
    StrategyStats,
    _compute_strategy_stats,
)


def make_record(
    strategy: ExecutionStrategy = ExecutionStrategy.MARKET,
    filled: bool = True,
    slippage_bps: float = 3.0,
    cost_bps: float = 5.0,
    timestamp_ms: int = 1000,
    size_usd: float = 5000.0,
    latency_ms: float = 50.0,
) -> ExecutionRecord:
    return ExecutionRecord(
        strategy=strategy,
        timestamp_ms=timestamp_ms,
        filled=filled,
        slippage_bps=slippage_bps,
        cost_bps=cost_bps,
        size_usd=size_usd,
        latency_ms=latency_ms,
    )


class TestComputeStrategyStats:
    def test_empty_records(self) -> None:
        stats = _compute_strategy_stats(ExecutionStrategy.MARKET, [])
        assert stats.trade_count == 0
        assert stats.fill_rate == 0.0

    def test_all_filled(self) -> None:
        records = [make_record(slippage_bps=i * 1.0, cost_bps=5.0) for i in range(10)]
        stats = _compute_strategy_stats(ExecutionStrategy.MARKET, records)
        assert stats.trade_count == 10
        assert stats.fill_rate == 1.0
        assert stats.mean_cost_bps == 5.0

    def test_partial_fills(self) -> None:
        records = [
            make_record(filled=True, slippage_bps=3.0),
            make_record(filled=True, slippage_bps=4.0),
            make_record(filled=False, slippage_bps=0.0),
        ]
        stats = _compute_strategy_stats(ExecutionStrategy.LIMIT_IOC, records)
        assert stats.trade_count == 3
        assert stats.fill_rate == pytest.approx(2 / 3)
        assert stats.mean_slippage_bps == pytest.approx(3.5)  # Only filled trades


class TestExecutionQualityTracker:
    @pytest.fixture()
    def tracker(self) -> ExecutionQualityTracker:
        return ExecutionQualityTracker(min_trades_per_arm=5)

    def test_initial_strategy_is_market(self, tracker: ExecutionQualityTracker) -> None:
        assert tracker.get_strategy() == ExecutionStrategy.MARKET

    def test_alternates_during_testing(self, tracker: ExecutionQualityTracker) -> None:
        # First trade: market (0 market, 0 limit → market)
        assert tracker.get_strategy() == ExecutionStrategy.MARKET
        tracker.record_execution(make_record(strategy=ExecutionStrategy.MARKET))

        # Second trade: limit (1 market, 0 limit → limit)
        assert tracker.get_strategy() == ExecutionStrategy.LIMIT_IOC
        tracker.record_execution(make_record(strategy=ExecutionStrategy.LIMIT_IOC))

        # Third trade: market again (1 market, 1 limit → market)
        assert tracker.get_strategy() == ExecutionStrategy.MARKET

    def test_record_execution(self, tracker: ExecutionQualityTracker) -> None:
        tracker.record_execution(make_record(strategy=ExecutionStrategy.MARKET))
        assert tracker.get_market_stats().trade_count == 1
        assert tracker.get_limit_stats().trade_count == 0

    def test_auto_select_market_wins(self, tracker: ExecutionQualityTracker) -> None:
        # Market: lower cost
        for i in range(5):
            tracker.record_execution(make_record(
                strategy=ExecutionStrategy.MARKET,
                cost_bps=4.0,
                slippage_bps=2.0,
            ))
        # Limit IOC: higher cost + some unfilled
        for i in range(5):
            tracker.record_execution(make_record(
                strategy=ExecutionStrategy.LIMIT_IOC,
                cost_bps=6.0,
                slippage_bps=1.0,
                filled=i < 3,  # 60% fill rate
            ))
        assert tracker.auto_selected
        assert tracker.active_strategy == ExecutionStrategy.MARKET

    def test_auto_select_limit_wins(self, tracker: ExecutionQualityTracker) -> None:
        # Market: high cost
        for i in range(5):
            tracker.record_execution(make_record(
                strategy=ExecutionStrategy.MARKET,
                cost_bps=8.0,
                slippage_bps=5.0,
            ))
        # Limit IOC: low cost, high fill rate
        for i in range(5):
            tracker.record_execution(make_record(
                strategy=ExecutionStrategy.LIMIT_IOC,
                cost_bps=3.0,
                slippage_bps=0.5,
                filled=True,
            ))
        assert tracker.auto_selected
        assert tracker.active_strategy == ExecutionStrategy.LIMIT_IOC

    def test_no_auto_select_before_min_trades(
        self, tracker: ExecutionQualityTracker
    ) -> None:
        for i in range(4):
            tracker.record_execution(make_record(strategy=ExecutionStrategy.MARKET))
            tracker.record_execution(make_record(strategy=ExecutionStrategy.LIMIT_IOC))
        assert not tracker.auto_selected

    def test_strategy_fixed_after_auto_select(
        self, tracker: ExecutionQualityTracker
    ) -> None:
        for i in range(5):
            tracker.record_execution(make_record(
                strategy=ExecutionStrategy.MARKET, cost_bps=4.0
            ))
            tracker.record_execution(make_record(
                strategy=ExecutionStrategy.LIMIT_IOC, cost_bps=10.0
            ))
        assert tracker.auto_selected
        # Strategy should be fixed now
        selected = tracker.active_strategy
        for _ in range(10):
            assert tracker.get_strategy() == selected

    def test_selection_reason_populated(
        self, tracker: ExecutionQualityTracker
    ) -> None:
        for i in range(5):
            tracker.record_execution(make_record(strategy=ExecutionStrategy.MARKET))
            tracker.record_execution(make_record(strategy=ExecutionStrategy.LIMIT_IOC))
        assert tracker.selection_reason != ""

    def test_get_summary(self, tracker: ExecutionQualityTracker) -> None:
        for i in range(3):
            tracker.record_execution(make_record(
                strategy=ExecutionStrategy.MARKET, cost_bps=5.0
            ))
        summary = tracker.get_summary()
        assert summary["active_strategy"] == "market"
        assert summary["market"]["trades"] == 3
        assert summary["limit_ioc"]["trades"] == 0
        assert "auto_selected" in summary

    def test_reset(self, tracker: ExecutionQualityTracker) -> None:
        for i in range(5):
            tracker.record_execution(make_record(strategy=ExecutionStrategy.MARKET))
            tracker.record_execution(make_record(strategy=ExecutionStrategy.LIMIT_IOC))
        tracker.reset()
        assert not tracker.auto_selected
        assert tracker.get_market_stats().trade_count == 0
        assert tracker.get_limit_stats().trade_count == 0

    def test_max_records_bounded(self) -> None:
        tracker = ExecutionQualityTracker(min_trades_per_arm=5, max_records=10)
        for i in range(20):
            tracker.record_execution(make_record(strategy=ExecutionStrategy.MARKET))
        assert tracker.get_market_stats().trade_count == 10


class TestExecutionStrategy:
    def test_enum_values(self) -> None:
        assert ExecutionStrategy.MARKET.value == "market"
        assert ExecutionStrategy.LIMIT_IOC.value == "limit_ioc"
