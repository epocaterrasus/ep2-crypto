"""A/B test framework for execution strategies.

Compares two execution strategies (market orders vs limit IOC) and
auto-switches to the best after sufficient samples.

Strategy A: Market orders (baseline)
Strategy B: Limit IOC 1-2 ticks from mid

Metrics: fill rate, slippage, total cost per strategy.
Auto-switch after 100+ trades per arm.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

MIN_TRADES_PER_ARM = 100


class ExecutionStrategy(StrEnum):
    MARKET = "market"
    LIMIT_IOC = "limit_ioc"


@dataclass
class ExecutionRecord:
    """Record of a single execution attempt."""

    strategy: ExecutionStrategy
    timestamp_ms: int
    filled: bool
    slippage_bps: float
    cost_bps: float  # Total cost including fees
    size_usd: float
    latency_ms: float


@dataclass
class StrategyStats:
    """Aggregate stats for one execution strategy."""

    strategy: ExecutionStrategy
    trade_count: int = 0
    fill_rate: float = 0.0
    mean_slippage_bps: float = 0.0
    mean_cost_bps: float = 0.0
    total_cost_bps: float = 0.0
    mean_latency_ms: float = 0.0


def _compute_strategy_stats(
    strategy: ExecutionStrategy,
    records: list[ExecutionRecord],
) -> StrategyStats:
    """Compute aggregate stats from execution records."""
    if not records:
        return StrategyStats(strategy=strategy)

    filled = [r for r in records if r.filled]
    fill_rate = len(filled) / len(records) if records else 0.0

    slippages = [r.slippage_bps for r in filled] if filled else [0.0]
    costs = [r.cost_bps for r in records]
    latencies = [r.latency_ms for r in records]

    return StrategyStats(
        strategy=strategy,
        trade_count=len(records),
        fill_rate=fill_rate,
        mean_slippage_bps=sum(slippages) / len(slippages),
        mean_cost_bps=sum(costs) / len(costs),
        total_cost_bps=sum(costs),
        mean_latency_ms=sum(latencies) / len(latencies),
    )


class ExecutionQualityTracker:
    """A/B test framework for execution strategies.

    Alternates between strategies until both have min_trades,
    then auto-switches to the winner.
    """

    def __init__(
        self,
        min_trades_per_arm: int = MIN_TRADES_PER_ARM,
        max_records: int = 5000,
    ) -> None:
        self._min_trades = min_trades_per_arm
        self._market_records: deque[ExecutionRecord] = deque(maxlen=max_records)
        self._limit_records: deque[ExecutionRecord] = deque(maxlen=max_records)
        self._active_strategy = ExecutionStrategy.MARKET
        self._auto_selected = False
        self._selection_reason: str = ""

    def get_strategy(self) -> ExecutionStrategy:
        """Get the strategy to use for the next trade.

        During A/B testing, alternates between strategies.
        After auto-selection, always returns the winner.
        """
        if self._auto_selected:
            return self._active_strategy

        # Alternate during testing phase
        market_count = len(self._market_records)
        limit_count = len(self._limit_records)

        if market_count <= limit_count:
            return ExecutionStrategy.MARKET
        return ExecutionStrategy.LIMIT_IOC

    def record_execution(self, record: ExecutionRecord) -> None:
        """Record an execution result and check if auto-selection should trigger."""
        if record.strategy == ExecutionStrategy.MARKET:
            self._market_records.append(record)
        else:
            self._limit_records.append(record)

        logger.debug(
            "execution_recorded",
            strategy=record.strategy.value,
            filled=record.filled,
            slippage_bps=record.slippage_bps,
            cost_bps=record.cost_bps,
        )

        # Check if we should auto-select
        if not self._auto_selected:
            self._check_auto_select()

    def _check_auto_select(self) -> None:
        """Auto-select the best strategy once both arms have enough data."""
        if (
            len(self._market_records) < self._min_trades
            or len(self._limit_records) < self._min_trades
        ):
            return

        market_stats = self.get_market_stats()
        limit_stats = self.get_limit_stats()

        # Compare on total effective cost = slippage + fees - fill_rate_penalty
        # Unfilled limit orders have opportunity cost
        market_effective = market_stats.mean_cost_bps
        # Penalize limit IOC for unfilled orders (missed trades)
        fill_penalty = (1.0 - limit_stats.fill_rate) * market_stats.mean_cost_bps * 2
        limit_effective = limit_stats.mean_cost_bps + fill_penalty

        if limit_effective < market_effective:
            self._active_strategy = ExecutionStrategy.LIMIT_IOC
            self._selection_reason = (
                f"limit_ioc effective cost {limit_effective:.2f} bps "
                f"< market {market_effective:.2f} bps"
            )
        else:
            self._active_strategy = ExecutionStrategy.MARKET
            self._selection_reason = (
                f"market effective cost {market_effective:.2f} bps "
                f"<= limit_ioc {limit_effective:.2f} bps"
            )

        self._auto_selected = True
        logger.info(
            "execution_strategy_auto_selected",
            strategy=self._active_strategy.value,
            reason=self._selection_reason,
            market_trades=market_stats.trade_count,
            limit_trades=limit_stats.trade_count,
        )

    def get_market_stats(self) -> StrategyStats:
        return _compute_strategy_stats(
            ExecutionStrategy.MARKET, list(self._market_records)
        )

    def get_limit_stats(self) -> StrategyStats:
        return _compute_strategy_stats(
            ExecutionStrategy.LIMIT_IOC, list(self._limit_records)
        )

    def get_summary(self) -> dict[str, Any]:
        """Summary dict for API/monitoring."""
        market = self.get_market_stats()
        limit = self.get_limit_stats()
        return {
            "active_strategy": self._active_strategy.value,
            "auto_selected": self._auto_selected,
            "selection_reason": self._selection_reason,
            "market": {
                "trades": market.trade_count,
                "fill_rate": round(market.fill_rate, 4),
                "mean_slippage_bps": round(market.mean_slippage_bps, 2),
                "mean_cost_bps": round(market.mean_cost_bps, 2),
            },
            "limit_ioc": {
                "trades": limit.trade_count,
                "fill_rate": round(limit.fill_rate, 4),
                "mean_slippage_bps": round(limit.mean_slippage_bps, 2),
                "mean_cost_bps": round(limit.mean_cost_bps, 2),
            },
        }

    @property
    def auto_selected(self) -> bool:
        return self._auto_selected

    @property
    def active_strategy(self) -> ExecutionStrategy:
        return self._active_strategy

    @property
    def selection_reason(self) -> str:
        return self._selection_reason

    def reset(self) -> None:
        """Reset all state for a new A/B test."""
        self._market_records.clear()
        self._limit_records.clear()
        self._active_strategy = ExecutionStrategy.MARKET
        self._auto_selected = False
        self._selection_reason = ""
