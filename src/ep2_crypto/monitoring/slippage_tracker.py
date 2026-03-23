"""Slippage tracking: expected vs actual, stats by size/time/regime.

Feeds stats back to position sizer for adaptive cost modeling.
Alerts when actual slippage consistently exceeds expected.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

CONSECUTIVE_ALERT_THRESHOLD = 10


@dataclass(frozen=True)
class SlippageRecord:
    """Single slippage observation."""

    timestamp_ms: int
    expected_bps: float
    actual_bps: float
    order_size_usd: float
    hour_utc: int
    regime: str


@dataclass
class SlippageStats:
    """Aggregate slippage statistics."""

    count: int = 0
    mean_expected_bps: float = 0.0
    mean_actual_bps: float = 0.0
    p50_actual_bps: float = 0.0
    p95_actual_bps: float = 0.0
    p99_actual_bps: float = 0.0
    mean_ratio: float = 0.0  # actual / expected


def _percentile(sorted_values: list[float], p: float) -> float:
    """Compute percentile from sorted values."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def _compute_stats(records: list[SlippageRecord]) -> SlippageStats:
    """Compute aggregate stats from a list of records."""
    if not records:
        return SlippageStats()

    expected = [r.expected_bps for r in records]
    actual = [r.actual_bps for r in records]
    sorted_actual = sorted(actual)
    ratios = [
        a / e if e > 0 else 0.0
        for a, e in zip(actual, expected, strict=True)
    ]

    return SlippageStats(
        count=len(records),
        mean_expected_bps=sum(expected) / len(expected),
        mean_actual_bps=sum(actual) / len(actual),
        p50_actual_bps=_percentile(sorted_actual, 0.50),
        p95_actual_bps=_percentile(sorted_actual, 0.95),
        p99_actual_bps=_percentile(sorted_actual, 0.99),
        mean_ratio=sum(ratios) / len(ratios) if ratios else 0.0,
    )


class SlippageTracker:
    """Tracks expected vs actual slippage and generates alerts.

    Stats are segmented by order size bucket, hour of day, and regime.
    Alerts when actual > 2x expected for N consecutive trades.
    """

    def __init__(
        self,
        max_records: int = 5000,
        alert_ratio: float = 2.0,
        consecutive_alert_count: int = CONSECUTIVE_ALERT_THRESHOLD,
        size_buckets: list[float] | None = None,
    ) -> None:
        self._max_records = max_records
        self._alert_ratio = alert_ratio
        self._consecutive_alert_count = consecutive_alert_count
        self._size_buckets = size_buckets or [1000, 5000, 10000, 50000]

        self._records: deque[SlippageRecord] = deque(maxlen=max_records)
        self._consecutive_excess = 0
        self._alert_active = False

    def record(
        self,
        timestamp_ms: int,
        expected_bps: float,
        actual_bps: float,
        order_size_usd: float,
        hour_utc: int,
        regime: str = "unknown",
    ) -> bool:
        """Record a slippage observation. Returns True if alert fires."""
        rec = SlippageRecord(
            timestamp_ms=timestamp_ms,
            expected_bps=expected_bps,
            actual_bps=actual_bps,
            order_size_usd=order_size_usd,
            hour_utc=hour_utc,
            regime=regime,
        )
        self._records.append(rec)

        # Track consecutive excess
        if expected_bps > 0 and actual_bps > expected_bps * self._alert_ratio:
            self._consecutive_excess += 1
        else:
            self._consecutive_excess = 0

        if self._consecutive_excess >= self._consecutive_alert_count:
            self._alert_active = True
            logger.warning(
                "slippage_alert",
                consecutive_excess=self._consecutive_excess,
                latest_expected=expected_bps,
                latest_actual=actual_bps,
                ratio=round(actual_bps / expected_bps, 2) if expected_bps > 0 else 0,
            )
            return True

        self._alert_active = False
        return False

    def get_overall_stats(self) -> SlippageStats:
        """Aggregate stats across all records."""
        return _compute_stats(list(self._records))

    def get_stats_by_size_bucket(self) -> dict[str, SlippageStats]:
        """Stats segmented by order size bucket."""
        buckets: dict[str, list[SlippageRecord]] = {}
        for rec in self._records:
            bucket_name = self._classify_size(rec.order_size_usd)
            buckets.setdefault(bucket_name, []).append(rec)
        return {k: _compute_stats(v) for k, v in sorted(buckets.items())}

    def get_stats_by_hour(self) -> dict[int, SlippageStats]:
        """Stats segmented by hour of day (UTC)."""
        by_hour: dict[int, list[SlippageRecord]] = {}
        for rec in self._records:
            by_hour.setdefault(rec.hour_utc, []).append(rec)
        return {k: _compute_stats(v) for k, v in sorted(by_hour.items())}

    def get_stats_by_regime(self) -> dict[str, SlippageStats]:
        """Stats segmented by volatility regime."""
        by_regime: dict[str, list[SlippageRecord]] = {}
        for rec in self._records:
            by_regime.setdefault(rec.regime, []).append(rec)
        return {k: _compute_stats(v) for k, v in sorted(by_regime.items())}

    def get_adaptive_slippage_estimate(self, regime: str | None = None) -> float:
        """Get current best estimate of actual slippage for position sizing.

        Uses regime-specific stats if available, falls back to overall.
        Returns p95 actual slippage in bps.
        """
        if regime:
            regime_records = [r for r in self._records if r.regime == regime]
            if len(regime_records) >= 20:
                stats = _compute_stats(regime_records)
                return stats.p95_actual_bps

        overall = self.get_overall_stats()
        if overall.count >= 10:
            return overall.p95_actual_bps
        return 0.0

    def _classify_size(self, size_usd: float) -> str:
        """Classify order size into a bucket label."""
        for i, threshold in enumerate(self._size_buckets):
            if size_usd <= threshold:
                if i == 0:
                    return f"<={threshold}"
                return f"{self._size_buckets[i-1]}-{threshold}"
        return f">{self._size_buckets[-1]}"

    @property
    def alert_active(self) -> bool:
        return self._alert_active

    @property
    def consecutive_excess(self) -> int:
        return self._consecutive_excess

    @property
    def record_count(self) -> int:
        return len(self._records)

    def get_summary(self) -> dict[str, Any]:
        """Summary dict for API/monitoring endpoints."""
        stats = self.get_overall_stats()
        return {
            "total_records": stats.count,
            "mean_expected_bps": round(stats.mean_expected_bps, 2),
            "mean_actual_bps": round(stats.mean_actual_bps, 2),
            "p50_actual_bps": round(stats.p50_actual_bps, 2),
            "p95_actual_bps": round(stats.p95_actual_bps, 2),
            "p99_actual_bps": round(stats.p99_actual_bps, 2),
            "mean_ratio": round(stats.mean_ratio, 3),
            "alert_active": self._alert_active,
            "consecutive_excess": self._consecutive_excess,
        }

    def reset(self) -> None:
        """Clear all records and state."""
        self._records.clear()
        self._consecutive_excess = 0
        self._alert_active = False
