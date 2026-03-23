"""Tests for slippage tracker."""

from __future__ import annotations

import pytest

from ep2_crypto.monitoring.slippage_tracker import (
    SlippageStats,
    SlippageTracker,
    _compute_stats,
    _percentile,
    SlippageRecord,
)


class TestPercentile:
    def test_empty_returns_zero(self) -> None:
        assert _percentile([], 0.5) == 0.0

    def test_single_value(self) -> None:
        assert _percentile([5.0], 0.5) == 5.0

    def test_median_odd(self) -> None:
        assert _percentile([1.0, 2.0, 3.0], 0.5) == 2.0

    def test_p95(self) -> None:
        values = sorted(list(range(1, 101)))
        result = _percentile([float(v) for v in values], 0.95)
        assert 95 <= result <= 96


class TestSlippageTracker:
    @pytest.fixture()
    def tracker(self) -> SlippageTracker:
        return SlippageTracker(
            max_records=1000,
            alert_ratio=2.0,
            consecutive_alert_count=5,
        )

    def test_record_basic(self, tracker: SlippageTracker) -> None:
        alert = tracker.record(
            timestamp_ms=1000,
            expected_bps=3.0,
            actual_bps=3.5,
            order_size_usd=5000,
            hour_utc=14,
            regime="trending",
        )
        assert not alert
        assert tracker.record_count == 1

    def test_no_alert_normal_slippage(self, tracker: SlippageTracker) -> None:
        for i in range(20):
            alert = tracker.record(
                timestamp_ms=i * 1000,
                expected_bps=3.0,
                actual_bps=3.5,
                order_size_usd=5000,
                hour_utc=14,
            )
            assert not alert

    def test_alert_on_consecutive_excess(self, tracker: SlippageTracker) -> None:
        fired = False
        for i in range(10):
            alert = tracker.record(
                timestamp_ms=i * 1000,
                expected_bps=3.0,
                actual_bps=7.0,  # > 2x expected
                order_size_usd=5000,
                hour_utc=14,
            )
            if alert:
                fired = True
                break
        assert fired
        assert tracker.alert_active
        assert tracker.consecutive_excess >= 5

    def test_consecutive_resets_on_normal(self, tracker: SlippageTracker) -> None:
        # 4 excessive trades
        for i in range(4):
            tracker.record(
                timestamp_ms=i * 1000,
                expected_bps=3.0,
                actual_bps=7.0,
                order_size_usd=5000,
                hour_utc=14,
            )
        assert tracker.consecutive_excess == 4
        # 1 normal trade resets
        tracker.record(
            timestamp_ms=5000,
            expected_bps=3.0,
            actual_bps=3.5,
            order_size_usd=5000,
            hour_utc=14,
        )
        assert tracker.consecutive_excess == 0

    def test_overall_stats(self, tracker: SlippageTracker) -> None:
        for i in range(50):
            tracker.record(
                timestamp_ms=i * 1000,
                expected_bps=3.0,
                actual_bps=3.0 + i * 0.1,
                order_size_usd=5000,
                hour_utc=14,
            )
        stats = tracker.get_overall_stats()
        assert stats.count == 50
        assert stats.mean_expected_bps == pytest.approx(3.0)
        assert stats.mean_actual_bps > 3.0
        assert stats.p50_actual_bps > 0
        assert stats.p95_actual_bps > stats.p50_actual_bps

    def test_stats_by_regime(self, tracker: SlippageTracker) -> None:
        for i in range(20):
            tracker.record(
                timestamp_ms=i * 1000,
                expected_bps=3.0,
                actual_bps=3.5,
                order_size_usd=5000,
                hour_utc=14,
                regime="trending",
            )
        for i in range(10):
            tracker.record(
                timestamp_ms=(20 + i) * 1000,
                expected_bps=3.0,
                actual_bps=5.0,
                order_size_usd=5000,
                hour_utc=14,
                regime="choppy",
            )
        by_regime = tracker.get_stats_by_regime()
        assert "trending" in by_regime
        assert "choppy" in by_regime
        assert by_regime["trending"].count == 20
        assert by_regime["choppy"].count == 10
        assert by_regime["choppy"].mean_actual_bps > by_regime["trending"].mean_actual_bps

    def test_stats_by_hour(self, tracker: SlippageTracker) -> None:
        for i in range(10):
            tracker.record(
                timestamp_ms=i * 1000, expected_bps=3.0, actual_bps=3.5,
                order_size_usd=5000, hour_utc=9,
            )
        for i in range(10):
            tracker.record(
                timestamp_ms=(10 + i) * 1000, expected_bps=3.0, actual_bps=4.0,
                order_size_usd=5000, hour_utc=15,
            )
        by_hour = tracker.get_stats_by_hour()
        assert 9 in by_hour
        assert 15 in by_hour

    def test_stats_by_size_bucket(self, tracker: SlippageTracker) -> None:
        sizes = [500, 3000, 8000, 25000, 100000]
        for i, size in enumerate(sizes):
            tracker.record(
                timestamp_ms=i * 1000, expected_bps=3.0, actual_bps=3.5,
                order_size_usd=size, hour_utc=14,
            )
        by_size = tracker.get_stats_by_size_bucket()
        assert len(by_size) >= 3

    def test_adaptive_slippage_estimate_overall(self, tracker: SlippageTracker) -> None:
        for i in range(50):
            tracker.record(
                timestamp_ms=i * 1000, expected_bps=3.0,
                actual_bps=3.0 + i * 0.1,
                order_size_usd=5000, hour_utc=14,
            )
        estimate = tracker.get_adaptive_slippage_estimate()
        assert estimate > 0

    def test_adaptive_slippage_estimate_by_regime(
        self, tracker: SlippageTracker
    ) -> None:
        for i in range(30):
            tracker.record(
                timestamp_ms=i * 1000, expected_bps=3.0, actual_bps=5.0,
                order_size_usd=5000, hour_utc=14, regime="volatile",
            )
        estimate = tracker.get_adaptive_slippage_estimate(regime="volatile")
        assert estimate > 0

    def test_adaptive_slippage_falls_back_to_overall(
        self, tracker: SlippageTracker
    ) -> None:
        for i in range(30):
            tracker.record(
                timestamp_ms=i * 1000, expected_bps=3.0, actual_bps=4.0,
                order_size_usd=5000, hour_utc=14, regime="trending",
            )
        # Request unknown regime — falls back to overall
        estimate = tracker.get_adaptive_slippage_estimate(regime="unknown_regime")
        assert estimate > 0

    def test_max_records_bounded(self) -> None:
        tracker = SlippageTracker(max_records=10)
        for i in range(20):
            tracker.record(
                timestamp_ms=i * 1000, expected_bps=3.0, actual_bps=3.5,
                order_size_usd=5000, hour_utc=14,
            )
        assert tracker.record_count == 10

    def test_get_summary(self, tracker: SlippageTracker) -> None:
        for i in range(10):
            tracker.record(
                timestamp_ms=i * 1000, expected_bps=3.0, actual_bps=3.5,
                order_size_usd=5000, hour_utc=14,
            )
        summary = tracker.get_summary()
        assert summary["total_records"] == 10
        assert "mean_expected_bps" in summary
        assert "p95_actual_bps" in summary
        assert "alert_active" in summary

    def test_reset(self, tracker: SlippageTracker) -> None:
        for i in range(10):
            tracker.record(
                timestamp_ms=i * 1000, expected_bps=3.0, actual_bps=7.0,
                order_size_usd=5000, hour_utc=14,
            )
        tracker.reset()
        assert tracker.record_count == 0
        assert tracker.consecutive_excess == 0
        assert not tracker.alert_active

    def test_empty_stats(self, tracker: SlippageTracker) -> None:
        stats = tracker.get_overall_stats()
        assert stats.count == 0
        assert stats.mean_actual_bps == 0.0

    def test_empty_adaptive_estimate(self, tracker: SlippageTracker) -> None:
        assert tracker.get_adaptive_slippage_estimate() == 0.0
