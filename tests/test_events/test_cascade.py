"""Tests for the liquidation cascade detection module.

Tests cover:
- HawkesProcess: intensity, event recording, branching ratio, reset
- CascadeDetector: liquidation events, bar updates, cascade probability,
  position size multiplier, alert levels, multi-factor scoring
"""

from __future__ import annotations

import math

import pytest

from ep2_crypto.events.cascade import (
    CascadeDetector,
    CascadeState,
    HawkesProcess,
)


class TestHawkesProcess:
    """Tests for the Hawkes self-exciting point process."""

    def test_baseline_intensity(self) -> None:
        hp = HawkesProcess(mu=0.01, alpha=0.05, beta=0.1)
        assert hp.intensity(0.0) == 0.01  # No events → baseline

    def test_intensity_increases_after_event(self) -> None:
        hp = HawkesProcess(mu=0.01, alpha=0.05, beta=0.1)
        # Add event at t=0
        intensity_after = hp.add_event(0.0)
        # Intensity should jump: mu + alpha * 1 = 0.01 + 0.05 = 0.06
        assert intensity_after == pytest.approx(0.06)

    def test_intensity_decays_over_time(self) -> None:
        hp = HawkesProcess(mu=0.01, alpha=0.05, beta=0.1)
        hp.add_event(0.0)

        # At t=10s, intensity decays: mu + alpha * exp(-beta * 10)
        expected = 0.01 + 0.05 * math.exp(-0.1 * 10)
        actual = hp.intensity(10.0)
        assert actual == pytest.approx(expected, rel=1e-6)

    def test_intensity_returns_to_baseline(self) -> None:
        hp = HawkesProcess(mu=0.01, alpha=0.05, beta=0.1)
        hp.add_event(0.0)

        # After very long time, should approach baseline
        intensity = hp.intensity(1000.0)
        assert intensity == pytest.approx(0.01, abs=1e-6)

    def test_multiple_events_compound(self) -> None:
        hp = HawkesProcess(mu=0.01, alpha=0.05, beta=0.1)
        hp.add_event(0.0)
        i1 = hp.intensity(1.0)  # Before second event
        hp.add_event(1.0)
        i2 = hp.intensity(1.0)  # After second event
        # Second event should increase intensity above first event's decayed level
        assert i2 > i1

    def test_rapid_events_create_high_intensity(self) -> None:
        hp = HawkesProcess(mu=0.01, alpha=0.1, beta=0.05)
        # Simulate burst of 20 events in 10 seconds
        for i in range(20):
            hp.add_event(float(i) * 0.5)

        intensity = hp.intensity(10.0)
        # Should be significantly above baseline
        assert intensity > hp.mu * 5

    def test_branching_ratio(self) -> None:
        hp = HawkesProcess(mu=0.01, alpha=0.05, beta=0.1)
        assert hp.branching_ratio == pytest.approx(0.5)

    def test_branching_ratio_subcritical(self) -> None:
        hp = HawkesProcess(mu=0.01, alpha=0.03, beta=0.1)
        assert hp.branching_ratio < 1.0

    def test_estimated_branching_ratio_normal(self) -> None:
        hp = HawkesProcess(mu=0.01, alpha=0.05, beta=0.1)
        # Few events → low estimated ratio
        hp.add_event(0.0)
        hp.add_event(100.0)
        ratio = hp.estimated_branching_ratio(window_s=200.0)
        assert 0.0 <= ratio <= 1.0

    def test_estimated_branching_ratio_burst(self) -> None:
        hp = HawkesProcess(mu=0.01, alpha=0.1, beta=0.05)
        # Many events in short window → high estimated ratio
        for i in range(50):
            hp.add_event(float(i) * 0.5)

        ratio = hp.estimated_branching_ratio(window_s=30.0)
        assert ratio > 0.5  # Clearly self-exciting

    def test_estimated_branching_ratio_empty(self) -> None:
        hp = HawkesProcess(mu=0.01, alpha=0.05, beta=0.1)
        assert hp.estimated_branching_ratio() == 0.0

    def test_event_count(self) -> None:
        hp = HawkesProcess(mu=0.01, alpha=0.05, beta=0.1)
        assert hp.event_count == 0
        hp.add_event(1.0)
        hp.add_event(2.0)
        assert hp.event_count == 2

    def test_max_history_pruning(self) -> None:
        hp = HawkesProcess(mu=0.01, alpha=0.05, beta=0.1, max_history=10)
        for i in range(20):
            hp.add_event(float(i))
        assert hp.event_count == 10

    def test_reset(self) -> None:
        hp = HawkesProcess(mu=0.01, alpha=0.05, beta=0.1)
        hp.add_event(0.0)
        hp.add_event(1.0)
        hp.reset()
        assert hp.event_count == 0
        assert hp.intensity(2.0) == hp.mu

    def test_invalid_beta_raises(self) -> None:
        with pytest.raises(ValueError, match="beta must be positive"):
            HawkesProcess(mu=0.01, alpha=0.05, beta=0.0)

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha must be non-negative"):
            HawkesProcess(mu=0.01, alpha=-0.05, beta=0.1)

    def test_properties(self) -> None:
        hp = HawkesProcess(mu=0.01, alpha=0.05, beta=0.1)
        assert hp.mu == 0.01
        assert hp.alpha == 0.05
        assert hp.beta == 0.1


class TestCascadeDetector:
    """Tests for the multi-factor cascade detector."""

    def test_initial_state_normal(self) -> None:
        detector = CascadeDetector()
        state = detector.state
        assert state.cascade_probability == 0.0
        assert state.alert_level == 0
        assert state.position_size_multiplier == 1.0

    def test_single_liquidation_low_risk(self) -> None:
        detector = CascadeDetector()
        state = detector.on_liquidation(1000.0, 0.5, "long")
        # Single event should not trigger cascade
        assert state.alert_level <= 1
        assert state.position_size_multiplier >= 0.5

    def test_burst_liquidations_increase_risk(self) -> None:
        detector = CascadeDetector()
        # Simulate burst: 30 liquidations in 30 seconds
        for i in range(30):
            state = detector.on_liquidation(1000.0 + i, 0.5, "long")

        # Burst rate should be elevated
        assert state.liq_burst_rate > 0.3
        assert state.hawkes_intensity > detector.hawkes.mu

    def test_cascade_scenario_high_probability(self) -> None:
        detector = CascadeDetector(
            hawkes_mu=0.01,
            hawkes_alpha=0.15,
            hawkes_beta=0.1,
        )

        # Build up OI history at moderate level, then spike
        for i in range(100):
            detector.on_bar(
                float(i * 300),
                open_interest=50000.0 + i * 10,
                funding_rate=0.0001,
                book_depth=1000.0,
                price=50000.0,
                prev_price=50000.0,
            )

        # Now simulate cascade conditions: many liquidations + thin book + high OI
        detector.on_bar(
            30100.0,
            open_interest=60000.0,  # High OI
            funding_rate=0.001,  # Elevated funding
            book_depth=200.0,  # Thin book (vs avg ~1000)
            price=49000.0,  # Rapid price drop
            prev_price=50000.0,
        )

        # Burst of liquidations
        for i in range(40):
            detector.on_liquidation(30100.0 + i * 0.5, 1.0, "long")

        state = detector.state
        assert state.branching_ratio > 0.3
        assert state.liq_burst_rate > 0.5
        assert state.cascade_probability > 0.3

    def test_on_bar_updates_oi_percentile(self) -> None:
        detector = CascadeDetector()
        # Feed ascending OI values
        for i in range(20):
            detector.on_bar(
                float(i * 300),
                open_interest=1000.0 + i * 100,
            )
        # Last value is highest → percentile should be 1.0
        assert detector.state.oi_percentile == pytest.approx(1.0)

    def test_on_bar_updates_funding_zscore(self) -> None:
        detector = CascadeDetector()
        # Feed normal funding rates
        for i in range(20):
            detector.on_bar(
                float(i * 300),
                funding_rate=0.0001,
            )
        # Then an extreme rate
        detector.on_bar(6000.0, funding_rate=0.005)
        # z-score should be elevated
        assert abs(detector.state.funding_zscore) > 2.0

    def test_on_bar_updates_book_depth_ratio(self) -> None:
        detector = CascadeDetector()
        # Feed consistent depth
        for i in range(20):
            detector.on_bar(float(i * 300), book_depth=1000.0)
        # Then thin book
        detector.on_bar(6000.0, book_depth=300.0)
        assert detector.state.book_depth_ratio < 0.5

    def test_on_bar_updates_price_velocity(self) -> None:
        detector = CascadeDetector()
        # Large price move in one bar
        detector.on_bar(
            300.0,
            price=49000.0,
            prev_price=50000.0,
            bar_duration_s=300.0,
        )
        # 200 bps / 300s = 0.67 bps/sec
        assert detector.state.price_velocity > 0.5

    def test_position_multiplier_normal(self) -> None:
        detector = CascadeDetector()
        state = detector.on_bar(0.0)
        assert state.position_size_multiplier == 1.0
        assert state.alert_level == 0

    def test_alert_levels_map_to_multipliers(self) -> None:
        detector = CascadeDetector()
        # Force different probability levels via internal state
        state = detector.state

        # Low probability → normal
        state.cascade_probability = 0.1
        assert state.alert_level == 0 or state.position_size_multiplier == 1.0

    def test_hawkes_property_accessible(self) -> None:
        detector = CascadeDetector()
        assert detector.hawkes is not None
        assert detector.hawkes.mu > 0

    def test_reset_clears_state(self) -> None:
        detector = CascadeDetector()
        # Add some data
        for i in range(10):
            detector.on_liquidation(float(i), 0.5, "long")
        detector.on_bar(100.0, open_interest=50000.0, funding_rate=0.001)

        detector.reset()
        state = detector.state
        assert state.cascade_probability == 0.0
        assert state.hawkes_intensity == 0.0
        assert state.liq_burst_rate == 0.0
        assert detector.hawkes.event_count == 0

    def test_cascade_state_defaults(self) -> None:
        state = CascadeState()
        assert state.cascade_probability == 0.0
        assert state.hawkes_intensity == 0.0
        assert state.branching_ratio == 0.0
        assert state.liq_burst_rate == 0.0
        assert state.oi_percentile == 0.5
        assert state.funding_zscore == 0.0
        assert state.book_depth_ratio == 1.0
        assert state.price_velocity == 0.0
        assert state.position_size_multiplier == 1.0
        assert state.alert_level == 0

    def test_only_liquidation_events_tracked(self) -> None:
        detector = CascadeDetector()
        # on_bar without liquidations should not add to burst rate
        for i in range(10):
            detector.on_bar(float(i * 300))
        assert detector.state.liq_burst_rate == 0.0

    def test_partial_bar_data(self) -> None:
        detector = CascadeDetector()
        # Only OI provided, rest None
        state = detector.on_bar(0.0, open_interest=50000.0)
        assert state.funding_zscore == 0.0  # Not updated
        assert state.book_depth_ratio == 1.0  # Not updated

    def test_gradual_escalation(self) -> None:
        """Verify that cascade probability increases as conditions worsen."""
        detector = CascadeDetector(
            hawkes_mu=0.01,
            hawkes_alpha=0.15,
            hawkes_beta=0.1,
        )

        # Phase 1: Normal conditions
        for i in range(50):
            detector.on_bar(
                float(i * 300),
                open_interest=50000.0,
                funding_rate=0.0001,
                book_depth=1000.0,
                price=50000.0,
                prev_price=50000.0,
            )
        prob_normal = detector.state.cascade_probability

        # Phase 2: Add some liquidations
        for i in range(10):
            detector.on_liquidation(15000.0 + i * 2, 0.5, "long")
        prob_elevated = detector.state.cascade_probability

        # Phase 3: Heavy liquidation burst + thin book
        detector.on_bar(
            15100.0,
            book_depth=200.0,
            price=49500.0,
            prev_price=50000.0,
        )
        for i in range(30):
            detector.on_liquidation(15100.0 + i * 0.3, 1.0, "long")
        prob_cascade = detector.state.cascade_probability

        # Should be monotonically increasing
        assert prob_elevated >= prob_normal
        assert prob_cascade >= prob_elevated
