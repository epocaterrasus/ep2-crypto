"""Tests for the event-driven macro module.

Tests cover:
- MacroEvent data model
- EWMACorrelation computation
- MacroEventMonitor signal generation
- VIX gating
- Correlation gating
- NQ minimum move threshold
- Signal expiry
- Calendar lookup
"""

from __future__ import annotations

import math

import pytest

from ep2_crypto.events.macro import (
    EWMACorrelation,
    MacroEvent,
    MacroEventMonitor,
    MacroEventType,
)


class TestMacroEvent:
    """Tests for MacroEvent data model."""

    def test_macro_event_creation(self) -> None:
        event = MacroEvent(
            event_type=MacroEventType.CPI,
            timestamp_ms=1736857800000,
            name="CPI Dec 2024",
            expected_impact=3,
        )
        assert event.event_type == MacroEventType.CPI
        assert event.timestamp_ms == 1736857800000
        assert event.name == "CPI Dec 2024"
        assert event.expected_impact == 3

    def test_macro_event_default_impact(self) -> None:
        event = MacroEvent(
            event_type=MacroEventType.NFP,
            timestamp_ms=1000000,
            name="NFP Test",
        )
        assert event.expected_impact == 2

    def test_macro_event_frozen(self) -> None:
        event = MacroEvent(
            event_type=MacroEventType.FOMC,
            timestamp_ms=1000000,
            name="FOMC Test",
        )
        with pytest.raises(AttributeError):
            event.name = "Changed"  # type: ignore[misc]

    def test_macro_event_types(self) -> None:
        assert len(MacroEventType) == 8
        assert MacroEventType.CPI.value == "cpi"
        assert MacroEventType.FOMC.value == "fomc"
        assert MacroEventType.NFP.value == "nfp"


class TestEWMACorrelation:
    """Tests for EWMA BTC-NQ correlation tracker."""

    def test_returns_nan_before_min_bars(self) -> None:
        ewma = EWMACorrelation(halflife_bars=10, min_bars=5)
        for i in range(4):
            result = ewma.update(0.01 * (i + 1), 0.01 * (i + 1))
        assert math.isnan(result)
        assert ewma.bars_seen == 4

    def test_returns_value_after_min_bars(self) -> None:
        ewma = EWMACorrelation(halflife_bars=10, min_bars=5)
        for i in range(6):
            result = ewma.update(0.01 * (i + 1), 0.01 * (i + 1))
        assert not math.isnan(result)
        assert ewma.bars_seen == 6

    def test_perfectly_correlated_returns(self) -> None:
        ewma = EWMACorrelation(halflife_bars=10, min_bars=5)
        for i in range(50):
            val = 0.01 * math.sin(i * 0.5)
            result = ewma.update(val, val)
        # Perfectly correlated → correlation near 1
        assert result > 0.9

    def test_anti_correlated_returns(self) -> None:
        ewma = EWMACorrelation(halflife_bars=10, min_bars=5)
        for i in range(50):
            val = 0.01 * math.sin(i * 0.5)
            result = ewma.update(val, -val)
        # Anti-correlated → correlation near -1
        assert result < -0.9

    def test_uncorrelated_returns(self) -> None:
        ewma = EWMACorrelation(halflife_bars=10, min_bars=5)
        rng = __import__("numpy").random.default_rng(42)
        for _ in range(200):
            result = ewma.update(
                float(rng.normal(0, 0.01)),
                float(rng.normal(0, 0.01)),
            )
        # Uncorrelated → correlation near 0
        assert abs(result) < 0.3

    def test_correlation_property_matches_update(self) -> None:
        ewma = EWMACorrelation(halflife_bars=10, min_bars=5)
        for i in range(20):
            last = ewma.update(0.01 * math.sin(i), 0.01 * math.cos(i))
        assert ewma.correlation == pytest.approx(last, abs=1e-10)

    def test_zero_variance_returns_zero(self) -> None:
        ewma = EWMACorrelation(halflife_bars=10, min_bars=3)
        for _ in range(10):
            result = ewma.update(0.0, 0.0)
        assert result == 0.0


class TestMacroEventMonitor:
    """Tests for macro event monitor signal generation."""

    @staticmethod
    def _make_calendar() -> list[MacroEvent]:
        """Create a test calendar with a single event at T=1,000,000,000 ms."""
        return [
            MacroEvent(
                event_type=MacroEventType.CPI,
                timestamp_ms=1_000_000_000,
                name="Test CPI",
                expected_impact=3,
            ),
        ]

    @staticmethod
    def _warmup_monitor(
        monitor: MacroEventMonitor,
        n_bars: int = 20,
        start_ms: int = 993_000_000,
    ) -> None:
        """Warm up the EWMA correlation with correlated data.

        Default start_ms is chosen so warmup ends before the test event
        at T=1,000,000,000ms (20 bars * 300,000ms = 6,000,000ms → ends at 998,700,000ms).
        """
        for i in range(n_bars):
            ts = start_ms + i * 300_000
            btc = 50000.0 + 100.0 * math.sin(i * 0.3)
            btc_prev = 50000.0 + 100.0 * math.sin((i - 1) * 0.3)
            nq = 18000.0 + 50.0 * math.sin(i * 0.3)
            nq_prev = 18000.0 + 50.0 * math.sin((i - 1) * 0.3)
            monitor.update(ts, btc, btc_prev, nq, nq_prev, vix_level=20.0)

    def test_no_signal_outside_event_window(self) -> None:
        monitor = MacroEventMonitor(calendar=self._make_calendar())
        # Way before the event
        signal = monitor.update(
            900_000_000, 50000.0, 49900.0, 18000.0, 17950.0, vix_level=20.0,
        )
        assert signal is None

    def test_signal_generated_at_delay(self) -> None:
        calendar = self._make_calendar()
        monitor = MacroEventMonitor(
            calendar=calendar,
            nq_min_move_pct=0.01,
            min_correlation=0.1,
        )
        self._warmup_monitor(monitor)

        # At T+2min (signal delay = 120,000 ms)
        event_time = 1_000_000_000
        signal_time = event_time + 120_000

        signal = monitor.update(
            signal_time,
            50100.0,  # btc up
            50000.0,
            18050.0,  # nq up
            18000.0,
            vix_level=20.0,
        )

        assert signal is not None
        assert signal.direction == 1  # NQ up → BTC long
        assert signal.event.name == "Test CPI"
        assert signal.confidence > 0.0

    def test_signal_direction_short(self) -> None:
        calendar = self._make_calendar()
        monitor = MacroEventMonitor(
            calendar=calendar,
            nq_min_move_pct=0.01,
            min_correlation=0.1,
        )
        self._warmup_monitor(monitor)

        event_time = 1_000_000_000
        signal_time = event_time + 120_000

        signal = monitor.update(
            signal_time,
            49900.0,
            50000.0,
            17950.0,  # nq down
            18000.0,
            vix_level=20.0,
        )

        assert signal is not None
        assert signal.direction == -1  # NQ down → BTC short

    def test_vix_gate_blocks_signal(self) -> None:
        calendar = self._make_calendar()
        monitor = MacroEventMonitor(
            calendar=calendar,
            vix_threshold=35.0,
            nq_min_move_pct=0.01,
            min_correlation=0.1,
        )
        self._warmup_monitor(monitor)

        event_time = 1_000_000_000
        signal_time = event_time + 120_000

        signal = monitor.update(
            signal_time,
            50100.0, 50000.0, 18050.0, 18000.0,
            vix_level=40.0,  # VIX too high
        )

        assert signal is None

    def test_min_move_gate(self) -> None:
        calendar = self._make_calendar()
        monitor = MacroEventMonitor(
            calendar=calendar,
            nq_min_move_pct=0.1,
            min_correlation=0.1,
        )
        self._warmup_monitor(monitor)

        event_time = 1_000_000_000
        signal_time = event_time + 120_000

        # Tiny NQ move (< 0.1%)
        signal = monitor.update(
            signal_time,
            50001.0, 50000.0, 18001.0, 18000.0,
            vix_level=20.0,
        )

        assert signal is None

    def test_signal_expiry(self) -> None:
        calendar = self._make_calendar()
        monitor = MacroEventMonitor(
            calendar=calendar,
            nq_min_move_pct=0.01,
            min_correlation=0.1,
            signal_expiry_ms=600_000,
        )
        self._warmup_monitor(monitor)

        # Generate signal at T+2min
        event_time = 1_000_000_000
        signal_time = event_time + 120_000
        signal = monitor.update(
            signal_time,
            50100.0, 50000.0, 18050.0, 18000.0,
            vix_level=20.0,
        )
        assert signal is not None

        # After expiry (T+12min, > 10min signal expiry)
        expired_time = signal_time + 700_000
        signal = monitor.update(
            expired_time,
            50100.0, 50000.0, 18050.0, 18000.0,
            vix_level=20.0,
        )
        assert signal is None

    def test_no_duplicate_signal_same_event(self) -> None:
        calendar = self._make_calendar()
        monitor = MacroEventMonitor(
            calendar=calendar,
            nq_min_move_pct=0.01,
            min_correlation=0.1,
        )
        self._warmup_monitor(monitor)

        event_time = 1_000_000_000
        signal_time = event_time + 120_000

        # First signal
        sig1 = monitor.update(
            signal_time,
            50100.0, 50000.0, 18050.0, 18000.0,
            vix_level=20.0,
        )
        assert sig1 is not None

        # Same window, should return same signal (not create new)
        sig2 = monitor.update(
            signal_time + 60_000,
            50150.0, 50100.0, 18060.0, 18050.0,
            vix_level=20.0,
        )
        assert sig2 is not None
        assert sig2.signal_timestamp_ms == sig1.signal_timestamp_ms

    def test_get_next_event(self) -> None:
        calendar = [
            MacroEvent(MacroEventType.CPI, 1000, "CPI 1"),
            MacroEvent(MacroEventType.FOMC, 2000, "FOMC 1"),
            MacroEvent(MacroEventType.NFP, 3000, "NFP 1"),
        ]
        monitor = MacroEventMonitor(calendar=calendar)

        assert monitor.get_next_event(500) is not None
        assert monitor.get_next_event(500).name == "CPI 1"  # type: ignore[union-attr]
        assert monitor.get_next_event(1500).name == "FOMC 1"  # type: ignore[union-attr]
        assert monitor.get_next_event(3500) is None

    def test_time_to_next_event(self) -> None:
        calendar = [
            MacroEvent(MacroEventType.CPI, 1_000_000, "CPI"),
        ]
        monitor = MacroEventMonitor(calendar=calendar)
        assert monitor.time_to_next_event_ms(800_000) == 200_000
        assert monitor.time_to_next_event_ms(1_100_000) is None

    def test_confidence_components(self) -> None:
        calendar = self._make_calendar()
        monitor = MacroEventMonitor(
            calendar=calendar,
            nq_min_move_pct=0.01,
            min_correlation=0.1,
        )
        self._warmup_monitor(monitor)

        event_time = 1_000_000_000
        signal_time = event_time + 120_000

        signal = monitor.update(
            signal_time,
            50100.0, 50000.0, 18050.0, 18000.0,
            vix_level=15.0,  # Low VIX = high confidence
        )

        assert signal is not None
        assert 0.0 < signal.confidence <= 1.0
        assert signal.btc_nq_correlation > 0
        assert signal.vix_level == 15.0

    def test_no_vix_available(self) -> None:
        calendar = self._make_calendar()
        monitor = MacroEventMonitor(
            calendar=calendar,
            nq_min_move_pct=0.01,
            min_correlation=0.1,
        )
        self._warmup_monitor(monitor)

        event_time = 1_000_000_000
        signal_time = event_time + 120_000

        signal = monitor.update(
            signal_time,
            50100.0, 50000.0, 18050.0, 18000.0,
            vix_level=None,  # No VIX data
        )

        assert signal is not None
        assert signal.vix_level is None
        # Should still generate signal with moderate confidence

    def test_empty_calendar(self) -> None:
        monitor = MacroEventMonitor(calendar=[])
        signal = monitor.update(
            1_000_000_000, 50000.0, 49900.0, 18000.0, 17950.0,
        )
        assert signal is None

    def test_calendar_property(self) -> None:
        calendar = self._make_calendar()
        monitor = MacroEventMonitor(calendar=calendar)
        assert len(monitor.calendar) == 1
        assert monitor.calendar[0].name == "Test CPI"

    def test_default_calendar_loaded(self) -> None:
        monitor = MacroEventMonitor()
        assert len(monitor.calendar) > 0
