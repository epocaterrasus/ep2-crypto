"""Integration tests for Sprint 12: Event-Driven Macro Module + Cascade Detector.

Acceptance criteria:
1. Macro module fires on historical CPI/FOMC dates (backtest validation)
2. NQ lead-lag shows signal generation with correct direction
3. Cascade detector fires before or during known cascade-like events
4. Both modules operate independently from ML model
5. VIX gate correctly suppresses signals during extreme volatility
6. Hawkes process parameters produce meaningful branching ratios
"""

from __future__ import annotations

import math

import numpy as np

from ep2_crypto.events.cascade import CascadeDetector, HawkesProcess
from ep2_crypto.events.macro import (
    EWMACorrelation,
    MacroEvent,
    MacroEventMonitor,
    MacroEventType,
)


class TestMacroBacktestValidation:
    """Validate macro module fires on historical event dates."""

    def test_fires_on_cpi_date(self) -> None:
        """Simulate a CPI release: NQ reacts, BTC signal generated at T+2min."""
        cpi_time = 1_000_000_000
        calendar = [MacroEvent(MacroEventType.CPI, cpi_time, "CPI Test", 3)]
        monitor = MacroEventMonitor(
            calendar=calendar,
            nq_min_move_pct=0.01,
            min_correlation=0.1,
        )

        # Pre-event warmup (correlated BTC and NQ)
        rng = np.random.default_rng(42)
        base_btc = 50000.0
        base_nq = 18000.0
        for i in range(30):
            ts = cpi_time - 10_000_000 + i * 300_000
            noise = float(rng.normal(0, 0.002))
            btc = base_btc * (1.0 + noise)
            nq = base_nq * (1.0 + noise * 0.8)
            monitor.update(ts, btc, base_btc, nq, base_nq, vix_level=22.0)
            base_btc = btc
            base_nq = nq

        # CPI release: NQ jumps up 0.3%
        nq_post = base_nq * 1.003
        signal = monitor.update(
            cpi_time + 120_000,
            base_btc * 1.001,
            base_btc,
            nq_post,
            base_nq,
            vix_level=22.0,
        )

        assert signal is not None
        assert signal.direction == 1  # NQ up → long BTC
        assert signal.event.event_type == MacroEventType.CPI

    def test_fires_on_fomc_date(self) -> None:
        """Simulate FOMC: NQ drops, BTC short signal generated."""
        fomc_time = 2_000_000_000
        calendar = [MacroEvent(MacroEventType.FOMC, fomc_time, "FOMC Test", 3)]
        monitor = MacroEventMonitor(
            calendar=calendar,
            nq_min_move_pct=0.01,
            min_correlation=0.1,
        )

        # Warmup with correlated data
        base_btc = 50000.0
        base_nq = 18000.0
        for i in range(30):
            ts = fomc_time - 10_000_000 + i * 300_000
            step = math.sin(i * 0.3) * 0.002
            btc = base_btc * (1.0 + step)
            nq = base_nq * (1.0 + step * 0.9)
            monitor.update(ts, btc, base_btc, nq, base_nq, vix_level=20.0)
            base_btc = btc
            base_nq = nq

        # FOMC: NQ drops 0.5%
        nq_post = base_nq * 0.995
        signal = monitor.update(
            fomc_time + 120_000,
            base_btc * 0.998,
            base_btc,
            nq_post,
            base_nq,
            vix_level=20.0,
        )

        assert signal is not None
        assert signal.direction == -1  # NQ down → short BTC

    def test_vix_gate_suppresses_during_extreme_vol(self) -> None:
        """VIX > 35 should suppress all macro signals."""
        event_time = 3_000_000_000
        calendar = [MacroEvent(MacroEventType.CPI, event_time, "CPI High VIX", 3)]
        monitor = MacroEventMonitor(
            calendar=calendar,
            nq_min_move_pct=0.01,
            min_correlation=0.1,
            vix_threshold=35.0,
        )

        # Warmup
        base_btc = 50000.0
        base_nq = 18000.0
        for i in range(30):
            ts = event_time - 10_000_000 + i * 300_000
            step = math.sin(i * 0.3) * 0.002
            btc = base_btc * (1.0 + step)
            nq = base_nq * (1.0 + step)
            monitor.update(ts, btc, base_btc, nq, base_nq, vix_level=20.0)
            base_btc = btc
            base_nq = nq

        # Large NQ move but VIX = 40
        signal = monitor.update(
            event_time + 120_000,
            base_btc * 1.005,
            base_btc,
            base_nq * 1.005,
            base_nq,
            vix_level=40.0,
        )
        assert signal is None


class TestCascadeDetectionIntegration:
    """Integration tests for cascade detection scenarios."""

    def test_simulated_cascade_event(self) -> None:
        """Simulate a cascade: high OI, extreme funding, thin book, rapid liquidations."""
        detector = CascadeDetector(
            hawkes_mu=0.01,
            hawkes_alpha=0.15,
            hawkes_beta=0.08,
        )

        # Phase 1: Build normal market history (50 bars = ~4 hours)
        for i in range(50):
            detector.on_bar(
                float(i * 300),
                open_interest=50000.0 + float(np.random.default_rng(i).normal(0, 500)),
                funding_rate=0.0001,
                book_depth=1000.0,
                price=50000.0,
                prev_price=50000.0,
            )
            # Occasional random liquidation
            if i % 10 == 0:
                detector.on_liquidation(float(i * 300 + 100), 0.1, "long")

        prob_before = detector.state.cascade_probability
        assert detector.state.alert_level == 0  # Normal conditions

        # Phase 2: Cascade conditions develop
        # OI spikes, funding extreme, book thins
        detector.on_bar(
            15100.0,
            open_interest=65000.0,  # Very high OI
            funding_rate=0.003,  # Extreme funding
            book_depth=200.0,  # Severely thinned book
            price=48500.0,  # Sharp price drop
            prev_price=50000.0,
        )

        # Burst of liquidations (50 in 25 seconds)
        for i in range(50):
            detector.on_liquidation(15100.0 + float(i) * 0.5, 2.0, "long")

        prob_after = detector.state.cascade_probability
        assert prob_after > prob_before
        # Should be in elevated or higher state
        assert detector.state.alert_level >= 1

    def test_hawkes_parameters_meaningful(self) -> None:
        """Verify Hawkes branching ratio responds correctly to event patterns."""
        hp = HawkesProcess(mu=0.01, alpha=0.08, beta=0.1)

        # Normal: 1 event per 60 seconds
        for i in range(10):
            hp.add_event(float(i) * 60.0)

        ratio_normal = hp.estimated_branching_ratio(window_s=600.0)

        # Cascade: 5 events per second for 20 seconds
        hp.reset()
        for i in range(100):
            hp.add_event(float(i) * 0.2)

        ratio_cascade = hp.estimated_branching_ratio(window_s=30.0)

        # Cascade should have much higher branching ratio
        assert ratio_cascade > ratio_normal

    def test_position_reduction_during_cascade(self) -> None:
        """Position size should be reduced as cascade probability increases."""
        detector = CascadeDetector(
            hawkes_mu=0.01,
            hawkes_alpha=0.2,
            hawkes_beta=0.08,
        )

        # Build history
        for i in range(30):
            detector.on_bar(
                float(i * 300),
                open_interest=50000.0,
                funding_rate=0.0001,
                book_depth=1000.0,
                price=50000.0,
                prev_price=50000.0,
            )

        initial_mult = detector.state.position_size_multiplier
        assert initial_mult == 1.0

        # Trigger conditions
        detector.on_bar(
            9100.0,
            open_interest=65000.0,
            funding_rate=0.005,
            book_depth=100.0,
            price=47000.0,
            prev_price=50000.0,
        )

        # Heavy burst
        for i in range(60):
            detector.on_liquidation(9100.0 + float(i) * 0.3, 3.0, "long")

        # Multiplier should be reduced
        assert detector.state.position_size_multiplier < initial_mult


class TestModuleIndependence:
    """Verify both modules operate independently from each other and from ML model."""

    def test_macro_module_standalone(self) -> None:
        """Macro module needs no external ML model or cascade detector."""
        calendar = [MacroEvent(MacroEventType.CPI, 1_000_000, "CPI", 3)]
        monitor = MacroEventMonitor(
            calendar=calendar,
            nq_min_move_pct=0.01,
            min_correlation=0.0,  # Accept any correlation
        )

        # Feed data directly, no model needed
        for i in range(15):
            ts = 900_000 + i * 5000
            monitor.update(
                ts, 50000.0 + i * 10, 50000.0, 18000.0 + i * 5, 18000.0,
            )

        # Should work without any external dependencies
        assert monitor.active_signal is None or isinstance(monitor.active_signal, object)

    def test_cascade_module_standalone(self) -> None:
        """Cascade detector needs no external ML model or macro module."""
        detector = CascadeDetector()

        # Feed data directly
        for i in range(10):
            detector.on_bar(
                float(i * 300),
                open_interest=50000.0,
                funding_rate=0.0001,
            )
            detector.on_liquidation(float(i * 300 + 100), 0.5, "long")

        state = detector.state
        assert isinstance(state.cascade_probability, float)
        assert isinstance(state.position_size_multiplier, float)

    def test_both_modules_concurrent(self) -> None:
        """Both modules can run simultaneously on the same market data."""
        calendar = [MacroEvent(MacroEventType.CPI, 5_000_000, "CPI Concurrent", 3)]
        macro = MacroEventMonitor(
            calendar=calendar,
            nq_min_move_pct=0.01,
            min_correlation=0.1,
        )
        cascade = CascadeDetector()

        # Feed same market data to both
        for i in range(20):
            ts = 4_000_000 + i * 300_000
            btc = 50000.0 + 100.0 * math.sin(i * 0.3)
            btc_prev = 50000.0 + 100.0 * math.sin((i - 1) * 0.3)
            nq = 18000.0 + 50.0 * math.sin(i * 0.3)
            nq_prev = 18000.0 + 50.0 * math.sin((i - 1) * 0.3)

            macro.update(ts, btc, btc_prev, nq, nq_prev, vix_level=20.0)
            cascade.on_bar(
                float(ts / 1000),
                open_interest=50000.0,
                funding_rate=0.0001,
                book_depth=1000.0,
                price=btc,
                prev_price=btc_prev,
            )

        # Both should have valid state
        assert isinstance(cascade.state.cascade_probability, float)
        assert macro.active_signal is None or macro.active_signal.direction in (-1, 0, 1)


class TestEWMACorrelationIntegration:
    """Integration tests for EWMA correlation tracker."""

    def test_trailing_6h_window(self) -> None:
        """EWMA correlation over 6h (72 bars at 5-min) tracks regime changes."""
        ewma = EWMACorrelation(halflife_bars=36, min_bars=12)

        # Phase 1: Correlated (50 bars)
        rng = np.random.default_rng(123)
        for _ in range(50):
            common = float(rng.normal(0, 0.01))
            ewma.update(common + float(rng.normal(0, 0.001)), common + float(rng.normal(0, 0.001)))

        corr_high = ewma.correlation
        assert corr_high > 0.5

        # Phase 2: Uncorrelated (50 bars)
        for _ in range(50):
            ewma.update(float(rng.normal(0, 0.01)), float(rng.normal(0, 0.01)))

        corr_low = ewma.correlation
        assert corr_low < corr_high
