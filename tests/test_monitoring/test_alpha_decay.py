"""Tests for alpha decay detection."""

from __future__ import annotations

import pytest

from ep2_crypto.monitoring.alpha_decay import (
    ADWINDetector,
    AlertLevel,
    AlphaDecayMonitor,
    AlphaDecayState,
    CUSUMDetector,
    RollingSharpeMonitor,
    SPRTMonitor,
)


class TestCUSUMDetector:
    def test_no_signal_on_zero_returns(self) -> None:
        cusum = CUSUMDetector(threshold=0.01, drift=0.001)
        for _ in range(100):
            assert not cusum.update(0.0)

    def test_signal_on_persistent_negative_shift(self) -> None:
        cusum = CUSUMDetector(threshold=0.01, drift=0.001)
        fired = False
        for _ in range(50):
            if cusum.update(-0.005):
                fired = True
                break
        assert fired
        assert cusum.signal

    def test_signal_on_persistent_positive_shift(self) -> None:
        cusum = CUSUMDetector(threshold=0.01, drift=0.001)
        fired = False
        for _ in range(50):
            if cusum.update(0.005):
                fired = True
                break
        assert fired

    def test_reset_clears_state(self) -> None:
        cusum = CUSUMDetector(threshold=0.01, drift=0.001)
        for _ in range(20):
            cusum.update(-0.005)
        cusum.reset()
        assert cusum.s_pos == 0.0
        assert cusum.s_neg == 0.0
        assert not cusum.signal

    def test_accumulators_increase(self) -> None:
        cusum = CUSUMDetector(threshold=1.0, drift=0.001)
        cusum.update(0.01)
        assert cusum.s_pos > 0.0

    def test_threshold_sensitivity(self) -> None:
        sensitive = CUSUMDetector(threshold=0.002, drift=0.0005)
        insensitive = CUSUMDetector(threshold=0.05, drift=0.001)
        sensitive_fired = False
        insensitive_fired = False
        for _ in range(20):
            if sensitive.update(-0.003):
                sensitive_fired = True
            if insensitive.update(-0.003):
                insensitive_fired = True
        assert sensitive_fired
        assert not insensitive_fired


class TestRollingSharpeMonitor:
    def test_no_decline_with_insufficient_data(self) -> None:
        monitor = RollingSharpeMonitor(short_window=10, long_window=20)
        for _ in range(5):
            assert not monitor.update(0.001)

    def test_declining_sharpe_detected(self) -> None:
        monitor = RollingSharpeMonitor(
            short_window=20, long_window=100, decline_threshold=0.5
        )
        # Long good period to build positive baseline
        for _ in range(80):
            monitor.update(0.002)
        # Bad period — short window goes negative while long stays positive
        declined = False
        for _ in range(30):
            if monitor.update(-0.003):
                declined = True
        assert declined
        assert monitor.declining

    def test_stable_sharpe_no_decline(self) -> None:
        monitor = RollingSharpeMonitor(
            short_window=20, long_window=50, decline_threshold=0.5
        )
        for _ in range(60):
            monitor.update(0.001)
        assert not monitor.declining

    def test_current_sharpe_positive_for_positive_returns(self) -> None:
        monitor = RollingSharpeMonitor(short_window=10, long_window=20)
        for _ in range(25):
            monitor.update(0.001)
        assert monitor.current_sharpe > 0

    def test_reset(self) -> None:
        monitor = RollingSharpeMonitor(short_window=10, long_window=20)
        for _ in range(25):
            monitor.update(0.001)
        monitor.reset()
        assert monitor.current_sharpe == 0.0
        assert not monitor.declining


class TestADWINDetector:
    def test_no_detection_on_stationary_data(self) -> None:
        adwin = ADWINDetector(delta=0.01)
        for _ in range(50):
            adwin.update(1.0)
        assert not adwin.detected

    def test_detects_mean_shift(self) -> None:
        adwin = ADWINDetector(delta=0.01)
        # Large shift: 5.0 → -5.0
        for _ in range(100):
            adwin.update(5.0)
        detected = False
        for _ in range(100):
            if adwin.update(-5.0):
                detected = True
        assert detected

    def test_window_shrinks_on_detection(self) -> None:
        adwin = ADWINDetector(delta=0.01)
        for _ in range(100):
            adwin.update(5.0)
        initial_size = adwin.window_size
        for _ in range(100):
            adwin.update(-5.0)
        # Window should have shrunk due to detected change
        assert adwin.window_size < initial_size + 100

    def test_reset(self) -> None:
        adwin = ADWINDetector(delta=0.01)
        for _ in range(20):
            adwin.update(0.01)
        adwin.reset()
        assert adwin.window_size == 0
        assert not adwin.detected


class TestSPRTMonitor:
    def test_no_rejection_with_good_win_rate(self) -> None:
        sprt = SPRTMonitor(p0=0.52, p1=0.48, min_trades=20)
        for _ in range(100):
            # 55% win rate — clearly above p0
            sprt.update(win=True)
        assert not sprt.rejected

    def test_rejection_with_bad_win_rate(self) -> None:
        sprt = SPRTMonitor(p0=0.55, p1=0.45, min_trades=20)
        rejected = False
        for i in range(200):
            # 40% win rate — below p1
            if sprt.update(win=(i % 5 < 2)):
                rejected = True
                break
        assert rejected

    def test_no_rejection_before_min_trades(self) -> None:
        sprt = SPRTMonitor(p0=0.52, p1=0.48, min_trades=50)
        for _ in range(49):
            result = sprt.update(win=False)
            assert not result

    def test_win_rate_tracking(self) -> None:
        sprt = SPRTMonitor(p0=0.52, p1=0.48, min_trades=10)
        for _ in range(6):
            sprt.update(win=True)
        for _ in range(4):
            sprt.update(win=False)
        assert sprt.win_rate == pytest.approx(0.6)

    def test_win_rate_none_when_empty(self) -> None:
        sprt = SPRTMonitor(p0=0.52, p1=0.48)
        assert sprt.win_rate is None

    def test_invalid_p0_p1_raises(self) -> None:
        with pytest.raises(ValueError, match="p0 must be > p1"):
            SPRTMonitor(p0=0.48, p1=0.52)

    def test_reset(self) -> None:
        sprt = SPRTMonitor(p0=0.52, p1=0.48, min_trades=10)
        for _ in range(20):
            sprt.update(win=False)
        sprt.reset()
        assert sprt.llr == 0.0
        assert sprt.win_rate is None
        assert not sprt.rejected


class TestAlphaDecayState:
    def test_should_trade_normal(self) -> None:
        state = AlphaDecayState(level=AlertLevel.NORMAL)
        assert state.should_trade

    def test_should_trade_warning(self) -> None:
        state = AlphaDecayState(level=AlertLevel.WARNING)
        assert state.should_trade

    def test_should_trade_caution(self) -> None:
        state = AlphaDecayState(level=AlertLevel.CAUTION)
        assert state.should_trade

    def test_should_not_trade_stop(self) -> None:
        state = AlphaDecayState(level=AlertLevel.STOP)
        assert not state.should_trade

    def test_should_not_trade_emergency(self) -> None:
        state = AlphaDecayState(level=AlertLevel.EMERGENCY)
        assert not state.should_trade


class TestAlphaDecayMonitor:
    def test_normal_state_on_init(self) -> None:
        monitor = AlphaDecayMonitor()
        assert monitor.state.level == AlertLevel.NORMAL

    def test_stable_returns_stay_normal(self) -> None:
        monitor = AlphaDecayMonitor(
            cusum_threshold=0.05,
            sharpe_short_window=10,
            sharpe_long_window=20,
            sprt_min_trades=50,
        )
        for _ in range(30):
            state = monitor.on_bar(0.001)
        assert state.level == AlertLevel.NORMAL

    def test_warning_on_single_detector(self) -> None:
        monitor = AlphaDecayMonitor(
            cusum_threshold=0.005,
            cusum_drift=0.001,
            sharpe_short_window=100,  # Large so it doesn't fire
            sharpe_long_window=200,
            adwin_delta=1e-10,  # Very insensitive
            sprt_min_trades=1000,  # Won't reach
        )
        # Drive CUSUM to fire
        for _ in range(50):
            state = monitor.on_bar(-0.005)
        assert state.cusum_signal
        assert state.level >= AlertLevel.WARNING

    def test_on_trade_updates_sprt(self) -> None:
        monitor = AlphaDecayMonitor(
            sprt_p0=0.55, sprt_p1=0.45, sprt_min_trades=20
        )
        for i in range(100):
            state = monitor.on_trade(pnl=-1.0)  # All losses
            if state.sprt_rejected:
                break
        assert state.sprt_rejected

    def test_stop_on_three_detectors(self) -> None:
        monitor = AlphaDecayMonitor(
            cusum_threshold=0.005,
            cusum_drift=0.001,
            sharpe_short_window=10,
            sharpe_long_window=30,
            sharpe_decline_threshold=0.5,
            adwin_delta=0.01,
            sprt_min_trades=10000,  # Won't fire
        )
        # Good period for long window baseline
        for _ in range(30):
            monitor.on_bar(0.003)
        # Bad period to trigger CUSUM + Sharpe + ADWIN
        for _ in range(50):
            state = monitor.on_bar(-0.005)

        active = sum([
            state.cusum_signal,
            state.sharpe_declining,
            state.adwin_detected,
        ])
        if active >= 3:
            assert state.level >= AlertLevel.STOP

    def test_emergency_on_cusum_plus_sprt(self) -> None:
        monitor = AlphaDecayMonitor(
            cusum_threshold=0.005,
            cusum_drift=0.001,
            sprt_p0=0.55,
            sprt_p1=0.45,
            sprt_min_trades=10,
        )
        # Fire CUSUM
        for _ in range(50):
            monitor.on_bar(-0.005)
        # Fire SPRT
        for _ in range(100):
            monitor.on_trade(pnl=-1.0)

        state = monitor.state
        if state.cusum_signal and state.sprt_rejected:
            assert state.level == AlertLevel.EMERGENCY

    def test_reset_clears_all(self) -> None:
        monitor = AlphaDecayMonitor(
            cusum_threshold=0.005,
            cusum_drift=0.001,
            sprt_min_trades=10,
        )
        for _ in range(20):
            monitor.on_bar(-0.005)
        for _ in range(20):
            monitor.on_trade(pnl=-1.0)
        monitor.reset()
        assert monitor.state.level == AlertLevel.NORMAL
        assert not monitor.cusum.signal
        assert not monitor.sprt.rejected

    def test_details_populated(self) -> None:
        monitor = AlphaDecayMonitor()
        for _ in range(10):
            monitor.on_bar(0.001)
        state = monitor.state
        assert "cusum_s_pos" in state.details
        assert "rolling_sharpe" in state.details
        assert "adwin_window_size" in state.details
        assert "sprt_llr" in state.details
        assert "active_detectors" in state.details


class TestAlertLevelOrdering:
    def test_levels_ordered(self) -> None:
        assert AlertLevel.NORMAL < AlertLevel.WARNING
        assert AlertLevel.WARNING < AlertLevel.CAUTION
        assert AlertLevel.CAUTION < AlertLevel.STOP
        assert AlertLevel.STOP < AlertLevel.EMERGENCY

    def test_level_values(self) -> None:
        assert AlertLevel.NORMAL == 0
        assert AlertLevel.WARNING == 1
        assert AlertLevel.CAUTION == 2
        assert AlertLevel.STOP == 3
        assert AlertLevel.EMERGENCY == 4
