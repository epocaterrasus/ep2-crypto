"""Tests for triple barrier labeling module."""

from __future__ import annotations

import numpy as np
import pytest

from ep2_crypto.models.labeling import (
    BarrierConfig,
    Direction,
    compute_atr,
    compute_barriers,
    compute_class_weights,
    compute_rolling_vol,
    label_fixed_threshold,
    label_triple_barrier,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def trending_up_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Strong uptrend: each bar closes higher with clear direction."""
    n = 100
    rng = np.random.default_rng(42)
    base = 50000.0 + np.cumsum(rng.uniform(5, 15, size=n))
    noise = rng.uniform(-2, 2, size=n)
    closes = base + noise
    highs = closes + rng.uniform(3, 10, size=n)
    lows = closes - rng.uniform(3, 10, size=n)
    return closes, highs, lows


@pytest.fixture
def trending_down_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Strong downtrend: each bar closes lower."""
    n = 100
    rng = np.random.default_rng(43)
    base = 50000.0 - np.cumsum(rng.uniform(5, 15, size=n))
    noise = rng.uniform(-2, 2, size=n)
    closes = base + noise
    highs = closes + rng.uniform(3, 10, size=n)
    lows = closes - rng.uniform(3, 10, size=n)
    return closes, highs, lows


@pytest.fixture
def sideways_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sideways market: oscillates around 50000 with tiny moves."""
    n = 100
    rng = np.random.default_rng(44)
    closes = 50000.0 + rng.normal(0, 1, size=n)  # Very small moves
    highs = closes + rng.uniform(0.5, 2, size=n)
    lows = closes - rng.uniform(0.5, 2, size=n)
    return closes, highs, lows


# ---------------------------------------------------------------------------
# ATR tests
# ---------------------------------------------------------------------------


class TestComputeATR:
    def test_basic_atr(self) -> None:
        """ATR is computed correctly with Wilder's smoothing."""
        highs = np.array([102, 104, 103, 105, 106, 104, 107], dtype=np.float64)
        lows = np.array([98, 99, 97, 100, 101, 99, 102], dtype=np.float64)
        closes = np.array([100, 102, 100, 103, 104, 101, 105], dtype=np.float64)

        atr = compute_atr(highs, lows, closes, window=3)

        # First 2 bars should be NaN
        assert np.isnan(atr[0])
        assert np.isnan(atr[1])
        # Bar 2 (index=2): simple average of first 3 TRs
        assert np.isfinite(atr[2])
        # Subsequent bars use Wilder's smoothing
        for i in range(3, len(atr)):
            assert np.isfinite(atr[i])
        # ATR should be positive
        assert np.all(atr[np.isfinite(atr)] > 0)

    def test_atr_too_few_bars(self) -> None:
        """ATR returns all NaN when fewer bars than window."""
        highs = np.array([102, 104], dtype=np.float64)
        lows = np.array([98, 99], dtype=np.float64)
        closes = np.array([100, 102], dtype=np.float64)
        atr = compute_atr(highs, lows, closes, window=5)
        assert np.all(np.isnan(atr))

    def test_atr_constant_prices(self) -> None:
        """ATR should be zero for constant prices."""
        n = 20
        closes = np.full(n, 100.0, dtype=np.float64)
        atr = compute_atr(closes, closes, closes, window=5)
        valid = atr[np.isfinite(atr)]
        assert np.allclose(valid, 0.0)


class TestComputeRollingVol:
    def test_basic_vol(self) -> None:
        """Rolling vol produces valid values after warmup."""
        rng = np.random.default_rng(42)
        closes = 50000.0 * np.exp(np.cumsum(rng.normal(0, 0.001, size=50)))
        vol = compute_rolling_vol(closes, window=10)

        # First 10 bars should be NaN
        assert np.all(np.isnan(vol[:10]))
        # After warmup, should be positive
        valid = vol[np.isfinite(vol)]
        assert len(valid) > 0
        assert np.all(valid > 0)

    def test_constant_prices_zero_vol(self) -> None:
        """Constant prices yield zero volatility."""
        closes = np.full(30, 100.0, dtype=np.float64)
        vol = compute_rolling_vol(closes, window=10)
        valid = vol[np.isfinite(vol)]
        assert np.allclose(valid, 0.0)


# ---------------------------------------------------------------------------
# Barrier computation tests
# ---------------------------------------------------------------------------


class TestComputeBarriers:
    def test_barriers_positive(self, trending_up_data: tuple) -> None:
        """Barriers should be positive after warmup."""
        closes, highs, lows = trending_up_data
        config = BarrierConfig(vol_window=10)
        upper, lower = compute_barriers(closes, highs, lows, config)

        valid_upper = upper[np.isfinite(upper)]
        valid_lower = lower[np.isfinite(lower)]
        assert len(valid_upper) > 0
        assert np.all(valid_upper > 0)
        assert np.all(valid_lower > 0)

    def test_min_barrier_enforced(self) -> None:
        """Minimum barrier width in bps is enforced."""
        closes = np.full(30, 50000.0, dtype=np.float64)
        highs = closes + 1
        lows = closes - 1
        config = BarrierConfig(vol_window=5, min_barrier_bps=10.0, use_atr=True)
        upper, _lower = compute_barriers(closes, highs, lows, config)

        min_expected = 50000.0 * 10.0 / 10_000.0  # = 50 USD
        valid = upper[np.isfinite(upper)]
        assert np.all(valid >= min_expected - 1e-10)

    def test_asymmetric_multipliers(self) -> None:
        """Different upper/lower multipliers produce different barriers."""
        rng = np.random.default_rng(42)
        closes = 50000.0 + np.cumsum(rng.normal(0, 10, size=50))
        highs = closes + rng.uniform(5, 20, size=50)
        lows = closes - rng.uniform(5, 20, size=50)
        config = BarrierConfig(
            vol_window=10,
            upper_multiplier=1.5,
            lower_multiplier=0.8,
            min_barrier_bps=0.1,
        )
        upper, lower = compute_barriers(closes, highs, lows, config)

        # Where both are valid, upper should be larger than lower
        mask = np.isfinite(upper) & np.isfinite(lower)
        assert np.all(upper[mask] > lower[mask])


# ---------------------------------------------------------------------------
# Triple barrier labeling tests
# ---------------------------------------------------------------------------


class TestTripleBarrierLabeling:
    def test_uptrend_mostly_up(self, trending_up_data: tuple) -> None:
        """Strong uptrend should produce majority UP labels."""
        closes, highs, lows = trending_up_data
        config = BarrierConfig(max_holding_bars=12, vol_window=10)
        labels, _returns_at_exit, _hold_periods = label_triple_barrier(
            closes,
            highs,
            lows,
            config,
        )

        # After warmup, majority should be UP
        warmup = config.vol_window
        post_warmup = labels[warmup : -config.max_holding_bars]
        up_count = np.sum(post_warmup == Direction.UP)
        assert up_count > len(post_warmup) * 0.5, (
            f"Expected majority UP in uptrend, got {up_count}/{len(post_warmup)}"
        )

    def test_downtrend_mostly_down(self, trending_down_data: tuple) -> None:
        """Strong downtrend should produce majority DOWN labels."""
        closes, highs, lows = trending_down_data
        config = BarrierConfig(max_holding_bars=12, vol_window=10)
        labels, _, _ = label_triple_barrier(closes, highs, lows, config)

        warmup = config.vol_window
        post_warmup = labels[warmup : -config.max_holding_bars]
        down_count = np.sum(post_warmup == Direction.DOWN)
        assert down_count > len(post_warmup) * 0.5

    def test_valid_label_values(self, trending_up_data: tuple) -> None:
        """All labels must be -1, 0, or +1."""
        closes, highs, lows = trending_up_data
        labels, _, _ = label_triple_barrier(closes, highs, lows)
        assert set(np.unique(labels)).issubset({-1, 0, 1})

    def test_hold_periods_bounded(self, trending_up_data: tuple) -> None:
        """Hold periods should never exceed max_holding_bars."""
        closes, highs, lows = trending_up_data
        config = BarrierConfig(max_holding_bars=8)
        _, _, hold_periods = label_triple_barrier(closes, highs, lows, config)
        assert np.all(hold_periods <= config.max_holding_bars)

    def test_returns_sign_matches_label(self, trending_up_data: tuple) -> None:
        """Returns at exit should match label sign (when barrier hit, not vertical)."""
        closes, highs, lows = trending_up_data
        config = BarrierConfig(max_holding_bars=12, vol_window=10)
        labels, returns_at_exit, hold_periods = label_triple_barrier(
            closes,
            highs,
            lows,
            config,
        )

        # For barrier-touch exits (not at max holding period), sign should match
        for i in range(len(labels)):
            if labels[i] == Direction.UP and hold_periods[i] < config.max_holding_bars:
                assert returns_at_exit[i] > 0, f"UP label at {i} has negative return"
            if labels[i] == Direction.DOWN and hold_periods[i] < config.max_holding_bars:
                assert returns_at_exit[i] < 0, f"DOWN label at {i} has positive return"

    def test_barrier_touch_order(self) -> None:
        """Verify that first barrier touched wins (using hand-crafted data)."""
        # Entry at 100. Upper barrier at 102, lower at 98.
        # Bar 1: high=101, low=99 — no touch
        # Bar 2: high=103, low=99.5 — upper barrier touched first
        closes = np.array(
            [100, 100.5, 102.5, 103, 104],
            dtype=np.float64,
        )
        highs = np.array(
            [100.5, 101, 103, 104, 105],
            dtype=np.float64,
        )
        lows = np.array(
            [99.5, 99, 99.5, 100, 101],
            dtype=np.float64,
        )

        # Use fixed barriers by setting vol_window=1 with known ATR
        config = BarrierConfig(
            max_holding_bars=5,
            vol_window=1,
            upper_multiplier=2.0,
            lower_multiplier=2.0,
            min_barrier_bps=0.1,
        )
        labels, _, hold_periods = label_triple_barrier(closes, highs, lows, config)
        # Bar 0: entry=100, ATR at bar 0 = H-L = 1.0, barrier = 2.0
        # Upper = 102, Lower = 98
        # Bar 1: high=101 < 102, low=99 > 98 — no touch
        # Bar 2: high=103 >= 102 — UP!
        assert labels[0] == Direction.UP
        assert hold_periods[0] == 2

    def test_lower_barrier_touch(self) -> None:
        """Lower barrier touched first → DOWN label."""
        closes = np.array(
            [100, 99, 96, 95, 94],
            dtype=np.float64,
        )
        highs = np.array(
            [100.5, 100, 97, 96, 95],
            dtype=np.float64,
        )
        lows = np.array(
            [99.5, 97, 95, 93, 92],
            dtype=np.float64,
        )
        config = BarrierConfig(
            max_holding_bars=5,
            vol_window=1,
            upper_multiplier=2.0,
            lower_multiplier=2.0,
            min_barrier_bps=0.1,
        )
        labels, _, _ = label_triple_barrier(closes, highs, lows, config)
        # Bar 0: entry=100, ATR=1.0, barriers at 102/98
        # Bar 1: low=97 <= 98 → DOWN
        assert labels[0] == Direction.DOWN

    def test_vertical_barrier(self) -> None:
        """No barrier touched → vertical barrier, label from close return."""
        # Tiny moves, wide barriers, short holding period
        closes = np.array(
            [100, 100.1, 100.05, 100.2],
            dtype=np.float64,
        )
        highs = closes + 0.2
        lows = closes - 0.2
        config = BarrierConfig(
            max_holding_bars=2,
            vol_window=1,
            upper_multiplier=10.0,  # Very wide barriers
            lower_multiplier=10.0,
            min_barrier_bps=0.1,
        )
        _labels, returns_at_exit, hold_periods = label_triple_barrier(
            closes,
            highs,
            lows,
            config,
        )
        # Bar 0: barriers are ±10*ATR = ±4.0, so at 104/96
        # After 2 bars: close=100.05, return > 0 → UP via vertical
        assert hold_periods[0] == 2
        assert returns_at_exit[0] == pytest.approx(
            (100.05 - 100.0) / 100.0,
        )

    def test_end_of_array(self) -> None:
        """Last bars that cannot look forward get FLAT label."""
        closes = np.array([100, 101, 102], dtype=np.float64)
        highs = closes + 1
        lows = closes - 1
        config = BarrierConfig(max_holding_bars=5, vol_window=1)
        labels, _, hold_periods = label_triple_barrier(closes, highs, lows, config)
        # Last bar can't look forward at all
        assert labels[-1] == Direction.FLAT
        assert hold_periods[-1] == 0

    def test_output_shapes(self, trending_up_data: tuple) -> None:
        """All outputs have the same length as input."""
        closes, highs, lows = trending_up_data
        labels, returns_at_exit, hold_periods = label_triple_barrier(
            closes,
            highs,
            lows,
        )
        assert labels.shape == closes.shape
        assert returns_at_exit.shape == closes.shape
        assert hold_periods.shape == closes.shape

    def test_use_rolling_vol(self, trending_up_data: tuple) -> None:
        """Works correctly with rolling vol instead of ATR."""
        closes, highs, lows = trending_up_data
        config = BarrierConfig(use_atr=False, vol_window=10)
        labels, _, _ = label_triple_barrier(closes, highs, lows, config)
        assert set(np.unique(labels)).issubset({-1, 0, 1})


# ---------------------------------------------------------------------------
# Fixed threshold labeling tests
# ---------------------------------------------------------------------------


class TestFixedThreshold:
    def test_basic_labeling(self) -> None:
        """Fixed threshold correctly labels clear moves."""
        closes = np.array([100, 101, 100, 99, 100], dtype=np.float64)
        labels = label_fixed_threshold(closes, horizon=1, threshold_bps=50)
        # 100→101 = +100bps → UP
        assert labels[0] == Direction.UP
        # 101→100 = -99bps → DOWN
        assert labels[1] == Direction.DOWN
        # 100→99 = -100bps → DOWN
        assert labels[2] == Direction.DOWN
        # 99→100 = +101bps → UP
        assert labels[3] == Direction.UP

    def test_flat_within_threshold(self) -> None:
        """Moves within threshold are labeled FLAT."""
        closes = np.array([100, 100.01, 100.02], dtype=np.float64)
        labels = label_fixed_threshold(closes, horizon=1, threshold_bps=50)
        # 100→100.01 = 1bp → FLAT
        assert labels[0] == Direction.FLAT

    def test_last_bars_zero(self) -> None:
        """Last `horizon` bars have no forward data → FLAT."""
        closes = np.array([100, 101, 102, 103, 104], dtype=np.float64)
        labels = label_fixed_threshold(closes, horizon=2, threshold_bps=10)
        assert labels[-1] == Direction.FLAT
        assert labels[-2] == Direction.FLAT


# ---------------------------------------------------------------------------
# Class weights tests
# ---------------------------------------------------------------------------


class TestClassWeights:
    def test_balanced_data(self) -> None:
        """Equal class sizes → equal weights."""
        labels = np.array([-1, 0, 1, -1, 0, 1], dtype=np.int8)
        weights = compute_class_weights(labels)
        assert len(weights) == 3
        for w in weights.values():
            assert w == pytest.approx(1.0)

    def test_imbalanced_data(self) -> None:
        """Rare class gets higher weight."""
        labels = np.array(
            [-1, -1, -1, -1, 0, 0, 1],
            dtype=np.int8,
        )
        weights = compute_class_weights(labels)
        # DOWN is most common (4/7), UP is rarest (1/7)
        assert weights[1] > weights[-1]
        assert weights[1] > weights[0]

    def test_weights_sum(self) -> None:
        """Weighted sample count should equal total for each class."""
        labels = np.array([-1, -1, 0, 0, 0, 1], dtype=np.int8)
        weights = compute_class_weights(labels)
        # n / (n_classes * count) * count = n / n_classes for each class
        for label_val in [-1, 0, 1]:
            count = np.sum(labels == label_val)
            assert weights[label_val] * count == pytest.approx(
                len(labels) / 3,
            )
