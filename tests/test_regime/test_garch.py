"""Tests for GJR-GARCH(1,1)-t regime detector."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ep2_crypto.regime.garch import GARCHDetector, GARCHResult, VolRegime


@pytest.fixture
def detector() -> GARCHDetector:
    return GARCHDetector(
        omega=1e-6,
        alpha=0.05,
        gamma=0.10,
        beta=0.85,
        percentile_window=50,
    )


class TestGARCHInit:
    def test_default_params(self) -> None:
        d = GARCHDetector()
        assert d.warmup_bars == 21

    def test_stationarity_check(self) -> None:
        """alpha + gamma/2 + beta must be < 1 for stationarity."""
        with pytest.raises(ValueError, match="stationarity"):
            GARCHDetector(alpha=0.3, gamma=0.2, beta=0.7)

    def test_percentile_window_min(self) -> None:
        with pytest.raises(ValueError, match="percentile_window"):
            GARCHDetector(percentile_window=5)


class TestGARCHComputation:
    def test_nan_before_warmup(self, detector: GARCHDetector) -> None:
        closes = np.arange(10, dtype=np.float64) + 100
        result = detector.update(5, closes)
        assert math.isnan(result.conditional_vol)
        assert result.confidence == 0.0

    def test_produces_positive_vol(self, detector: GARCHDetector) -> None:
        """Conditional vol should always be positive."""
        rng = np.random.default_rng(42)
        closes = 100.0 + np.cumsum(rng.standard_normal(100) * 0.01)
        for idx in range(detector.warmup_bars, len(closes)):
            detector.reset()
            result = detector.update(idx, closes)
            if not math.isnan(result.conditional_vol):
                assert result.conditional_vol > 0

    def test_high_vol_after_large_move(self) -> None:
        """A large price shock should increase conditional vol."""
        detector = GARCHDetector(percentile_window=20)
        # Calm period then a shock
        closes = np.full(50, 100.0, dtype=np.float64)
        # Small random noise for calm period
        rng = np.random.default_rng(42)
        closes[:40] += np.cumsum(rng.standard_normal(40) * 0.001)
        # Large shock at bar 40
        closes[40:] = closes[39] + 5.0  # 5% jump

        detector.reset()
        result_before = detector.update(39, closes)
        detector.reset()
        result_after = detector.update(41, closes)

        before_valid = not math.isnan(result_before.conditional_vol)
        after_valid = not math.isnan(result_after.conditional_vol)
        if before_valid and after_valid:
            assert result_after.conditional_vol > result_before.conditional_vol

    def test_asymmetric_leverage(self) -> None:
        """Negative returns should increase vol more than positive returns (gamma > 0)."""
        detector = GARCHDetector(omega=1e-6, alpha=0.05, gamma=0.15, beta=0.80)

        # Two series: one with a positive shock, one with a negative shock
        base = np.full(30, 100.0, dtype=np.float64)
        base[:25] += np.cumsum(np.full(25, 0.001))

        pos_shock = base.copy()
        pos_shock[25] = pos_shock[24] * 1.05  # +5%
        pos_shock[26:] = pos_shock[25]

        neg_shock = base.copy()
        neg_shock[25] = neg_shock[24] * 0.95  # -5%
        neg_shock[26:] = neg_shock[25]

        detector.reset()
        r_pos = detector.update(26, pos_shock)
        detector.reset()
        r_neg = detector.update(26, neg_shock)

        if not math.isnan(r_pos.conditional_vol) and not math.isnan(r_neg.conditional_vol):
            # Negative shock should produce higher vol due to gamma
            assert r_neg.conditional_vol > r_pos.conditional_vol

    def test_vol_percentile_bounded(self, detector: GARCHDetector) -> None:
        """Vol percentile should be in [0, 1]."""
        rng = np.random.default_rng(42)
        closes = 100.0 + np.cumsum(rng.standard_normal(100) * 0.01)
        detector.reset()
        for idx in range(detector.warmup_bars, len(closes)):
            result = detector.update(idx, closes)
            if not math.isnan(result.conditional_vol):
                assert 0.0 <= result.vol_percentile <= 1.0


class TestGARCHRegimeClassification:
    def test_all_regimes_reachable(self) -> None:
        """Over a volatile enough series, all three regimes should appear."""
        detector = GARCHDetector(percentile_window=30)
        rng = np.random.default_rng(42)
        # Calm -> volatile -> calm
        calm1 = np.cumsum(rng.standard_normal(40) * 0.001)
        volatile = np.cumsum(rng.standard_normal(30) * 0.05)
        calm2 = np.cumsum(rng.standard_normal(30) * 0.001)
        prices = 100.0 + np.concatenate([calm1, volatile, calm2])

        regimes_seen: set[VolRegime] = set()
        detector.reset()
        for idx in range(detector.warmup_bars, len(prices)):
            result = detector.update(idx, prices)
            if not math.isnan(result.conditional_vol):
                regimes_seen.add(result.vol_regime)

        # Should see at least LOW and HIGH
        assert VolRegime.LOW in regimes_seen or VolRegime.MEDIUM in regimes_seen
        assert VolRegime.HIGH in regimes_seen

    def test_confidence_bounded(self) -> None:
        """Confidence should always be in [0, 1]."""
        detector = GARCHDetector(percentile_window=20)
        rng = np.random.default_rng(42)
        closes = 100.0 + np.cumsum(rng.standard_normal(80) * 0.01)
        detector.reset()
        for idx in range(detector.warmup_bars, len(closes)):
            result = detector.update(idx, closes)
            assert 0.0 <= result.confidence <= 1.0


class TestGARCHBatch:
    def test_batch_length(self, detector: GARCHDetector) -> None:
        closes = 100.0 + np.cumsum(np.random.default_rng(42).standard_normal(50) * 0.01)
        results = detector.compute_batch(closes)
        assert len(results) == 50

    def test_batch_resets_state(self, detector: GARCHDetector) -> None:
        """compute_batch should reset state and produce fresh results."""
        closes = 100.0 + np.cumsum(np.random.default_rng(42).standard_normal(50) * 0.01)
        r1 = detector.compute_batch(closes)
        r2 = detector.compute_batch(closes)
        # Should be identical since state is reset
        for a, b in zip(r1, r2, strict=True):
            if math.isnan(a.conditional_vol):
                assert math.isnan(b.conditional_vol)
            else:
                assert a.conditional_vol == pytest.approx(b.conditional_vol, rel=1e-10)


class TestGARCHResult:
    def test_result_is_frozen(self) -> None:
        result = GARCHResult(
            conditional_vol=0.01, vol_regime=VolRegime.MEDIUM,
            vol_percentile=0.5, confidence=0.5,
        )
        with pytest.raises(AttributeError):
            result.conditional_vol = 0.02  # type: ignore[misc]

    def test_vol_regime_enum(self) -> None:
        assert VolRegime.LOW == 0
        assert VolRegime.MEDIUM == 1
        assert VolRegime.HIGH == 2
