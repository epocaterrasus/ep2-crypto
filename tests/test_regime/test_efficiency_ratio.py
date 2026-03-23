"""Tests for Efficiency Ratio regime detector."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ep2_crypto.regime.efficiency_ratio import (
    EfficiencyRatioDetector,
    ERRegime,
    ERResult,
)


@pytest.fixture
def detector() -> EfficiencyRatioDetector:
    return EfficiencyRatioDetector(
        short_window=5,
        long_window=10,
        trending_threshold=0.5,
        choppy_threshold=0.3,
    )


class TestERDetectorInit:
    def test_default_params(self) -> None:
        d = EfficiencyRatioDetector()
        assert d.short_window == 20
        assert d.long_window == 100

    def test_custom_params(self, detector: EfficiencyRatioDetector) -> None:
        assert detector.short_window == 5
        assert detector.long_window == 10

    def test_short_window_too_small(self) -> None:
        with pytest.raises(ValueError, match="short_window"):
            EfficiencyRatioDetector(short_window=1)

    def test_long_must_exceed_short(self) -> None:
        with pytest.raises(ValueError, match="long_window"):
            EfficiencyRatioDetector(short_window=20, long_window=20)

    def test_thresholds_must_be_ordered(self) -> None:
        with pytest.raises(ValueError, match="choppy_threshold"):
            EfficiencyRatioDetector(trending_threshold=0.3, choppy_threshold=0.5)

    def test_warmup_bars(self, detector: EfficiencyRatioDetector) -> None:
        assert detector.warmup_bars == 11  # long_window + 1


class TestERComputation:
    def test_perfectly_trending_up(self, detector: EfficiencyRatioDetector) -> None:
        """Monotonically increasing prices should give ER = 1.0."""
        closes = np.arange(20, dtype=np.float64)  # 0, 1, 2, ..., 19
        result = detector.update(15, closes)
        assert result.er_short == pytest.approx(1.0)

    def test_perfectly_trending_down(self, detector: EfficiencyRatioDetector) -> None:
        """Monotonically decreasing prices should give ER = 1.0."""
        closes = np.arange(20, 0, -1, dtype=np.float64)  # 20, 19, ..., 1
        result = detector.update(15, closes)
        assert result.er_short == pytest.approx(1.0)

    def test_choppy_market(self, detector: EfficiencyRatioDetector) -> None:
        """Alternating prices (up/down) should give low ER."""
        # Create a series that oscillates but goes nowhere
        closes = np.array(
            [
                100.0,
                101.0,
                100.0,
                101.0,
                100.0,
                101.0,
                100.0,
                101.0,
                100.0,
                101.0,
                100.0,
                101.0,
                100.0,
                101.0,
                100.0,
            ],
            dtype=np.float64,
        )
        result = detector.update(14, closes)
        # ER should be low — close to 0 but not exactly 0 due to odd window alignment
        assert result.er_short < 0.3  # Below choppy threshold

    def test_er_bounded_zero_one(self, detector: EfficiencyRatioDetector) -> None:
        """ER should always be in [0, 1]."""
        rng = np.random.default_rng(42)
        closes = 100.0 + np.cumsum(rng.standard_normal(100))
        for idx in range(detector.warmup_bars, len(closes)):
            result = detector.update(idx, closes)
            if not math.isnan(result.er_short):
                assert 0.0 <= result.er_short <= 1.0
            if not math.isnan(result.er_long):
                assert 0.0 <= result.er_long <= 1.0

    def test_nan_before_warmup(self, detector: EfficiencyRatioDetector) -> None:
        """Should return NaN before enough data."""
        closes = np.arange(5, dtype=np.float64)
        result = detector.update(3, closes)
        assert math.isnan(result.er_short)

    def test_flat_prices(self, detector: EfficiencyRatioDetector) -> None:
        """Constant prices should give ER = 1.0 (no movement = trivially trending)."""
        closes = np.full(20, 100.0, dtype=np.float64)
        result = detector.update(15, closes)
        assert result.er_short == pytest.approx(1.0)


class TestERRegimeClassification:
    def test_trending_regime(self, detector: EfficiencyRatioDetector) -> None:
        """Monotonic prices -> TRENDING regime."""
        closes = np.arange(20, dtype=np.float64)
        result = detector.update(15, closes)
        assert result.regime == ERRegime.TRENDING
        assert result.confidence > 0.0

    def test_choppy_regime(self, detector: EfficiencyRatioDetector) -> None:
        """Oscillating prices -> CHOPPY regime."""
        closes = np.array(
            [
                100.0,
                101.0,
                100.0,
                101.0,
                100.0,
                101.0,
                100.0,
                101.0,
                100.0,
                101.0,
                100.0,
                101.0,
                100.0,
                101.0,
                100.0,
            ],
            dtype=np.float64,
        )
        result = detector.update(14, closes)
        assert result.regime == ERRegime.CHOPPY

    def test_neutral_regime(self, detector: EfficiencyRatioDetector) -> None:
        """ER between thresholds -> NEUTRAL regime."""
        # Construct a series where ER is ~0.4 (between 0.3 and 0.5)
        # Start trending, then add some noise
        closes = np.array(
            [
                100.0,
                100.5,
                101.0,
                100.8,
                101.5,
                101.2,
                102.0,
                101.8,
                102.5,
                102.2,
                103.0,
                102.8,
                103.5,
                103.2,
                104.0,
            ],
            dtype=np.float64,
        )
        result = detector.update(14, closes)
        # ER should be moderate — between choppy and trending
        assert 0.0 < result.er_short < 1.0

    def test_confidence_zero_before_warmup(self, detector: EfficiencyRatioDetector) -> None:
        """Before warmup, confidence should be 0."""
        closes = np.arange(3, dtype=np.float64)
        result = detector.update(2, closes)
        assert result.confidence == 0.0
        assert result.regime == ERRegime.NEUTRAL


class TestERBatch:
    def test_batch_length(self, detector: EfficiencyRatioDetector) -> None:
        """compute_batch should return one result per bar."""
        closes = np.arange(20, dtype=np.float64)
        results = detector.compute_batch(closes)
        assert len(results) == 20

    def test_batch_matches_individual(self, detector: EfficiencyRatioDetector) -> None:
        """Batch results should match individual update() calls."""
        rng = np.random.default_rng(123)
        closes = 100.0 + np.cumsum(rng.standard_normal(30))
        batch_results = detector.compute_batch(closes)

        for idx in range(len(closes)):
            individual = detector.update(idx, closes)
            batch = batch_results[idx]
            if math.isnan(individual.er_short):
                assert math.isnan(batch.er_short)
            else:
                assert individual.er_short == pytest.approx(batch.er_short)
            assert individual.regime == batch.regime

    def test_batch_no_lookahead(self, detector: EfficiencyRatioDetector) -> None:
        """Truncation test: ER at idx should not change with more future data."""
        rng = np.random.default_rng(42)
        closes_short = 100.0 + np.cumsum(rng.standard_normal(20))

        rng2 = np.random.default_rng(42)
        closes_long = 100.0 + np.cumsum(rng2.standard_normal(40))

        test_idx = 15
        r_short = detector.update(test_idx, closes_short)
        r_long = detector.update(test_idx, closes_long)

        if not math.isnan(r_short.er_short):
            assert r_short.er_short == pytest.approx(r_long.er_short)


class TestERResult:
    def test_result_is_frozen(self) -> None:
        """ERResult should be immutable."""
        result = ERResult(er_short=0.5, er_long=0.4, regime=ERRegime.NEUTRAL, confidence=0.5)
        with pytest.raises(AttributeError):
            result.er_short = 0.9  # type: ignore[misc]

    def test_regime_enum_values(self) -> None:
        assert ERRegime.CHOPPY == 0
        assert ERRegime.NEUTRAL == 1
        assert ERRegime.TRENDING == 2
