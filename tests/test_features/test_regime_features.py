"""Tests for regime feature computers: ER, GARCH vol, HMM proxy."""

from __future__ import annotations

import numpy as np

from ep2_crypto.features.regime_features import (
    ERFeatureComputer,
    GARCHFeatureComputer,
    HMMFeatureComputer,
)


def _make_regime_data(n: int = 100, volatility: float = 0.001) -> dict[str, np.ndarray]:
    """Create synthetic data with controllable volatility."""
    rng = np.random.default_rng(42)
    returns = rng.standard_normal(n) * volatility
    prices = 50000.0 * np.exp(np.cumsum(returns))

    return {
        "timestamps": np.arange(n, dtype=np.int64) * 300_000,
        "opens": prices * (1 - rng.uniform(0, 0.0005, n)),
        "highs": prices * (1 + rng.uniform(0, 0.001, n)),
        "lows": prices * (1 - rng.uniform(0, 0.001, n)),
        "closes": prices,
        "volumes": rng.uniform(100, 1000, n),
    }


def _make_trending_data(n: int = 100) -> dict[str, np.ndarray]:
    """Create strongly trending data (ER should be high)."""
    prices = 50000.0 + np.arange(n) * 10.0  # Pure uptrend
    return {
        "timestamps": np.arange(n, dtype=np.int64) * 300_000,
        "opens": prices - 1.0,
        "highs": prices + 2.0,
        "lows": prices - 2.0,
        "closes": prices,
        "volumes": np.ones(n) * 500.0,
    }


def _make_choppy_data(n: int = 100) -> dict[str, np.ndarray]:
    """Create choppy data (ER should be low)."""
    # Alternating up and down moves
    base = 50000.0
    prices = np.zeros(n)
    for i in range(n):
        prices[i] = base + (20.0 if i % 2 == 0 else -20.0)
    return {
        "timestamps": np.arange(n, dtype=np.int64) * 300_000,
        "opens": prices - 1.0,
        "highs": prices + 2.0,
        "lows": prices - 2.0,
        "closes": prices,
        "volumes": np.ones(n) * 500.0,
    }


# ---- ER Feature Tests ----


class TestERFeature:
    def test_output_names(self) -> None:
        comp = ERFeatureComputer()
        assert comp.output_names() == ["er_10", "er_20"]

    def test_warmup(self) -> None:
        comp = ERFeatureComputer(window=10)
        assert comp.warmup_bars == 11
        assert comp.name == "er_feature"

    def test_nan_before_warmup(self) -> None:
        comp = ERFeatureComputer(window=10)
        data = _make_regime_data()
        result = comp.compute(
            5, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
        )
        assert np.isnan(result["er_10"])

    def test_trending_market_high_er(self) -> None:
        """Strong trend -> ER near 1.0."""
        comp = ERFeatureComputer(window=10)
        data = _make_trending_data()
        result = comp.compute(
            50, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
        )
        assert result["er_10"] > 0.9, f"ER on trend = {result['er_10']}"

    def test_choppy_market_low_er(self) -> None:
        """Choppy market -> ER near 0.0."""
        comp = ERFeatureComputer(window=10)
        data = _make_choppy_data()
        result = comp.compute(
            50, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
        )
        assert result["er_10"] < 0.2, f"ER on chop = {result['er_10']}"

    def test_bounded_zero_one(self) -> None:
        """ER should always be in [0, 1]."""
        comp = ERFeatureComputer(window=10)
        data = _make_regime_data()
        for i in range(comp.warmup_bars - 1, len(data["closes"])):
            result = comp.compute(
                i, data["timestamps"], data["opens"], data["highs"],
                data["lows"], data["closes"], data["volumes"],
            )
            if np.isfinite(result["er_10"]):
                assert 0.0 <= result["er_10"] <= 1.0

    def test_er_20_has_more_smoothing(self) -> None:
        """ER_20 uses longer window; both should be valid after warmup."""
        comp = ERFeatureComputer(window=10)
        data = _make_regime_data()
        result = comp.compute(
            50, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
        )
        assert np.isfinite(result["er_10"])
        assert np.isfinite(result["er_20"])


# ---- GARCH Feature Tests ----


class TestGARCHFeature:
    def test_output_names(self) -> None:
        comp = GARCHFeatureComputer()
        assert comp.output_names() == ["garch_vol", "garch_vol_ratio"]

    def test_warmup(self) -> None:
        comp = GARCHFeatureComputer()
        assert comp.warmup_bars == 20
        assert comp.name == "garch_feature"

    def test_nan_before_warmup(self) -> None:
        comp = GARCHFeatureComputer()
        data = _make_regime_data()
        result = comp.compute(
            10, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
        )
        assert np.isnan(result["garch_vol"])

    def test_positive_vol(self) -> None:
        """GARCH vol should always be positive."""
        comp = GARCHFeatureComputer()
        data = _make_regime_data()
        result = comp.compute(
            50, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
        )
        assert result["garch_vol"] > 0

    def test_high_vol_data_higher_garch(self) -> None:
        """Higher volatility data should produce higher GARCH vol."""
        comp = GARCHFeatureComputer()
        low_vol = _make_regime_data(volatility=0.0005)
        high_vol = _make_regime_data(volatility=0.005)

        r_low = comp.compute(
            50, low_vol["timestamps"], low_vol["opens"], low_vol["highs"],
            low_vol["lows"], low_vol["closes"], low_vol["volumes"],
        )
        r_high = comp.compute(
            50, high_vol["timestamps"], high_vol["opens"], high_vol["highs"],
            high_vol["lows"], high_vol["closes"], high_vol["volumes"],
        )
        assert r_high["garch_vol"] > r_low["garch_vol"]

    def test_vol_ratio_reasonable(self) -> None:
        """Vol ratio should be positive and not extreme."""
        comp = GARCHFeatureComputer()
        data = _make_regime_data()
        result = comp.compute(
            50, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
        )
        assert result["garch_vol_ratio"] > 0
        assert result["garch_vol_ratio"] < 100  # Shouldn't be extreme

    def test_finite_values(self) -> None:
        """All outputs should be finite after warmup."""
        comp = GARCHFeatureComputer()
        data = _make_regime_data()
        for i in range(comp.warmup_bars - 1, len(data["closes"])):
            result = comp.compute(
                i, data["timestamps"], data["opens"], data["highs"],
                data["lows"], data["closes"], data["volumes"],
            )
            assert np.isfinite(result["garch_vol"]), f"garch_vol not finite at {i}"
            assert np.isfinite(result["garch_vol_ratio"]), f"garch_vol_ratio not finite at {i}"


# ---- HMM Feature Tests ----


class TestHMMFeature:
    def test_output_names(self) -> None:
        comp = HMMFeatureComputer()
        assert comp.output_names() == ["hmm_high_vol_prob", "hmm_regime_change"]

    def test_warmup(self) -> None:
        comp = HMMFeatureComputer(vol_window=20, smooth_window=10)
        assert comp.warmup_bars == 31
        assert comp.name == "hmm_feature"

    def test_nan_before_warmup(self) -> None:
        comp = HMMFeatureComputer()
        data = _make_regime_data()
        result = comp.compute(
            15, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
        )
        assert np.isnan(result["hmm_high_vol_prob"])

    def test_prob_bounded(self) -> None:
        """Probability should be in [0, 1]."""
        comp = HMMFeatureComputer(vol_window=20, smooth_window=10)
        data = _make_regime_data()
        for i in range(comp.warmup_bars - 1, len(data["closes"])):
            result = comp.compute(
                i, data["timestamps"], data["opens"], data["highs"],
                data["lows"], data["closes"], data["volumes"],
            )
            assert 0.0 <= result["hmm_high_vol_prob"] <= 1.0, (
                f"Prob={result['hmm_high_vol_prob']} at idx={i}"
            )

    def test_regime_change_non_negative(self) -> None:
        """Regime change should be non-negative (absolute difference)."""
        comp = HMMFeatureComputer(vol_window=20, smooth_window=10)
        data = _make_regime_data()
        for i in range(comp.warmup_bars, len(data["closes"])):
            result = comp.compute(
                i, data["timestamps"], data["opens"], data["highs"],
                data["lows"], data["closes"], data["volumes"],
            )
            assert result["hmm_regime_change"] >= 0.0

    def test_constant_data_centered_prob(self) -> None:
        """With constant volatility, prob should be near 0.5."""
        n = 100
        # Very uniform returns
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(n) * 0.001
        prices = 50000.0 * np.exp(np.cumsum(returns))
        data = {
            "timestamps": np.arange(n, dtype=np.int64) * 300_000,
            "opens": prices,
            "highs": prices * 1.001,
            "lows": prices * 0.999,
            "closes": prices,
            "volumes": np.ones(n) * 500.0,
        }
        comp = HMMFeatureComputer(vol_window=20, smooth_window=10)
        result = comp.compute(
            50, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
        )
        # Should be roughly centered around 0.5 for uniform vol
        assert 0.2 <= result["hmm_high_vol_prob"] <= 0.8

    def test_finite_after_warmup(self) -> None:
        """All outputs should be finite after warmup."""
        comp = HMMFeatureComputer(vol_window=20, smooth_window=10)
        data = _make_regime_data()
        for i in range(comp.warmup_bars, len(data["closes"])):
            result = comp.compute(
                i, data["timestamps"], data["opens"], data["highs"],
                data["lows"], data["closes"], data["volumes"],
            )
            assert np.isfinite(result["hmm_high_vol_prob"])
            assert np.isfinite(result["hmm_regime_change"])
