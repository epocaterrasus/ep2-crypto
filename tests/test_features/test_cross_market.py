"""Tests for cross-market feature computers: NQ returns, ETH ratio, lead-lag, divergence."""

from __future__ import annotations

import numpy as np
import pytest

from ep2_crypto.features.cross_market import (
    DivergenceComputer,
    ETHRatioComputer,
    LeadLagComputer,
    NQReturnComputer,
)


def _make_cross_market_data(n: int = 50) -> dict[str, np.ndarray]:
    """Create synthetic BTC + NQ + ETH data for testing."""
    rng = np.random.default_rng(42)

    btc_mid = 50000.0 + np.cumsum(rng.standard_normal(n) * 10)
    nq_mid = 18000.0 + np.cumsum(rng.standard_normal(n) * 5)
    eth_mid = 3200.0 + np.cumsum(rng.standard_normal(n) * 2)

    opens = btc_mid - rng.uniform(0, 5, n)
    closes = btc_mid + rng.uniform(0, 5, n)
    highs = np.maximum(opens, closes) + rng.uniform(0, 3, n)
    lows = np.minimum(opens, closes) - rng.uniform(0, 3, n)
    volumes = rng.uniform(100, 1000, n)

    # Timestamps: 5-min bars starting at 16:00 UTC (US session)
    base_ts = 16 * 3_600_000  # 16:00 UTC in ms
    timestamps = np.arange(n, dtype=np.int64) * 300_000 + base_ts

    return {
        "timestamps": timestamps,
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "volumes": volumes,
        "nq_closes": nq_mid,
        "eth_closes": eth_mid,
    }


# ---- NQ Return Tests ----


class TestNQReturn:
    def test_output_names(self) -> None:
        comp = NQReturnComputer()
        assert comp.output_names() == ["nq_ret_lag1", "nq_ret_lag2", "nq_ret_lag3"]

    def test_warmup(self) -> None:
        comp = NQReturnComputer()
        assert comp.warmup_bars == 4
        assert comp.name == "nq_return"

    def test_nan_before_warmup(self) -> None:
        comp = NQReturnComputer()
        data = _make_cross_market_data()
        result = comp.compute(
            2, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
            nq_closes=data["nq_closes"],
        )
        assert np.isnan(result["nq_ret_lag1"])

    def test_nan_without_nq_data(self) -> None:
        comp = NQReturnComputer()
        data = _make_cross_market_data()
        result = comp.compute(
            10, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
        )
        assert np.isnan(result["nq_ret_lag1"])

    def test_returns_during_us_session(self) -> None:
        """During US session (16:00-00:00 UTC), returns should be computed."""
        comp = NQReturnComputer()
        data = _make_cross_market_data()
        result = comp.compute(
            10, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
            nq_closes=data["nq_closes"],
        )
        # Should produce finite values
        assert np.isfinite(result["nq_ret_lag1"])
        assert np.isfinite(result["nq_ret_lag2"])
        assert np.isfinite(result["nq_ret_lag3"])

    def test_zero_outside_us_session(self) -> None:
        """Outside US session, returns should be zero."""
        comp = NQReturnComputer()
        data = _make_cross_market_data()
        # Set timestamps to 08:00 UTC (outside US session)
        n = len(data["timestamps"])
        data["timestamps"] = np.arange(n, dtype=np.int64) * 300_000 + 8 * 3_600_000
        result = comp.compute(
            10, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
            nq_closes=data["nq_closes"],
        )
        assert result["nq_ret_lag1"] == 0.0
        assert result["nq_ret_lag2"] == 0.0
        assert result["nq_ret_lag3"] == 0.0

    def test_lag1_golden_value(self) -> None:
        """Verify lag-1 return matches hand calculation."""
        comp = NQReturnComputer()
        n = 10
        nq = np.array([100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 107.0, 108.0, 110.0])
        btc = np.ones(n) * 50000.0
        ts = np.arange(n, dtype=np.int64) * 300_000 + 16 * 3_600_000  # US session

        result = comp.compute(
            5, ts, btc, btc, btc, btc, np.ones(n),
            nq_closes=nq,
        )
        # lag1: return at idx=4 -> (105-103)/103
        expected = (105.0 - 103.0) / 103.0
        assert result["nq_ret_lag1"] == pytest.approx(expected, rel=1e-10)

    def test_lag2_golden_value(self) -> None:
        """Verify lag-2 return matches hand calculation."""
        comp = NQReturnComputer()
        n = 10
        nq = np.array([100.0, 102.0, 101.0, 103.0, 105.0, 104.0, 106.0, 107.0, 108.0, 110.0])
        btc = np.ones(n) * 50000.0
        ts = np.arange(n, dtype=np.int64) * 300_000 + 16 * 3_600_000

        result = comp.compute(
            5, ts, btc, btc, btc, btc, np.ones(n),
            nq_closes=nq,
        )
        # lag2: return at idx=3 -> (103-101)/101
        expected = (103.0 - 101.0) / 101.0
        assert result["nq_ret_lag2"] == pytest.approx(expected, rel=1e-10)


# ---- ETH Ratio Tests ----


class TestETHRatio:
    def test_output_names(self) -> None:
        comp = ETHRatioComputer()
        assert comp.output_names() == ["eth_btc_ratio", "eth_btc_ratio_roc1", "eth_btc_ratio_roc6"]

    def test_warmup(self) -> None:
        comp = ETHRatioComputer()
        assert comp.warmup_bars == 7
        assert comp.name == "eth_ratio"

    def test_nan_before_warmup(self) -> None:
        comp = ETHRatioComputer()
        data = _make_cross_market_data()
        result = comp.compute(
            3, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
            eth_closes=data["eth_closes"],
        )
        assert np.isnan(result["eth_btc_ratio"])

    def test_nan_without_eth_data(self) -> None:
        comp = ETHRatioComputer()
        data = _make_cross_market_data()
        result = comp.compute(
            10, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
        )
        assert np.isnan(result["eth_btc_ratio"])

    def test_ratio_golden_value(self) -> None:
        """Verify ratio matches hand calculation."""
        comp = ETHRatioComputer()
        n = 10
        btc = np.ones(n) * 50000.0
        eth = np.ones(n) * 3200.0
        ts = np.arange(n, dtype=np.int64) * 300_000

        result = comp.compute(
            8, ts, btc, btc, btc, btc, np.ones(n),
            eth_closes=eth,
        )
        assert result["eth_btc_ratio"] == pytest.approx(3200.0 / 50000.0, rel=1e-10)

    def test_ratio_roc_when_constant(self) -> None:
        """When prices are constant, ROC should be zero."""
        comp = ETHRatioComputer()
        n = 10
        btc = np.ones(n) * 50000.0
        eth = np.ones(n) * 3200.0
        ts = np.arange(n, dtype=np.int64) * 300_000

        result = comp.compute(
            8, ts, btc, btc, btc, btc, np.ones(n),
            eth_closes=eth,
        )
        assert result["eth_btc_ratio_roc1"] == pytest.approx(0.0, abs=1e-10)
        assert result["eth_btc_ratio_roc6"] == pytest.approx(0.0, abs=1e-10)

    def test_ratio_roc_with_trend(self) -> None:
        """When ETH strengthens vs BTC, roc should be positive."""
        comp = ETHRatioComputer()
        n = 10
        btc = np.ones(n) * 50000.0
        eth = np.linspace(3000.0, 3500.0, n)  # ETH trending up
        ts = np.arange(n, dtype=np.int64) * 300_000

        result = comp.compute(
            8, ts, btc, btc, btc, btc, np.ones(n),
            eth_closes=eth,
        )
        assert result["eth_btc_ratio_roc1"] > 0
        assert result["eth_btc_ratio_roc6"] > 0

    def test_ratio_positive(self) -> None:
        """Ratio should always be positive when prices are positive."""
        comp = ETHRatioComputer()
        data = _make_cross_market_data()
        result = comp.compute(
            10, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
            eth_closes=data["eth_closes"],
        )
        assert result["eth_btc_ratio"] > 0


# ---- Lead-Lag Tests ----


class TestLeadLag:
    def test_output_names(self) -> None:
        comp = LeadLagComputer(window=20)
        assert comp.output_names() == ["lead_lag_corr_1", "lead_lag_corr_2", "lead_lag_corr_3"]

    def test_warmup(self) -> None:
        comp = LeadLagComputer(window=20)
        assert comp.warmup_bars == 23  # window + max lag
        assert comp.name == "lead_lag"

    def test_nan_before_warmup(self) -> None:
        comp = LeadLagComputer(window=20)
        data = _make_cross_market_data()
        result = comp.compute(
            10, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
            nq_closes=data["nq_closes"],
        )
        assert np.isnan(result["lead_lag_corr_1"])

    def test_nan_without_nq_data(self) -> None:
        comp = LeadLagComputer(window=20)
        data = _make_cross_market_data()
        result = comp.compute(
            30, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
        )
        assert np.isnan(result["lead_lag_corr_1"])

    def test_correlation_bounded(self) -> None:
        """Correlation should be in [-1, 1]."""
        comp = LeadLagComputer(window=20)
        data = _make_cross_market_data()
        result = comp.compute(
            30, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
            nq_closes=data["nq_closes"],
        )
        for lag in [1, 2, 3]:
            val = result[f"lead_lag_corr_{lag}"]
            assert -1.0 <= val <= 1.0, f"Correlation out of bounds: {val}"

    def test_perfect_correlation(self) -> None:
        """When NQ and BTC move identically (shifted), correlation should be near 1."""
        n = 50
        comp = LeadLagComputer(window=15)
        # BTC follows NQ with lag 1
        nq = 18000.0 + np.cumsum(np.ones(n) * 5.0)
        btc = np.zeros(n)
        btc[0] = 50000.0
        for i in range(1, n):
            btc[i] = btc[i - 1] + 5.0  # Same increment, no lag difference
        ts = np.arange(n, dtype=np.int64) * 300_000

        result = comp.compute(
            30, ts, btc, btc, btc, btc, np.ones(n),
            nq_closes=nq,
        )
        # With identical trends, correlations should be very high
        assert result["lead_lag_corr_1"] > 0.5

    def test_small_window(self) -> None:
        """Should work with smaller window."""
        comp = LeadLagComputer(window=10)
        assert comp.warmup_bars == 13
        data = _make_cross_market_data()
        result = comp.compute(
            20, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
            nq_closes=data["nq_closes"],
        )
        assert np.isfinite(result["lead_lag_corr_1"])


# ---- Divergence Tests ----


class TestDivergence:
    def test_output_names(self) -> None:
        comp = DivergenceComputer()
        assert comp.output_names() == ["div_btc_nq", "div_btc_eth"]

    def test_warmup(self) -> None:
        comp = DivergenceComputer(lookback=6)
        assert comp.warmup_bars == 7
        assert comp.name == "divergence"

    def test_nan_before_warmup(self) -> None:
        comp = DivergenceComputer(lookback=6)
        data = _make_cross_market_data()
        result = comp.compute(
            3, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
            nq_closes=data["nq_closes"], eth_closes=data["eth_closes"],
        )
        assert np.isnan(result["div_btc_nq"])

    def test_zero_without_cross_data(self) -> None:
        """Without cross-market data, divergence should be 0 (not NaN)."""
        comp = DivergenceComputer(lookback=6)
        data = _make_cross_market_data()
        result = comp.compute(
            10, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
        )
        assert result["div_btc_nq"] == 0.0
        assert result["div_btc_eth"] == 0.0

    def test_golden_value_no_divergence(self) -> None:
        """When BTC and NQ move identically (% wise), divergence should be ~0."""
        n = 15
        comp = DivergenceComputer(lookback=5)
        # Both assets gain exactly 1%
        btc = np.ones(n) * 50000.0
        btc[10:] = 50500.0  # +1%
        nq = np.ones(n) * 18000.0
        nq[10:] = 18180.0   # +1%
        ts = np.arange(n, dtype=np.int64) * 300_000

        result = comp.compute(
            12, ts, btc, btc, btc, btc, np.ones(n),
            nq_closes=nq,
        )
        assert result["div_btc_nq"] == pytest.approx(0.0, abs=1e-10)

    def test_positive_divergence(self) -> None:
        """When BTC outperforms NQ, divergence should be positive."""
        n = 15
        comp = DivergenceComputer(lookback=5)
        btc = np.ones(n) * 50000.0
        btc[10:] = 51000.0  # +2%
        nq = np.ones(n) * 18000.0
        nq[10:] = 18090.0   # +0.5%
        ts = np.arange(n, dtype=np.int64) * 300_000

        result = comp.compute(
            12, ts, btc, btc, btc, btc, np.ones(n),
            nq_closes=nq,
        )
        assert result["div_btc_nq"] > 0

    def test_negative_divergence(self) -> None:
        """When BTC underperforms ETH, divergence should be negative."""
        n = 15
        comp = DivergenceComputer(lookback=5)
        btc = np.ones(n) * 50000.0
        btc[10:] = 50250.0   # +0.5%
        eth = np.ones(n) * 3200.0
        eth[10:] = 3264.0    # +2%
        ts = np.arange(n, dtype=np.int64) * 300_000

        result = comp.compute(
            12, ts, btc, btc, btc, btc, np.ones(n),
            eth_closes=eth,
        )
        assert result["div_btc_eth"] < 0

    def test_with_both_cross_assets(self) -> None:
        """Should compute both NQ and ETH divergence when both provided."""
        comp = DivergenceComputer(lookback=6)
        data = _make_cross_market_data()
        result = comp.compute(
            10, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
            nq_closes=data["nq_closes"], eth_closes=data["eth_closes"],
        )
        assert np.isfinite(result["div_btc_nq"])
        assert np.isfinite(result["div_btc_eth"])
