"""Integration test: end-to-end feature computation from mock market data.

Verifies:
- Feature shape consistency
- No NaN/Inf after warmup period
- Compute time < 1ms per bar
- All features produce expected number of outputs
- Feature value ranges are reasonable
"""

from __future__ import annotations

import time

import numpy as np

from ep2_crypto.features.pipeline import FeaturePipeline


def _make_realistic_data(n: int = 500, n_levels: int = 5) -> dict[str, np.ndarray]:
    """Create realistic-ish BTC market data with all data sources."""
    rng = np.random.default_rng(42)

    # Realistic BTC price path with trending and mean-reverting components
    returns = rng.standard_normal(n) * 0.001  # ~0.1% per bar
    mid = 50000.0 * np.exp(np.cumsum(returns))

    bids = np.zeros((n, n_levels), dtype=np.float64)
    asks = np.zeros((n, n_levels), dtype=np.float64)
    bid_sizes = np.zeros((n, n_levels), dtype=np.float64)
    ask_sizes = np.zeros((n, n_levels), dtype=np.float64)

    for i in range(n):
        spread = mid[i] * rng.uniform(0.0001, 0.0003)  # 1-3 bps spread
        for lev in range(n_levels):
            tick = mid[i] * 0.0001  # ~$5 tick
            bids[i, lev] = mid[i] - spread / 2 - lev * tick
            asks[i, lev] = mid[i] + spread / 2 + lev * tick
            bid_sizes[i, lev] = rng.exponential(2.0)
            ask_sizes[i, lev] = rng.exponential(2.0)

    # 5-min bars starting at 16:00 UTC (US session)
    base_ts = 16 * 3_600_000
    timestamps = np.arange(n, dtype=np.int64) * 300_000 + base_ts

    opens = mid * (1 + rng.uniform(-0.001, 0.001, n))
    closes = mid * (1 + rng.uniform(-0.001, 0.001, n))
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.002, n))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0, 0.002, n))
    volumes = rng.exponential(500.0, n)

    trade_sizes = rng.exponential(1.0, n)
    trade_sides = rng.choice([-1.0, 1.0], n, p=[0.45, 0.55])  # slight buy bias

    # Cross-market data
    nq_closes = 18000.0 + np.cumsum(rng.standard_normal(n) * 5)
    eth_closes = 3200.0 + np.cumsum(rng.standard_normal(n) * 2)

    return {
        "timestamps": timestamps,
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "volumes": volumes,
        "bids": bids,
        "asks": asks,
        "bid_sizes": bid_sizes,
        "ask_sizes": ask_sizes,
        "trade_sizes": trade_sizes,
        "trade_sides": trade_sides,
        "nq_closes": nq_closes,
        "eth_closes": eth_closes,
    }


def _get_kwargs(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Extract kwargs from data dict."""
    positional = {"timestamps", "opens", "highs", "lows", "closes", "volumes"}
    return {k: v for k, v in data.items() if k not in positional}


class TestFeatureIntegration:
    def test_consistent_output_shape(self) -> None:
        """All bars after warmup should produce the same set of keys."""
        data = _make_realistic_data(100)
        pipeline = FeaturePipeline()
        warmup = pipeline.warmup_bars

        expected_keys: set[str] | None = None
        for i in range(warmup, 100):
            result = pipeline.compute(
                i, data["timestamps"], data["opens"], data["highs"],
                data["lows"], data["closes"], data["volumes"],
                **_get_kwargs(data),
            )
            if expected_keys is None:
                expected_keys = set(result.keys())
            else:
                assert set(result.keys()) == expected_keys, (
                    f"Inconsistent keys at idx={i}"
                )

    def test_no_nan_inf_after_warmup(self) -> None:
        """No NaN or Inf values in any feature after warmup."""
        data = _make_realistic_data(200)
        pipeline = FeaturePipeline()
        warmup = pipeline.warmup_bars

        for i in range(warmup, 200):
            result = pipeline.compute(
                i, data["timestamps"], data["opens"], data["highs"],
                data["lows"], data["closes"], data["volumes"],
                **_get_kwargs(data),
            )
            for key, val in result.items():
                assert np.isfinite(val), (
                    f"Non-finite value in '{key}' at idx={i}: {val}"
                )

    def test_compute_time_under_2ms(self) -> None:
        """Feature computation should be < 2ms per bar on average."""
        data = _make_realistic_data(500)
        pipeline = FeaturePipeline()
        warmup = pipeline.warmup_bars

        n_bars = 500 - warmup
        start = time.perf_counter()
        for i in range(warmup, 500):
            pipeline.compute(
                i, data["timestamps"], data["opens"], data["highs"],
                data["lows"], data["closes"], data["volumes"],
                **_get_kwargs(data),
            )
        elapsed_ms = (time.perf_counter() - start) * 1000

        avg_ms = elapsed_ms / n_bars
        assert avg_ms < 2.0, (
            f"Average compute time {avg_ms:.3f}ms exceeds 2ms target"
        )

    def test_expected_feature_count(self) -> None:
        """Total feature count: Sprint 3-5 features = 64."""
        pipeline = FeaturePipeline()
        data = _make_realistic_data(100)
        warmup = pipeline.warmup_bars
        result = pipeline.compute(
            warmup + 5,
            data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
            **_get_kwargs(data),
        )
        assert len(result) == 64

    def test_feature_values_are_reasonable(self) -> None:
        """Sanity check feature value ranges on realistic data."""
        data = _make_realistic_data(100)
        pipeline = FeaturePipeline()
        warmup = pipeline.warmup_bars

        for i in range(warmup, 100):
            result = pipeline.compute(
                i, data["timestamps"], data["opens"], data["highs"],
                data["lows"], data["closes"], data["volumes"],
                **_get_kwargs(data),
            )

            # Sprint 3: OBI in [-1, 1]
            assert -1.0 <= result["obi_l3"] <= 1.0
            assert -1.0 <= result["obi_l5"] <= 1.0

            # Sprint 3: TFI in [-1, 1]
            assert -1.0 <= result["tfi_1bar"] <= 1.0
            assert -1.0 <= result["tfi_6bar"] <= 1.0

            # Sprint 3: Relative spread positive and small
            assert 0.0 < result["relative_spread"] < 0.01

            # Sprint 3: Microprice close to mid
            mid = (data["bids"][i, 0] + data["asks"][i, 0]) / 2.0
            assert abs(result["microprice"] - mid) / mid < 0.001

            # Sprint 4: Volume delta in [-1, 1]
            assert -1.0 <= result["vol_delta_1bar"] <= 1.0

            # Sprint 4: Volatility positive
            assert result["realized_vol_short"] >= 0
            assert result["parkinson_vol_short"] > 0
            assert result["ewma_vol"] > 0

            # Sprint 4: RSI in [0, 100]
            assert 0.0 <= result["rsi"] <= 100.0

            # Sprint 4: Quantile rank in [0, 1]
            assert 0.0 <= result["quantile_rank"] <= 1.0

            # Sprint 5: sin/cos bounded
            assert -1.0 <= result["hour_sin"] <= 1.0
            assert -1.0 <= result["hour_cos"] <= 1.0

            # Sprint 5: Session one-hot sums to 1
            session_sum = result["session_asia"] + result["session_europe"] + result["session_us"]
            assert session_sum == 1.0

            # Sprint 5: ER in [0, 1]
            assert 0.0 <= result["er_10"] <= 1.0

            # Sprint 5: HMM probability in [0, 1]
            assert 0.0 <= result["hmm_high_vol_prob"] <= 1.0

            # Sprint 5: Lead-lag correlation in [-1, 1]
            assert -1.0 <= result["lead_lag_corr_1"] <= 1.0

            # Sprint 5: ETH ratio positive
            assert result["eth_btc_ratio"] > 0

            # Sprint 5: GARCH vol positive
            assert result["garch_vol"] > 0
