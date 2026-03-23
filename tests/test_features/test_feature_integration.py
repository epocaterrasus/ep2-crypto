"""Integration test: end-to-end feature computation from mock market data.

Verifies:
- Feature shape consistency
- No NaN/Inf after warmup period
- Compute time < 1ms per bar
- All features produce expected number of outputs
"""

from __future__ import annotations

import time

import numpy as np

from ep2_crypto.features.base import FeatureRegistry
from ep2_crypto.features.microstructure import (
    KyleLambdaComputer,
    MicropriceComputer,
    OBIComputer,
    OFIComputer,
    TFIComputer,
)


def _make_realistic_data(n: int = 500, n_levels: int = 5) -> dict[str, np.ndarray]:
    """Create realistic-ish BTC market data."""
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

    timestamps = np.arange(n, dtype=np.int64) * 300_000  # 5-min bars
    opens = mid * (1 + rng.uniform(-0.001, 0.001, n))
    closes = mid * (1 + rng.uniform(-0.001, 0.001, n))
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.002, n))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0, 0.002, n))
    volumes = rng.exponential(500.0, n)

    trade_sizes = rng.exponential(1.0, n)
    trade_sides = rng.choice([-1.0, 1.0], n, p=[0.45, 0.55])  # slight buy bias

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
    }


def _build_full_registry() -> FeatureRegistry:
    reg = FeatureRegistry()
    reg.register(OBIComputer())
    reg.register(OFIComputer())
    reg.register(MicropriceComputer())
    reg.register(TFIComputer())
    reg.register(KyleLambdaComputer(window=20))
    return reg


class TestFeatureIntegration:
    def test_consistent_output_shape(self) -> None:
        """All bars after warmup should produce the same set of keys."""
        data = _make_realistic_data(100)
        reg = _build_full_registry()
        warmup = reg.max_warmup()

        expected_keys: set[str] | None = None
        for i in range(warmup, 100):
            result = reg.compute_all(
                i,
                data["timestamps"], data["opens"], data["highs"],
                data["lows"], data["closes"], data["volumes"],
                bids=data["bids"], asks=data["asks"],
                bid_sizes=data["bid_sizes"], ask_sizes=data["ask_sizes"],
                trade_sizes=data["trade_sizes"], trade_sides=data["trade_sides"],
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
        reg = _build_full_registry()
        warmup = reg.max_warmup()

        for i in range(warmup, 200):
            result = reg.compute_all(
                i,
                data["timestamps"], data["opens"], data["highs"],
                data["lows"], data["closes"], data["volumes"],
                bids=data["bids"], asks=data["asks"],
                bid_sizes=data["bid_sizes"], ask_sizes=data["ask_sizes"],
                trade_sizes=data["trade_sizes"], trade_sides=data["trade_sides"],
            )
            for key, val in result.items():
                assert np.isfinite(val), (
                    f"Non-finite value in '{key}' at idx={i}: {val}"
                )

    def test_compute_time_under_1ms(self) -> None:
        """Feature computation should be < 1ms per bar on average."""
        data = _make_realistic_data(500)
        reg = _build_full_registry()
        warmup = reg.max_warmup()

        n_bars = 500 - warmup
        start = time.perf_counter()
        for i in range(warmup, 500):
            reg.compute_all(
                i,
                data["timestamps"], data["opens"], data["highs"],
                data["lows"], data["closes"], data["volumes"],
                bids=data["bids"], asks=data["asks"],
                bid_sizes=data["bid_sizes"], ask_sizes=data["ask_sizes"],
                trade_sizes=data["trade_sizes"], trade_sides=data["trade_sides"],
            )
        elapsed_ms = (time.perf_counter() - start) * 1000

        avg_ms = elapsed_ms / n_bars
        assert avg_ms < 1.0, (
            f"Average compute time {avg_ms:.3f}ms exceeds 1ms target"
        )

    def test_expected_feature_count(self) -> None:
        """Total microstructure feature count should be 14."""
        reg = _build_full_registry()
        data = _make_realistic_data(30)
        result = reg.compute_all(
            25,
            data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
            bids=data["bids"], asks=data["asks"],
            bid_sizes=data["bid_sizes"], ask_sizes=data["ask_sizes"],
            trade_sizes=data["trade_sizes"], trade_sides=data["trade_sides"],
        )
        assert len(result) == 14

    def test_feature_values_are_reasonable(self) -> None:
        """Sanity check feature value ranges on realistic data."""
        data = _make_realistic_data(100)
        reg = _build_full_registry()
        warmup = reg.max_warmup()

        for i in range(warmup, 100):
            result = reg.compute_all(
                i,
                data["timestamps"], data["opens"], data["highs"],
                data["lows"], data["closes"], data["volumes"],
                bids=data["bids"], asks=data["asks"],
                bid_sizes=data["bid_sizes"], ask_sizes=data["ask_sizes"],
                trade_sizes=data["trade_sizes"], trade_sides=data["trade_sides"],
            )

            # OBI in [-1, 1]
            assert -1.0 <= result["obi_l3"] <= 1.0
            assert -1.0 <= result["obi_l5"] <= 1.0

            # TFI in [-1, 1]
            assert -1.0 <= result["tfi_1bar"] <= 1.0
            assert -1.0 <= result["tfi_6bar"] <= 1.0

            # Relative spread should be positive and small for BTC
            assert 0.0 < result["relative_spread"] < 0.01  # < 100 bps

            # Microprice should be close to mid
            mid = (data["bids"][i, 0] + data["asks"][i, 0]) / 2.0
            assert abs(result["microprice"] - mid) / mid < 0.001  # < 10 bps from mid
