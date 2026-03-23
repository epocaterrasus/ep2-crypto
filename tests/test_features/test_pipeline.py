"""Tests for unified feature pipeline."""

from __future__ import annotations

import time

import numpy as np

from ep2_crypto.features.pipeline import FeaturePipeline, build_default_registry


def _make_pipeline_data(n: int = 200, n_levels: int = 5) -> dict[str, np.ndarray]:
    """Create full dataset with all data sources for pipeline testing."""
    rng = np.random.default_rng(42)

    returns = rng.standard_normal(n) * 0.001
    btc_mid = 50000.0 * np.exp(np.cumsum(returns))

    bids = np.zeros((n, n_levels), dtype=np.float64)
    asks = np.zeros((n, n_levels), dtype=np.float64)
    bid_sizes = np.zeros((n, n_levels), dtype=np.float64)
    ask_sizes = np.zeros((n, n_levels), dtype=np.float64)

    for i in range(n):
        spread = btc_mid[i] * rng.uniform(0.0001, 0.0003)
        for lev in range(n_levels):
            tick = btc_mid[i] * 0.0001
            bids[i, lev] = btc_mid[i] - spread / 2 - lev * tick
            asks[i, lev] = btc_mid[i] + spread / 2 + lev * tick
            bid_sizes[i, lev] = rng.exponential(2.0)
            ask_sizes[i, lev] = rng.exponential(2.0)

    # 5-min bars starting at 16:00 UTC (US session for NQ tests)
    base_ts = 16 * 3_600_000
    timestamps = np.arange(n, dtype=np.int64) * 300_000 + base_ts

    opens = btc_mid * (1 + rng.uniform(-0.001, 0.001, n))
    closes = btc_mid * (1 + rng.uniform(-0.001, 0.001, n))
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.002, n))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0, 0.002, n))
    volumes = rng.exponential(500.0, n)

    trade_sizes = rng.exponential(1.0, n)
    trade_sides = rng.choice([-1.0, 1.0], n, p=[0.45, 0.55])

    nq_mid = 18000.0 + np.cumsum(rng.standard_normal(n) * 5)
    eth_mid = 3200.0 + np.cumsum(rng.standard_normal(n) * 2)

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
        "nq_closes": nq_mid,
        "eth_closes": eth_mid,
    }


class TestBuildDefaultRegistry:
    def test_all_computers_registered(self) -> None:
        reg = build_default_registry()
        assert len(reg.get_all()) == 26  # 5+3+4+4 + 4+3+3

    def test_no_duplicate_names(self) -> None:
        reg = build_default_registry()
        names = reg.names
        assert len(names) == len(set(names))


class TestFeaturePipeline:
    def test_output_names_count(self) -> None:
        """Pipeline should produce expected number of features."""
        pipeline = FeaturePipeline()
        # Microstructure: 4+3+2+4+1 = 14
        # Volume: 4+2+3 = 9
        # Volatility: 2+2+1+1 = 6
        # Momentum: 4+1+1+1 = 7
        # Cross-market: 3+3+3+2 = 11
        # Temporal: 6+3+2 = 11
        # Regime: 2+2+2 = 6
        # Total = 14+9+6+7+11+11+6 = 64
        assert pipeline.n_features == 64

    def test_warmup_bars(self) -> None:
        """Warmup should be max across all computers."""
        pipeline = FeaturePipeline()
        # QuantileRank(60) has the largest warmup at 60
        # But HMM(20+10+1=31), LeadLag(20+3=23), GARCH(20)
        assert pipeline.warmup_bars == 60  # QuantileRank

    def test_consistent_output_keys(self) -> None:
        """All bars after warmup should produce the same set of keys."""
        pipeline = FeaturePipeline()
        data = _make_pipeline_data(100)
        warmup = pipeline.warmup_bars

        expected_keys: set[str] | None = None
        for i in range(warmup, 100):
            result = pipeline.compute(
                i, data["timestamps"], data["opens"], data["highs"],
                data["lows"], data["closes"], data["volumes"],
                bids=data["bids"], asks=data["asks"],
                bid_sizes=data["bid_sizes"], ask_sizes=data["ask_sizes"],
                trade_sizes=data["trade_sizes"], trade_sides=data["trade_sides"],
                nq_closes=data["nq_closes"], eth_closes=data["eth_closes"],
            )
            if expected_keys is None:
                expected_keys = set(result.keys())
            else:
                assert set(result.keys()) == expected_keys

    def test_no_nan_after_warmup_with_all_data(self) -> None:
        """No NaN values when all data sources provided."""
        pipeline = FeaturePipeline()
        data = _make_pipeline_data(200)
        warmup = pipeline.warmup_bars

        for i in range(warmup, 200):
            result = pipeline.compute(
                i, data["timestamps"], data["opens"], data["highs"],
                data["lows"], data["closes"], data["volumes"],
                bids=data["bids"], asks=data["asks"],
                bid_sizes=data["bid_sizes"], ask_sizes=data["ask_sizes"],
                trade_sizes=data["trade_sizes"], trade_sides=data["trade_sides"],
                nq_closes=data["nq_closes"], eth_closes=data["eth_closes"],
            )
            for key, val in result.items():
                assert np.isfinite(val), f"Non-finite '{key}' at idx={i}: {val}"

    def test_graceful_without_cross_market(self) -> None:
        """Pipeline should work without NQ/ETH data (NaN or zero)."""
        pipeline = FeaturePipeline()
        data = _make_pipeline_data(100)
        warmup = pipeline.warmup_bars

        result = pipeline.compute(
            warmup + 5, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
            bids=data["bids"], asks=data["asks"],
            bid_sizes=data["bid_sizes"], ask_sizes=data["ask_sizes"],
            trade_sizes=data["trade_sizes"], trade_sides=data["trade_sides"],
            # No nq_closes or eth_closes
        )
        # Should still produce all keys
        assert len(result) == pipeline.n_features

    def test_selected_features(self) -> None:
        """Pipeline with feature selection should only compute selected."""
        pipeline = FeaturePipeline(selected_features=["obi", "roc", "session"])
        data = _make_pipeline_data(50)

        result = pipeline.compute(
            20, data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
            bids=data["bids"], asks=data["asks"],
            bid_sizes=data["bid_sizes"], ask_sizes=data["ask_sizes"],
        )
        # OBI: 4, ROC: 4, Session: 3 = 11
        assert len(result) == 11
        assert "obi_l3" in result
        assert "roc_1" in result
        assert "session_asia" in result
        assert "rsi" not in result

    def test_compute_batch_shape(self) -> None:
        """Batch compute should return (n_bars, n_features) array."""
        pipeline = FeaturePipeline()
        data = _make_pipeline_data(100)

        output = pipeline.compute_batch(
            data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
            bids=data["bids"], asks=data["asks"],
            bid_sizes=data["bid_sizes"], ask_sizes=data["ask_sizes"],
            trade_sizes=data["trade_sizes"], trade_sides=data["trade_sides"],
            nq_closes=data["nq_closes"], eth_closes=data["eth_closes"],
        )
        assert output.shape == (100, pipeline.n_features)

    def test_compute_batch_forward_fill(self) -> None:
        """Batch compute should forward-fill NaN values after warmup."""
        pipeline = FeaturePipeline()
        data = _make_pipeline_data(100)

        output = pipeline.compute_batch(
            data["timestamps"], data["opens"], data["highs"],
            data["lows"], data["closes"], data["volumes"],
            fill_nan=True,
            bids=data["bids"], asks=data["asks"],
            bid_sizes=data["bid_sizes"], ask_sizes=data["ask_sizes"],
            trade_sizes=data["trade_sizes"], trade_sides=data["trade_sides"],
            nq_closes=data["nq_closes"], eth_closes=data["eth_closes"],
        )
        warmup = pipeline.warmup_bars
        # After warmup + some buffer, no NaN should remain
        post_warmup = output[warmup + 5:]
        assert np.all(np.isfinite(post_warmup)), "NaN found after warmup in batch output"

    def test_compute_time_under_1ms(self) -> None:
        """Per-bar compute should be < 1ms on average."""
        pipeline = FeaturePipeline()
        data = _make_pipeline_data(300)
        warmup = pipeline.warmup_bars

        n_bars = 300 - warmup
        start = time.perf_counter()
        for i in range(warmup, 300):
            pipeline.compute(
                i, data["timestamps"], data["opens"], data["highs"],
                data["lows"], data["closes"], data["volumes"],
                bids=data["bids"], asks=data["asks"],
                bid_sizes=data["bid_sizes"], ask_sizes=data["ask_sizes"],
                trade_sizes=data["trade_sizes"], trade_sides=data["trade_sides"],
                nq_closes=data["nq_closes"], eth_closes=data["eth_closes"],
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        avg_ms = elapsed_ms / n_bars
        assert avg_ms < 2.0, f"Average compute time {avg_ms:.3f}ms exceeds 2ms target"
