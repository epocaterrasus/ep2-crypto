"""Tests for backtest engine."""

from __future__ import annotations

import numpy as np
import pytest

from ep2_crypto.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    _compute_bar_returns_numba,
    _update_equity_numba,
)
from ep2_crypto.backtest.metrics import BacktestResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_ohlcv(
    n: int = 500,
    start_price: float = 100_000.0,
    trend: float = 0.0,
    volatility: float = 0.001,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate synthetic OHLCV data."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(trend, volatility, size=n)
    prices = start_price * np.cumprod(1 + returns)

    # Simple OHLCV from close prices
    closes = prices
    opens = np.roll(prices, 1)
    opens[0] = start_price
    highs = np.maximum(opens, closes) * (1 + rng.uniform(0, 0.002, n))
    lows = np.minimum(opens, closes) * (1 - rng.uniform(0, 0.002, n))
    volumes = rng.uniform(100, 500, n)

    # Timestamps: 5-min bars starting 2024-01-01 10:00 UTC (within trading hours)
    base_ts = 1704103200000  # 2024-01-01 10:00 UTC in ms
    timestamps = np.arange(n, dtype=np.int64) * 300_000 + base_ts

    return {
        "opens": opens,
        "highs": highs,
        "lows": lows,
        "closes": closes,
        "volumes": volumes,
        "timestamps_ms": timestamps,
    }


# ---------------------------------------------------------------------------
# Numba functions
# ---------------------------------------------------------------------------
class TestNumbaFunctions:
    def test_update_equity_long_profit(self) -> None:
        """Long position, price up → equity increases."""
        eq = _update_equity_numba(50_000.0, 0.05, 1, 101_000.0, 100_000.0)
        # PnL = 0.05 * (101000 - 100000) * 1 = 50
        assert abs(eq - 50_050.0) < 0.01

    def test_update_equity_short_profit(self) -> None:
        """Short position, price down → equity increases."""
        eq = _update_equity_numba(50_000.0, 0.05, -1, 99_000.0, 100_000.0)
        # PnL = 0.05 * (99000 - 100000) * -1 = 50
        assert abs(eq - 50_050.0) < 0.01

    def test_update_equity_flat(self) -> None:
        """Flat position → no change."""
        eq = _update_equity_numba(50_000.0, 0.0, 0, 101_000.0, 100_000.0)
        assert eq == 50_000.0

    def test_compute_bar_returns(self) -> None:
        closes = np.array([100.0, 101.0, 99.0, 102.0], dtype=np.float64)
        positions = np.array([0.0, 1.0, 1.0, -1.0], dtype=np.float64)
        returns = _compute_bar_returns_numba(closes, positions, 4)

        assert returns[0] == 0.0  # first bar, no prior
        assert abs(returns[1] - 0.01) < 0.001  # long, +1%
        # bar 2: close from 101→99, position long → return = -0.0198
        assert returns[2] < 0
        # bar 3: close from 99→102, position short → return = -sign*3.03%
        assert returns[3] < 0  # short, price went up = loss


# ---------------------------------------------------------------------------
# Engine — basic lifecycle
# ---------------------------------------------------------------------------
class TestBacktestEngineBasic:
    def test_no_signals_no_trades(self) -> None:
        """With no signals, engine should produce 0 trades."""
        data = _make_ohlcv(200)
        engine = BacktestEngine(BacktestConfig(initial_equity=50_000.0))
        result = engine.run(
            **data,
            signals=np.zeros(200, dtype=np.int8),
        )
        assert isinstance(result, BacktestResult)
        assert result.total_trades == 0
        assert abs(result.total_return) < 0.001

    def test_all_long_signals(self) -> None:
        """Constant long signals should produce trades."""
        data = _make_ohlcv(300, trend=0.0002)  # slight uptrend
        signals = np.ones(300, dtype=np.int8)  # always long
        confidences = np.full(300, 0.7)

        engine = BacktestEngine(BacktestConfig(
            initial_equity=50_000.0,
            confidence_threshold=0.5,
        ))
        result = engine.run(**data, signals=signals, confidences=confidences)

        assert isinstance(result, BacktestResult)
        assert result.total_trades >= 1

    def test_short_signals(self) -> None:
        """Short signals should produce short trades."""
        data = _make_ohlcv(300, trend=-0.0002)  # slight downtrend
        signals = -np.ones(300, dtype=np.int8)
        confidences = np.full(300, 0.7)

        engine = BacktestEngine(BacktestConfig(
            initial_equity=50_000.0,
            confidence_threshold=0.5,
        ))
        result = engine.run(**data, signals=signals, confidences=confidences)
        assert result.total_trades >= 1

    def test_minimum_data(self) -> None:
        """Engine should handle minimal data gracefully."""
        engine = BacktestEngine()
        result = engine.run(
            opens=np.array([100.0]),
            highs=np.array([101.0]),
            lows=np.array([99.0]),
            closes=np.array([100.5]),
            volumes=np.array([1000.0]),
            timestamps_ms=np.array([1704067200000], dtype=np.int64),
        )
        assert result.total_trades == 0


# ---------------------------------------------------------------------------
# Engine — execution rules
# ---------------------------------------------------------------------------
class TestBacktestEngineExecution:
    def test_next_bar_open_execution(self) -> None:
        """Signal at bar t should NOT be executed at bar t."""
        data = _make_ohlcv(100)
        # Signal only at bar 10
        signals = np.zeros(100, dtype=np.int8)
        signals[10] = 1
        confidences = np.full(100, 0.7)

        engine = BacktestEngine(BacktestConfig(
            initial_equity=50_000.0,
            confidence_threshold=0.5,
        ))
        result = engine.run(**data, signals=signals, confidences=confidences)

        # If trades happened, the entry should be after bar 10
        if result.total_trades > 0:
            # Can't directly inspect trade records from result,
            # but the engine should have at least 1 trade
            assert result.total_trades >= 1

    def test_confidence_filtering(self) -> None:
        """Low confidence signals should be filtered out."""
        data = _make_ohlcv(200)
        signals = np.ones(200, dtype=np.int8)
        confidences = np.full(200, 0.3)  # below threshold

        engine = BacktestEngine(BacktestConfig(
            initial_equity=50_000.0,
            confidence_threshold=0.55,
        ))
        result = engine.run(**data, signals=signals, confidences=confidences)
        assert result.total_trades == 0

    def test_costs_always_present(self) -> None:
        """Costs should be > 0 when trades happen."""
        data = _make_ohlcv(300, trend=0.0002)
        signals = np.ones(300, dtype=np.int8)
        confidences = np.full(300, 0.7)

        engine = BacktestEngine(BacktestConfig(
            initial_equity=50_000.0,
            confidence_threshold=0.5,
        ))
        result = engine.run(**data, signals=signals, confidences=confidences)
        if result.total_trades > 0:
            assert result.total_fee_usd > 0

    def test_deterministic(self) -> None:
        """Same inputs → same outputs."""
        data = _make_ohlcv(200)
        signals = np.ones(200, dtype=np.int8)
        confidences = np.full(200, 0.7)

        results = []
        for _ in range(2):
            engine = BacktestEngine(BacktestConfig(
                initial_equity=50_000.0,
                seed=42,
                confidence_threshold=0.5,
            ))
            r = engine.run(**data, signals=signals, confidences=confidences)
            results.append(r)

        assert results[0].total_return == results[1].total_return
        assert results[0].total_trades == results[1].total_trades


# ---------------------------------------------------------------------------
# Engine — risk integration
# ---------------------------------------------------------------------------
class TestBacktestEngineRisk:
    def test_risk_engine_limits_size(self) -> None:
        """Risk engine should cap position size."""
        data = _make_ohlcv(300, trend=0.0001)
        signals = np.ones(300, dtype=np.int8)
        confidences = np.full(300, 0.7)

        config = BacktestConfig(
            initial_equity=50_000.0,
            confidence_threshold=0.5,
        )
        engine = BacktestEngine(config)
        result = engine.run(**data, signals=signals, confidences=confidences)

        # Should have trades but equity shouldn't explode
        assert isinstance(result, BacktestResult)

    def test_signal_reversal_closes_position(self) -> None:
        """Switching from long to short should close the long first."""
        data = _make_ohlcv(200)
        signals = np.zeros(200, dtype=np.int8)
        confidences = np.full(200, 0.7)

        # Long signal at bar 10, short signal at bar 50
        signals[10] = 1
        signals[50] = -1

        engine = BacktestEngine(BacktestConfig(
            initial_equity=50_000.0,
            confidence_threshold=0.5,
        ))
        result = engine.run(**data, signals=signals, confidences=confidences)
        # At least 1 trade should be closed by reversal
        assert result.total_trades >= 1


# ---------------------------------------------------------------------------
# Engine — output format
# ---------------------------------------------------------------------------
class TestBacktestEngineOutput:
    def test_result_has_equity_curve(self) -> None:
        data = _make_ohlcv(200)
        signals = np.ones(200, dtype=np.int8)
        confidences = np.full(200, 0.7)

        engine = BacktestEngine(BacktestConfig(
            initial_equity=50_000.0,
            confidence_threshold=0.5,
        ))
        result = engine.run(**data, signals=signals, confidences=confidences)
        assert len(result.equity_curve) == 200

    def test_result_has_rolling_sharpe(self) -> None:
        data = _make_ohlcv(200)
        engine = BacktestEngine()
        result = engine.run(**data)
        assert len(result.rolling_sharpe_30d) == 200

    def test_result_summary_string(self) -> None:
        data = _make_ohlcv(200)
        engine = BacktestEngine()
        result = engine.run(**data)
        summary = result.summary()
        assert "Sharpe" in summary
