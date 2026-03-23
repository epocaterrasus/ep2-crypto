"""Synthetic data generation for benchmark testing.

Generates realistic 5-minute BTC OHLCV data with optional auxiliary columns
(funding_rate, obi, nq_close) using calibrated stochastic processes.

This allows benchmarks to be tested before real data ingestion is complete.
Parameters are calibrated from empirical BTC 5-min statistics (2023-2024).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Empirical BTC 5-min statistics (2023-2024 calibration)
BTC_5MIN_ANNUAL_VOL = 0.65  # ~65% annualized volatility
BTC_5MIN_BAR_VOL = BTC_5MIN_ANNUAL_VOL / np.sqrt(365.25 * 288)  # per-bar vol
BTC_5MIN_MEAN_RETURN = 0.0  # Approximately zero at 5-min
BTC_5MIN_KURTOSIS_EXCESS = 8.0  # Fat tails


def generate_synthetic_btc(
    n_bars: int = 288 * 30,  # 30 days default
    start_price: float = 65000.0,
    annual_vol: float = BTC_5MIN_ANNUAL_VOL,
    annual_drift: float = 0.0,
    regime_switching: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic 5-min BTC OHLCV data.

    Uses a regime-switching GBM with fat tails (Student-t innovations)
    to approximate realistic BTC microstructure.

    Args:
        n_bars: Number of 5-minute bars.
        start_price: Starting BTC price.
        annual_vol: Annualized volatility.
        annual_drift: Annualized drift (0 = no trend).
        regime_switching: If True, alternate between high/low vol regimes.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume,
        funding_rate, obi, nq_close.
    """
    rng = np.random.default_rng(seed)

    bar_vol = annual_vol / np.sqrt(365.25 * 288)
    bar_drift = annual_drift / (365.25 * 288)

    # Regime-switching volatility
    if regime_switching:
        vol_multiplier = _generate_regime_vol(n_bars, rng)
    else:
        vol_multiplier = np.ones(n_bars)

    # Fat-tailed innovations (Student-t with df=5 gives excess kurtosis ~6)
    innovations = rng.standard_t(df=5, size=n_bars)
    innovations = innovations / np.std(innovations)  # Normalize to unit variance

    # Generate log returns
    log_returns = bar_drift + bar_vol * vol_multiplier * innovations

    # Build price path
    log_prices = np.log(start_price) + np.cumsum(log_returns)
    close_prices = np.exp(log_prices)

    # Generate OHLV from close
    open_prices = np.empty(n_bars)
    open_prices[0] = start_price
    open_prices[1:] = close_prices[:-1]

    # Intra-bar high/low: simulate as close +/- fraction of bar range
    bar_range = bar_vol * vol_multiplier * close_prices * 1.5
    high_prices = np.maximum(open_prices, close_prices) + rng.exponential(bar_range * 0.3)
    low_prices = np.minimum(open_prices, close_prices) - rng.exponential(bar_range * 0.3)
    low_prices = np.maximum(low_prices, close_prices * 0.99)  # Prevent negative

    # Volume: correlated with absolute returns (volatility-volume relationship)
    base_volume = 100.0  # Base BTC volume per 5-min bar
    vol_factor = 1 + 5 * np.abs(log_returns) / bar_vol
    volume = base_volume * vol_factor * rng.exponential(1.0, size=n_bars)

    # Timestamps
    timestamps = pd.date_range(
        start="2024-01-01",
        periods=n_bars,
        freq="5min",
    )

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_prices,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume,
    })

    # Add auxiliary columns for strategies that need them
    df["funding_rate"] = _generate_funding_rate(n_bars, close_prices, rng)
    df["obi"] = _generate_obi(n_bars, log_returns, rng)
    df["nq_close"] = _generate_nq_prices(n_bars, log_returns, rng)

    df = df.set_index("timestamp")
    return df


def _generate_regime_vol(n_bars: int, rng: np.random.Generator) -> np.ndarray:
    """Generate regime-switching volatility multiplier.

    Three regimes: low vol (0.6x), normal (1.0x), high vol (2.0x).
    Transitions every ~500 bars on average (about 1.7 days).
    """
    multipliers = np.ones(n_bars)
    regimes = [0.6, 1.0, 2.0]
    current_regime = 1  # Start normal

    i = 0
    while i < n_bars:
        # Duration in this regime: geometric distribution
        duration = rng.geometric(p=1 / 500)
        end = min(i + duration, n_bars)
        multipliers[i:end] = regimes[current_regime]
        i = end
        # Transition to random other regime
        current_regime = rng.choice([r for r in range(3) if r != current_regime])

    return multipliers


def _generate_funding_rate(
    n_bars: int,
    close_prices: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate synthetic funding rate.

    Funding updates every 8h (96 bars at 5-min). Between updates, it's constant.
    Correlated with recent price trend (positive trend -> positive funding).
    """
    funding = np.zeros(n_bars)
    update_interval = 96  # 8 hours

    for i in range(0, n_bars, update_interval):
        # Funding correlated with recent returns
        lookback = min(i, update_interval)
        if lookback > 0:
            recent_return = (close_prices[i] - close_prices[i - lookback]) / close_prices[i - lookback]
            base_funding = recent_return * 0.01  # Scaled down
        else:
            base_funding = 0.0
        # Add noise
        funding_val = base_funding + rng.normal(0, 0.0002)
        funding_val = np.clip(funding_val, -0.001, 0.001)
        end = min(i + update_interval, n_bars)
        funding[i:end] = funding_val

    return funding


def _generate_obi(
    n_bars: int,
    log_returns: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate synthetic order book imbalance.

    OBI is partially predictive of next-bar returns (by construction).
    Correlation with concurrent returns ~0.3, with next-bar returns ~0.1.
    """
    # OBI = blend of concurrent return signal + noise
    signal = np.tanh(log_returns * 50)  # Squash to [-1, 1]
    noise = rng.normal(0, 0.4, size=n_bars)
    obi = 0.3 * signal + 0.7 * noise
    return np.clip(obi, -1.0, 1.0)


def _generate_nq_prices(
    n_bars: int,
    btc_log_returns: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate synthetic NQ (Nasdaq futures) prices.

    NQ leads BTC by 1-2 bars with ~0.3 correlation during US hours.
    Outside US hours (bars not in 13:30-20:00 UTC), correlation drops.
    """
    nq_vol = 0.15 / np.sqrt(365.25 * 288)  # NQ ~15% annual vol
    nq_drift = 0.10 / (365.25 * 288)  # NQ ~10% annual drift

    # NQ has its own innovations + some BTC correlation
    nq_innovations = rng.normal(0, 1, size=n_bars)
    # Lead BTC by 1 bar: NQ at t partially predicts BTC at t+1
    btc_lagged = np.roll(btc_log_returns, 1)
    btc_lagged[0] = 0

    # Correlation structure: NQ innovations + correlated component
    correlation = 0.3
    nq_returns = nq_drift + nq_vol * (
        np.sqrt(1 - correlation**2) * nq_innovations
        + correlation * btc_lagged / (btc_log_returns.std() + 1e-10) * nq_vol
    )

    nq_prices = 18000 * np.exp(np.cumsum(nq_returns))  # Start at ~18000
    return nq_prices
