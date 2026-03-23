"""Benchmark strategies — null hypotheses that the ML system must beat."""

from ep2_crypto.benchmarks.strategies import (
    BuyAndHold,
    CrossMarketLeadLag,
    FundingRateStrategy,
    MeanReversionRSI,
    MovingAverageCrossover,
    NaiveEnsemble,
    OracleStrategy,
    OrderBookImbalance,
    RandomEntry,
    SimpleMomentum,
    VolatilityBreakout,
)
from ep2_crypto.benchmarks.engine import BacktestEngine
from ep2_crypto.benchmarks.metrics import compute_metrics, BacktestMetrics
from ep2_crypto.benchmarks.statistics import StatisticalAnalyzer

__all__ = [
    "BacktestEngine",
    "BacktestMetrics",
    "BuyAndHold",
    "compute_metrics",
    "CrossMarketLeadLag",
    "FundingRateStrategy",
    "MeanReversionRSI",
    "MovingAverageCrossover",
    "NaiveEnsemble",
    "OracleStrategy",
    "OrderBookImbalance",
    "RandomEntry",
    "SimpleMomentum",
    "StatisticalAnalyzer",
    "VolatilityBreakout",
]
