"""Benchmark strategies — null hypotheses that the ML system must beat."""

from ep2_crypto.benchmarks.engine import BacktestEngine
from ep2_crypto.benchmarks.metrics import BacktestMetrics, compute_metrics
from ep2_crypto.benchmarks.statistics import StatisticalAnalyzer
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

__all__ = [
    "BacktestEngine",
    "BacktestMetrics",
    "BuyAndHold",
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
    "compute_metrics",
]
