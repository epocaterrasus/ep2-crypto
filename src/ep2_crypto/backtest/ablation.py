"""Ablation study: full system vs each component removed.

Runs the BacktestEngine under multiple configurations and reports
the Sharpe delta for each ablation variant, revealing the contribution
of confidence gating, risk engine, drawdown gate, and kill switches.

Usage::

    study = AblationStudy(initial_equity=50_000.0)
    results = study.run(opens, highs, lows, closes, volumes, timestamps_ms,
                        signals, confidences, regime_labels)
    print(results.summary())
"""

from __future__ import annotations

import dataclasses
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

from ep2_crypto.backtest.engine import BacktestConfig, BacktestEngine
from ep2_crypto.risk.config import RiskConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ep2_crypto.backtest.metrics import BacktestResult

logger = structlog.get_logger(__name__)


class AblationVariant(StrEnum):
    """Which component is removed in this ablation run."""

    FULL = "full"
    NO_CONFIDENCE_GATE = "no_confidence_gate"
    NO_RISK_ENGINE = "no_risk_engine"
    NO_DRAWDOWN_GATE = "no_drawdown_gate"
    NO_KILL_SWITCHES = "no_kill_switches"
    NO_REGIME_FILTER = "no_regime_filter"


@dataclasses.dataclass
class AblationVariantResult:
    """Single ablation run output."""

    variant: AblationVariant
    sharpe: float
    sortino: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    total_return: float
    delta_sharpe: float = 0.0  # vs FULL (populated by AblationResult)


@dataclasses.dataclass
class AblationResult:
    """Collection of all ablation variant results."""

    variants: list[AblationVariantResult]

    def __post_init__(self) -> None:
        full = self._get(AblationVariant.FULL)
        if full is not None:
            for v in self.variants:
                v.delta_sharpe = v.sharpe - full.sharpe

    def _get(self, variant: AblationVariant) -> AblationVariantResult | None:
        for v in self.variants:
            if v.variant == variant:
                return v
        return None

    def to_dict(self) -> list[dict[str, Any]]:
        return [dataclasses.asdict(v) for v in self.variants]

    def summary(self) -> str:
        header = (
            f"{'Variant':<28} {'Sharpe':>8} {'ΔSharpe':>9} "
            f"{'Sortino':>8} {'MaxDD':>8} {'Trades':>7} {'WinRate':>8}"
        )
        sep = "-" * len(header)
        rows = [header, sep]
        for v in self.variants:
            rows.append(
                f"{v.variant.value:<28} {v.sharpe:>8.3f} {v.delta_sharpe:>+9.3f} "
                f"{v.sortino:>8.3f} {v.max_drawdown:>8.2%} "
                f"{v.total_trades:>7d} {v.win_rate:>8.2%}"
            )
        return "\n".join(rows)


def _make_variant_result(
    variant: AblationVariant,
    result: BacktestResult,
) -> AblationVariantResult:
    return AblationVariantResult(
        variant=variant,
        sharpe=result.sharpe_ratio,
        sortino=result.sortino_ratio,
        max_drawdown=result.max_drawdown,
        total_trades=result.total_trades,
        win_rate=result.win_rate,
        total_return=result.total_return,
    )


def _run_engine(
    config: BacktestConfig,
    opens: NDArray[np.float64],
    highs: NDArray[np.float64],
    lows: NDArray[np.float64],
    closes: NDArray[np.float64],
    volumes: NDArray[np.float64],
    timestamps_ms: NDArray[np.int64],
    signals: NDArray[np.int8],
    confidences: NDArray[np.float64],
    funding_rates: NDArray[np.float64] | None,
    regime_labels: NDArray[np.int32] | None,
) -> BacktestResult:
    engine = BacktestEngine(config=config)
    return engine.run(
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        volumes=volumes,
        timestamps_ms=timestamps_ms,
        signals=signals,
        confidences=confidences,
        funding_rates=funding_rates,
        regime_labels=regime_labels,
    )


class AblationStudy:
    """Run the backtest engine under multiple ablation configurations.

    Each variant disables one component and re-runs the engine.
    The delta Sharpe shows each component's contribution.

    Args:
        initial_equity: Starting capital in USD.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        initial_equity: float = 50_000.0,
        seed: int = 42,
    ) -> None:
        self._initial_equity = initial_equity
        self._seed = seed

    def run(
        self,
        opens: NDArray[np.float64],
        highs: NDArray[np.float64],
        lows: NDArray[np.float64],
        closes: NDArray[np.float64],
        volumes: NDArray[np.float64],
        timestamps_ms: NDArray[np.int64],
        signals: NDArray[np.int8],
        confidences: NDArray[np.float64],
        funding_rates: NDArray[np.float64] | None = None,
        regime_labels: NDArray[np.int32] | None = None,
        variants: list[AblationVariant] | None = None,
    ) -> AblationResult:
        """Run ablation study across all (or specified) variants.

        Args:
            opens/highs/lows/closes/volumes: OHLCV arrays.
            timestamps_ms: Bar timestamps in milliseconds.
            signals: Per-bar signal (+1, -1, 0).
            confidences: Per-bar confidence [0,1].
            funding_rates: Per-bar funding rate (optional).
            regime_labels: Per-bar regime (optional).
            variants: Which variants to run (default: all).

        Returns:
            AblationResult with per-variant metrics and delta Sharpe.
        """
        if variants is None:
            variants = list(AblationVariant)

        results: list[AblationVariantResult] = []

        for variant in variants:
            logger.info("ablation.running", variant=variant.value)
            cfg, sigs, confs = self._build_config(variant, signals, confidences)
            result = _run_engine(
                config=cfg,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                volumes=volumes,
                timestamps_ms=timestamps_ms,
                signals=sigs,
                confidences=confs,
                funding_rates=funding_rates,
                regime_labels=regime_labels
                if variant != AblationVariant.NO_REGIME_FILTER
                else None,
            )
            results.append(_make_variant_result(variant, result))
            logger.info(
                "ablation.done",
                variant=variant.value,
                sharpe=round(result.sharpe_ratio, 3),
                trades=result.total_trades,
            )

        return AblationResult(variants=results)

    def _build_config(
        self,
        variant: AblationVariant,
        signals: NDArray[np.int8],
        confidences: NDArray[np.float64],
    ) -> tuple[BacktestConfig, NDArray[np.int8], NDArray[np.float64]]:
        """Return (BacktestConfig, signals, confidences) for this variant.

        Modifies config or signals to disable the target component.
        """
        if variant == AblationVariant.FULL:
            return (
                BacktestConfig(
                    initial_equity=self._initial_equity,
                    seed=self._seed,
                ),
                signals.copy(),
                confidences.copy(),
            )

        if variant == AblationVariant.NO_CONFIDENCE_GATE:
            # Set threshold to 0: every signal passes gating
            cfg = BacktestConfig(
                initial_equity=self._initial_equity,
                seed=self._seed,
                confidence_threshold=0.0,
            )
            return cfg, signals.copy(), np.ones_like(confidences)

        if variant == AblationVariant.NO_RISK_ENGINE:
            # No RiskConfig: engine uses no risk manager (bare engine with wide limits)
            cfg = BacktestConfig(
                initial_equity=self._initial_equity,
                seed=self._seed,
                risk_config=RiskConfig(
                    daily_loss_limit=0.99,
                    weekly_loss_limit=0.99,
                    max_drawdown_halt=0.99,
                    consecutive_loss_limit=10_000,
                    max_trades_per_day=10_000,
                    max_position_fraction=1.0,
                    kelly_fraction=1.0,
                    max_risk_per_trade=0.05,
                    enforce_trading_hours=False,
                    weekend_size_reduction=0.0,
                    min_volatility_ann=0.0,
                    max_volatility_ann=100.0,
                ),
            )
            return cfg, signals.copy(), confidences.copy()

        if variant == AblationVariant.NO_DRAWDOWN_GATE:
            # Disable drawdown gate by setting halt at 99%
            cfg = BacktestConfig(
                initial_equity=self._initial_equity,
                seed=self._seed,
                risk_config=RiskConfig(
                    max_drawdown_halt=0.99,
                    drawdown_cooldown_bars=0,
                ),
            )
            return cfg, signals.copy(), confidences.copy()

        if variant == AblationVariant.NO_KILL_SWITCHES:
            # Disable kill switches by setting all limits to near-infinity
            cfg = BacktestConfig(
                initial_equity=self._initial_equity,
                seed=self._seed,
                risk_config=RiskConfig(
                    daily_loss_limit=0.99,
                    weekly_loss_limit=0.99,
                    max_drawdown_halt=0.99,
                    consecutive_loss_limit=10_000,
                    max_trades_per_day=10_000,
                ),
            )
            return cfg, signals.copy(), confidences.copy()

        if variant == AblationVariant.NO_REGIME_FILTER:
            # Regime filtering is removed by passing None regime_labels (handled in run())
            cfg = BacktestConfig(
                initial_equity=self._initial_equity,
                seed=self._seed,
            )
            return cfg, signals.copy(), confidences.copy()

        msg = f"Unknown variant: {variant}"
        raise ValueError(msg)
