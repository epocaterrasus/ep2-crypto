"""Final performance report: OOS Sharpe CI, DSR, regime breakdown, Monte Carlo ruin.

Consolidates results from the full validation + ablation + stress pipeline
into a single structured report with a pass/fail verdict.

Key outputs (per BACKTESTING_PLAN.md):
  - OOS Sharpe with 95% bootstrap CI (block bootstrap, 10K iterations)
  - Deflated Sharpe Ratio (multiple-testing corrected)
  - Regime breakdown (per-regime Sharpe, return, drawdown)
  - Cost sensitivity and break-even cost
  - Monte Carlo ruin probability at 20% drawdown threshold
  - Overall deployment verdict: DEPLOY / CAUTION / DO_NOT_DEPLOY

Usage::

    reporter = PerformanceReporter()
    report = reporter.generate(
        returns=oos_returns,
        regime_labels=regime_labels,
        fold_sharpes=fold_sharpes,
        n_trials=50,
    )
    print(report.summary())
    print(report.to_markdown())
"""

from __future__ import annotations

import dataclasses
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog

from ep2_crypto.backtest.metrics import (
    BARS_PER_DAY,
    cost_sensitivity,
    find_breakeven_cost,
    lo_corrected_sharpe,
    max_drawdown_info,
    regime_metrics,
)
from ep2_crypto.backtest.validation import (
    ValidationResult,
    block_bootstrap_ci,
    deflated_sharpe_ratio,
    run_validation_suite,
    walk_forward_stability,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Deployment verdict
# ---------------------------------------------------------------------------


class Verdict(StrEnum):
    DEPLOY = "DEPLOY"
    CAUTION = "CAUTION"
    DO_NOT_DEPLOY = "DO_NOT_DEPLOY"


# ---------------------------------------------------------------------------
# Monte Carlo ruin
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class MonteCarloRuin:
    """Monte Carlo ruin probability results.

    Simulates N-trade paths by resampling from the observed trade returns
    and counts paths that breach the ruin threshold.
    """

    n_paths: int
    n_trades_per_path: int
    ruin_threshold: float  # e.g. 0.20 for 20% drawdown
    ruin_probability: float  # P(max_drawdown > ruin_threshold)
    median_max_drawdown: float
    p95_max_drawdown: float
    passed: bool  # ruin_probability < 0.05

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"Monte Carlo Ruin ({self.n_paths:,} paths × {self.n_trades_per_path:,} trades)\n"
            f"  P(ruin at {self.ruin_threshold:.0%} DD): {self.ruin_probability:.2%} [{status}]\n"
            f"  Median max DD: {self.median_max_drawdown:.2%}\n"
            f"  95th pctile max DD: {self.p95_max_drawdown:.2%}"
        )


def run_monte_carlo_ruin(
    trade_returns: NDArray[np.float64],
    n_paths: int = 10_000,
    n_trades_per_path: int = 5_000,
    ruin_threshold: float = 0.20,
    seed: int = 42,
) -> MonteCarloRuin:
    """Estimate ruin probability via trade-return resampling.

    Args:
        trade_returns: Per-trade return fractions (e.g. 0.01 = +1%).
        n_paths: Number of Monte Carlo paths.
        n_trades_per_path: Trades simulated per path.
        ruin_threshold: Drawdown level considered ruin.
        seed: RNG seed.

    Returns:
        MonteCarloRuin with probability statistics.
    """
    if len(trade_returns) < 2:
        return MonteCarloRuin(
            n_paths=n_paths,
            n_trades_per_path=n_trades_per_path,
            ruin_threshold=ruin_threshold,
            ruin_probability=0.0,
            median_max_drawdown=0.0,
            p95_max_drawdown=0.0,
            passed=True,
        )

    rng = np.random.default_rng(seed)
    max_dds = np.zeros(n_paths)

    for i in range(n_paths):
        # Resample trade returns with replacement (block of 1 trade per draw)
        sampled = rng.choice(trade_returns, size=n_trades_per_path, replace=True)
        # Equity curve starting at 1.0
        equity = np.cumprod(1 + sampled)
        equity = np.insert(equity, 0, 1.0)
        running_max = np.maximum.accumulate(equity)
        dd_series = (running_max - equity) / np.where(running_max > 0, running_max, 1.0)
        max_dds[i] = dd_series.max()

    ruin_prob = float((max_dds > ruin_threshold).mean())
    median_dd = float(np.median(max_dds))
    p95_dd = float(np.percentile(max_dds, 95))

    return MonteCarloRuin(
        n_paths=n_paths,
        n_trades_per_path=n_trades_per_path,
        ruin_threshold=ruin_threshold,
        ruin_probability=ruin_prob,
        median_max_drawdown=median_dd,
        p95_max_drawdown=p95_dd,
        passed=ruin_prob < 0.05,
    )


# ---------------------------------------------------------------------------
# Performance Report
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PerformanceReport:
    """Consolidated OOS performance report.

    All metrics are computed on out-of-sample data only.
    """

    # Core Sharpe
    sharpe_lo_corrected: float
    sharpe_ci_lower: float  # 95% bootstrap lower bound
    sharpe_ci_upper: float
    sharpe_std: float

    # Statistical tests
    dsr: float  # Deflated Sharpe Ratio (> 0.95 = pass)
    psr: float  # Probabilistic Sharpe Ratio
    permutation_p_value: float

    # Walk-forward stability
    wf_stability_cv: float  # CV of fold Sharpes (< 0.5 = pass)
    wf_stability_pct_positive: float

    # Cost analysis
    breakeven_cost_bps: float
    cost_sensitivity_rows: list[dict[str, float]]

    # Regime breakdown
    regime_breakdown: dict[int, dict[str, float]]

    # Drawdown
    max_drawdown: float
    max_drawdown_duration_bars: int

    # Monte Carlo
    monte_carlo: MonteCarloRuin

    # Summary counts
    n_bars: int
    n_trades: int
    total_return: float

    # Overall verdict
    verdict: Verdict
    criteria_passed: int
    criteria_total: int

    # Raw validation result
    validation: ValidationResult | None = dataclasses.field(default=None, compare=False, repr=False)

    def summary(self) -> str:
        verdict_line = f"{'=' * 60}\nVERDICT: {self.verdict.value}\n{'=' * 60}"
        lines = [
            verdict_line,
            f"OOS Sharpe (Lo-corrected): {self.sharpe_lo_corrected:.3f}",
            f"  95% CI: [{self.sharpe_ci_lower:.3f}, {self.sharpe_ci_upper:.3f}]",
            f"  CI lower > 0: {'PASS' if self.sharpe_ci_lower > 0 else 'FAIL'}",
            "",
            f"DSR: {self.dsr:.4f} ({'PASS' if self.dsr > 0.95 else 'FAIL'})",
            f"PSR: {self.psr:.4f} ({'PASS' if self.psr > 0.95 else 'FAIL'})",
            f"Permutation p-value: {self.permutation_p_value:.4f}"
            f" ({'PASS' if self.permutation_p_value < 0.05 else 'FAIL'})",
            "",
            f"Walk-forward CV: {self.wf_stability_cv:.3f}"
            f" ({'PASS' if self.wf_stability_cv < 0.5 else 'FAIL'})",
            f"  Folds positive: {self.wf_stability_pct_positive:.0%}",
            "",
            f"Break-even cost: {self.breakeven_cost_bps:.1f} bps"
            f" ({'PASS' if self.breakeven_cost_bps > 15 else 'FAIL'})",
            "",
            f"Max drawdown: {self.max_drawdown:.2%}",
            f"Max DD duration: {self.max_drawdown_duration_bars} bars"
            f" ({self.max_drawdown_duration_bars / BARS_PER_DAY:.1f} days)",
            "",
            self.monte_carlo.summary(),
            "",
            f"OOS bars: {self.n_bars:,} | trades: {self.n_trades:,}"
            f" | total return: {self.total_return:.2%}",
            f"Criteria: {self.criteria_passed}/{self.criteria_total} passed",
        ]
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines = [
            "# Performance Report",
            "",
            f"**Verdict**: {self.verdict.value} ({self.criteria_passed}/{self.criteria_total} criteria)",
            "",
            "## Statistical Tests",
            "| Metric | Value | Pass? |",
            "|--------|-------|-------|",
            f"| OOS Sharpe (Lo) | {self.sharpe_lo_corrected:.3f} | — |",
            f"| 95% CI lower | {self.sharpe_ci_lower:.3f} | {'✅' if self.sharpe_ci_lower > 0 else '❌'} |",
            f"| DSR | {self.dsr:.4f} | {'✅' if self.dsr > 0.95 else '❌'} |",
            f"| PSR | {self.psr:.4f} | {'✅' if self.psr > 0.95 else '❌'} |",
            f"| Permutation p | {self.permutation_p_value:.4f} | {'✅' if self.permutation_p_value < 0.05 else '❌'} |",
            f"| WF stability CV | {self.wf_stability_cv:.3f} | {'✅' if self.wf_stability_cv < 0.5 else '❌'} |",
            f"| Break-even cost | {self.breakeven_cost_bps:.1f} bps | {'✅' if self.breakeven_cost_bps > 15 else '❌'} |",
            f"| Monte Carlo P(ruin) | {self.monte_carlo.ruin_probability:.2%} | {'✅' if self.monte_carlo.passed else '❌'} |",
            "",
            "## Regime Breakdown",
        ]
        if self.regime_breakdown:
            lines += [
                "| Regime | Sharpe | Return | Max DD | Bars |",
                "|--------|--------|--------|--------|------|",
            ]
            for rid, stats in sorted(self.regime_breakdown.items()):
                lines.append(
                    f"| {rid} | {stats['sharpe']:.3f} | {stats['total_return']:.2%} "
                    f"| {stats['max_drawdown']:.2%} | {stats['n_bars']:,} |"
                )
        else:
            lines.append("No regime labels provided.")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d.pop("validation", None)
        return d


# ---------------------------------------------------------------------------
# PerformanceReporter
# ---------------------------------------------------------------------------


class PerformanceReporter:
    """Build a PerformanceReport from OOS returns and optional metadata.

    Args:
        n_bootstrap: Bootstrap iterations for CI (default: 10_000).
        n_permutations: Permutation test iterations (default: 5_000).
        n_mc_paths: Monte Carlo ruin paths (default: 10_000).
        n_mc_trades: Trades per MC path (default: 5_000).
        ruin_threshold: DD level for MC ruin (default: 0.20).
        seed: RNG seed.
    """

    def __init__(
        self,
        n_bootstrap: int = 10_000,
        n_permutations: int = 5_000,
        n_mc_paths: int = 10_000,
        n_mc_trades: int = 5_000,
        ruin_threshold: float = 0.20,
        seed: int = 42,
    ) -> None:
        self._n_bootstrap = n_bootstrap
        self._n_permutations = n_permutations
        self._n_mc_paths = n_mc_paths
        self._n_mc_trades = n_mc_trades
        self._ruin_threshold = ruin_threshold
        self._seed = seed

    def generate(
        self,
        returns: NDArray[np.float64],
        positions: NDArray[np.float64] | None = None,
        raw_market_returns: NDArray[np.float64] | None = None,
        regime_labels: NDArray[np.int32] | None = None,
        fold_sharpes: list[float] | None = None,
        trade_returns: NDArray[np.float64] | None = None,
        n_trades: int | None = None,
        n_trials: int = 1,
    ) -> PerformanceReport:
        """Generate the complete performance report.

        Args:
            returns: Per-bar net OOS returns (after costs).
            positions: Per-bar position signs (+1, -1, 0) for permutation test.
            raw_market_returns: Per-bar market returns for permutation test.
            regime_labels: Per-bar regime label for breakdown.
            fold_sharpes: Walk-forward fold Sharpe ratios.
            trade_returns: Per-trade return fractions for Monte Carlo.
            n_trades: Number of trades (estimated from returns if not provided).
            n_trials: Number of strategy configs tested (for DSR).

        Returns:
            PerformanceReport.
        """
        n = len(returns)
        logger.info("performance_report.start", n_bars=n, n_trials=n_trials)

        # ----- Sharpe + CI -----
        sharpe = lo_corrected_sharpe(returns)
        ci = block_bootstrap_ci(returns, n_iterations=self._n_bootstrap, seed=self._seed)

        # ----- DSR / PSR -----
        dsr = deflated_sharpe_ratio(returns, n_trials=n_trials)
        from ep2_crypto.backtest.validation import probabilistic_sharpe_ratio

        psr = probabilistic_sharpe_ratio(returns)

        # ----- Validation suite -----
        validation = run_validation_suite(
            returns=returns,
            positions=positions,
            raw_returns=raw_market_returns,
            fold_sharpes=fold_sharpes or [],
            n_trials=n_trials,
            n_permutations=self._n_permutations,
            n_bootstrap=self._n_bootstrap,
            seed=self._seed,
        )

        # ----- Walk-forward stability -----
        if fold_sharpes:
            wf = walk_forward_stability(fold_sharpes)
            wf_cv = wf["cv"]
            wf_pct_pos = wf["pct_positive"]
        else:
            wf_cv = float("inf")
            wf_pct_pos = float((returns > 0).mean())

        # ----- Cost sensitivity -----
        n_tr = n_trades if n_trades is not None else max(1, n // 6)
        cost_rows = cost_sensitivity(returns, n_trades=n_tr, n_bars=n)
        breakeven_bps = find_breakeven_cost(cost_rows)

        # ----- Regime breakdown -----
        regime_bd = regime_metrics(returns, regime_labels)

        # ----- Drawdown -----
        equity = np.cumprod(1 + returns)
        max_dd, max_dd_dur, _ = max_drawdown_info(equity)

        # ----- Monte Carlo ruin -----
        mc_returns = trade_returns if trade_returns is not None else returns
        mc = run_monte_carlo_ruin(
            mc_returns,
            n_paths=self._n_mc_paths,
            n_trades_per_path=self._n_mc_trades,
            ruin_threshold=self._ruin_threshold,
            seed=self._seed,
        )

        # ----- Total return -----
        total_ret = float(np.prod(1 + returns) - 1)

        # ----- Permutation p-value -----
        perm_p = validation.permutation_p_value

        # ----- Verdict -----
        criteria_passed, criteria_total = self._score_criteria(
            sharpe_ci_lower=ci["ci_lower"],
            dsr=dsr,
            psr=psr,
            permutation_p=perm_p,
            wf_cv=wf_cv,
            breakeven_bps=breakeven_bps,
            mc_ruin_passed=mc.passed,
        )
        verdict = self._compute_verdict(criteria_passed, criteria_total)

        logger.info(
            "performance_report.done",
            verdict=verdict.value,
            sharpe=round(sharpe, 3),
            dsr=round(dsr, 4),
            criteria=f"{criteria_passed}/{criteria_total}",
        )

        return PerformanceReport(
            sharpe_lo_corrected=sharpe,
            sharpe_ci_lower=ci["ci_lower"],
            sharpe_ci_upper=ci["ci_upper"],
            sharpe_std=ci["std_sharpe"],
            dsr=dsr,
            psr=psr,
            permutation_p_value=perm_p,
            wf_stability_cv=wf_cv,
            wf_stability_pct_positive=wf_pct_pos,
            breakeven_cost_bps=breakeven_bps,
            cost_sensitivity_rows=cost_rows,
            regime_breakdown=regime_bd,
            max_drawdown=max_dd,
            max_drawdown_duration_bars=max_dd_dur,
            monte_carlo=mc,
            n_bars=n,
            n_trades=n_tr,
            total_return=total_ret,
            verdict=verdict,
            criteria_passed=criteria_passed,
            criteria_total=criteria_total,
            validation=validation,
        )

    @staticmethod
    def _score_criteria(
        sharpe_ci_lower: float,
        dsr: float,
        psr: float,
        permutation_p: float,
        wf_cv: float,
        breakeven_bps: float,
        mc_ruin_passed: bool,
    ) -> tuple[int, int]:
        """Count how many deployment criteria are met."""
        passed = 0
        total = 7

        if sharpe_ci_lower > 0:
            passed += 1
        if dsr > 0.95:
            passed += 1
        if psr > 0.95:
            passed += 1
        if permutation_p < 0.05:
            passed += 1
        if wf_cv < 0.5:
            passed += 1
        if breakeven_bps > 15:
            passed += 1
        if mc_ruin_passed:
            passed += 1

        return passed, total

    @staticmethod
    def _compute_verdict(criteria_passed: int, criteria_total: int) -> Verdict:
        """Derive overall deployment verdict.

        DEPLOY:         >= 6/7 criteria met
        CAUTION:        4-5/7 criteria met
        DO_NOT_DEPLOY:  < 4/7 criteria met
        """
        fraction = criteria_passed / max(criteria_total, 1)
        if fraction >= 6 / 7:
            return Verdict.DEPLOY
        if fraction >= 4 / 7:
            return Verdict.CAUTION
        return Verdict.DO_NOT_DEPLOY
