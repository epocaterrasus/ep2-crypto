"""Statistical validation suite for backtest results.

Tests to distinguish genuine edge from noise/overfitting:
  1. Probabilistic Sharpe Ratio (PSR) — accounts for non-normality
  2. Deflated Sharpe Ratio (DSR) — corrects for multiple testing
  3. Permutation test — verifies timing matters (5K permutations)
  4. Block bootstrap CI — confidence intervals on metrics (10K iterations)
  5. Walk-forward stability — CV of fold Sharpes < 0.5

Verdict: GENUINE_EDGE / INCONCLUSIVE / LIKELY_NOISE

References:
  - Bailey & López de Prado (2012): The Sharpe Ratio Efficient Frontier
  - Bailey & López de Prado (2014): The Deflated Sharpe Ratio
  - BACKTESTING_PLAN.md section 3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import structlog
from scipy import stats

from ep2_crypto.backtest.metrics import SQRT_BARS_PER_YEAR

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = structlog.get_logger(__name__)


class Verdict(Enum):
    GENUINE_EDGE = "GENUINE_EDGE"
    INCONCLUSIVE = "INCONCLUSIVE"
    LIKELY_NOISE = "LIKELY_NOISE"


# ---------------------------------------------------------------------------
# 1. Probabilistic Sharpe Ratio (PSR)
# ---------------------------------------------------------------------------
def probabilistic_sharpe_ratio(
    returns: NDArray[np.float64],
    benchmark_sharpe: float = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012).

    Probability that the true Sharpe exceeds a benchmark, accounting
    for skewness and kurtosis of the return distribution.

    PSR = Phi((SR - SR*) / SE(SR))
    where SE(SR) = sqrt((1 - gamma3*SR + (gamma4-1)/4 * SR^2) / (n-1))

    Args:
        returns: Per-bar returns.
        benchmark_sharpe: Benchmark Sharpe (annualized) to test against.

    Returns:
        PSR probability in [0, 1]. Values > 0.95 are significant.
    """
    n = len(returns)
    if n < 10:
        return 0.5  # uninformative

    mean_r = returns.mean()
    std_r = returns.std(ddof=1)
    if std_r < 1e-15:
        return 0.5

    # Per-bar Sharpe
    sr = mean_r / std_r
    sr_benchmark = benchmark_sharpe / SQRT_BARS_PER_YEAR

    # Higher moments
    gamma3 = float(stats.skew(returns))
    gamma4 = float(stats.kurtosis(returns, fisher=True)) + 3  # excess → raw

    # Standard error of the Sharpe ratio
    se_sq = (1.0 - gamma3 * sr + (gamma4 - 1) / 4.0 * sr**2) / (n - 1)
    if se_sq <= 0:
        return 0.5
    se = np.sqrt(se_sq)

    # z-score
    z = (sr - sr_benchmark) / se
    return float(stats.norm.cdf(z))


# ---------------------------------------------------------------------------
# 2. Deflated Sharpe Ratio (DSR)
# ---------------------------------------------------------------------------
def deflated_sharpe_ratio(
    returns: NDArray[np.float64],
    n_trials: int = 1,
) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Adjusts for multiple testing: the expected maximum Sharpe from
    n_trials independent strategies under the null (no edge).

    DSR = PSR(SR, E[max SR under null])

    Args:
        returns: Per-bar returns.
        n_trials: Number of strategy configurations tested.

    Returns:
        DSR probability in [0, 1].
    """
    n = len(returns)
    if n < 10 or n_trials < 1:
        return 0.5

    # Expected maximum Sharpe from n_trials under null (SR=0, std=1)
    # E[max] ≈ sqrt(2 * ln(n_trials)) - (gamma + ln(ln(n_trials))) / (2 * sqrt(2 * ln(n_trials)))
    # where gamma = Euler-Mascheroni constant
    if n_trials == 1:
        e_max_sr = 0.0
    else:
        euler_gamma = 0.5772156649
        val = 2.0 * np.log(n_trials)
        if val <= 0:
            e_max_sr = 0.0
        else:
            sqrt_val = np.sqrt(val)
            log_log = np.log(np.log(n_trials)) if np.log(n_trials) > 0 else 0
            e_max_sr = sqrt_val - (euler_gamma + log_log) / (2.0 * sqrt_val)
            # Convert to per-bar: e_max_sr is already in per-bar units for std=1
            # but we need to scale by the observed std
            e_max_sr = e_max_sr / np.sqrt(n)

    # Annualize the expected max for comparison
    benchmark_annualized = e_max_sr * SQRT_BARS_PER_YEAR

    return probabilistic_sharpe_ratio(returns, benchmark_annualized)


# ---------------------------------------------------------------------------
# 3. Permutation Test
# ---------------------------------------------------------------------------
def permutation_test(
    returns: NDArray[np.float64],
    positions: NDArray[np.float64],
    n_permutations: int = 5000,
    seed: int = 42,
) -> dict[str, float]:
    """Permutation test: does signal timing matter?

    Shuffles the position signal while keeping returns fixed.
    If the original Sharpe exceeds most permuted Sharpes, timing matters.

    Args:
        returns: Per-bar raw (market) returns.
        positions: Per-bar positions (+1, -1, 0).
        n_permutations: Number of shuffled trials.
        seed: RNG seed.

    Returns:
        Dict with: observed_sharpe, mean_permuted, p_value.
    """
    rng = np.random.default_rng(seed)

    # Observed strategy returns
    strategy_returns = returns * positions
    if strategy_returns.std() < 1e-15:
        return {"observed_sharpe": 0.0, "mean_permuted": 0.0, "p_value": 1.0}
    observed_sharpe = float(strategy_returns.mean() / strategy_returns.std() * SQRT_BARS_PER_YEAR)

    # Permuted Sharpes
    permuted_sharpes = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm_pos = rng.permutation(positions)
        perm_returns = returns * perm_pos
        std = perm_returns.std()
        if std > 1e-15:
            permuted_sharpes[i] = perm_returns.mean() / std * SQRT_BARS_PER_YEAR
        else:
            permuted_sharpes[i] = 0.0

    p_value = float((permuted_sharpes >= observed_sharpe).mean())

    return {
        "observed_sharpe": observed_sharpe,
        "mean_permuted": float(permuted_sharpes.mean()),
        "std_permuted": float(permuted_sharpes.std()),
        "p_value": p_value,
        "significant_05": p_value < 0.05,
        "significant_01": p_value < 0.01,
    }


# ---------------------------------------------------------------------------
# 4. Block Bootstrap CI
# ---------------------------------------------------------------------------
def block_bootstrap_ci(
    returns: NDArray[np.float64],
    n_iterations: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Block bootstrap confidence intervals on Sharpe ratio.

    Uses overlapping blocks of length T^(1/3) to preserve serial correlation.

    Args:
        returns: Per-bar returns.
        n_iterations: Number of bootstrap samples.
        alpha: Significance level for CI.
        seed: RNG seed.

    Returns:
        Dict with: mean_sharpe, ci_lower, ci_upper, std_sharpe.
    """
    rng = np.random.default_rng(seed)
    n = len(returns)
    if n < 10:
        return {"mean_sharpe": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std_sharpe": 0.0}

    # Block size: T^(1/3)
    block_size = max(1, round(n ** (1 / 3)))
    n_blocks = max(1, (n + block_size - 1) // block_size)

    sharpes = np.zeros(n_iterations)

    for i in range(n_iterations):
        # Draw random block starting points
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        # Concatenate blocks
        boot_indices = np.concatenate([np.arange(s, min(s + block_size, n)) for s in starts])[:n]

        boot_returns = returns[boot_indices]
        std = boot_returns.std(ddof=1)
        if std > 1e-15:
            sharpes[i] = boot_returns.mean() / std * SQRT_BARS_PER_YEAR
        else:
            sharpes[i] = 0.0

    ci_lower = float(np.percentile(sharpes, 100 * alpha / 2))
    ci_upper = float(np.percentile(sharpes, 100 * (1 - alpha / 2)))

    return {
        "mean_sharpe": float(sharpes.mean()),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_sharpe": float(sharpes.std()),
        "ci_lower_positive": ci_lower > 0,
    }


# ---------------------------------------------------------------------------
# 5. Walk-Forward Stability
# ---------------------------------------------------------------------------
def walk_forward_stability(
    fold_sharpes: list[float] | NDArray[np.float64],
) -> dict[str, float]:
    """Assess stability of Sharpe across walk-forward folds.

    CV of fold Sharpes < 0.5 indicates stable performance.

    Args:
        fold_sharpes: Sharpe ratio from each walk-forward fold.

    Returns:
        Dict with: mean, std, cv, stable (cv < 0.5), pct_positive.
    """
    arr = np.array(fold_sharpes, dtype=np.float64)
    if len(arr) < 2:
        return {"mean": 0.0, "std": 0.0, "cv": float("inf"), "stable": False, "pct_positive": 0.0}

    mean_s = float(arr.mean())
    std_s = float(arr.std(ddof=1))
    cv = std_s / abs(mean_s) if abs(mean_s) > 1e-10 else float("inf")
    pct_pos = float((arr > 0).mean())

    return {
        "mean": mean_s,
        "std": std_s,
        "cv": cv,
        "stable": cv < 0.5,
        "pct_positive": pct_pos,
        "n_folds": len(arr),
    }


# ---------------------------------------------------------------------------
# 6. Combined Validation Suite
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    """Comprehensive validation output."""

    verdict: Verdict
    psr: float
    dsr: float
    permutation_p_value: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    wf_stability_cv: float
    tests_passed: int
    tests_total: int
    details: dict[str, dict] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Verdict:              {self.verdict.value}",
            f"Tests Passed:         {self.tests_passed}/{self.tests_total}",
            f"PSR:                  {self.psr:.4f} ({'PASS' if self.psr > 0.95 else 'FAIL'})",
            f"DSR:                  {self.dsr:.4f} ({'PASS' if self.dsr > 0.95 else 'FAIL'})",
            f"Permutation p-value:  {self.permutation_p_value:.4f}"
            f" ({'PASS' if self.permutation_p_value < 0.05 else 'FAIL'})",
            f"Bootstrap 95% CI:     [{self.bootstrap_ci_lower:.3f},"
            f" {self.bootstrap_ci_upper:.3f}]"
            f" ({'PASS' if self.bootstrap_ci_lower > 0 else 'FAIL'})",
            f"WF Stability CV:      {self.wf_stability_cv:.3f}"
            f" ({'PASS' if self.wf_stability_cv < 0.5 else 'FAIL'})",
        ]
        return "\n".join(lines)


def run_validation_suite(
    returns: NDArray[np.float64],
    positions: NDArray[np.float64] | None = None,
    raw_returns: NDArray[np.float64] | None = None,
    fold_sharpes: list[float] | None = None,
    n_trials: int = 1,
    n_permutations: int = 5000,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> ValidationResult:
    """Run the full statistical validation suite.

    Args:
        returns: Net strategy returns (after costs).
        positions: Position signals (for permutation test).
        raw_returns: Raw market returns (for permutation test).
        fold_sharpes: Sharpe from each walk-forward fold.
        n_trials: Number of strategies tested (for DSR).
        n_permutations: Permutation test iterations.
        n_bootstrap: Bootstrap iterations.
        seed: RNG seed.

    Returns:
        ValidationResult with verdict and test details.
    """
    tests_passed = 0
    tests_total = 0
    details: dict[str, dict] = {}

    # 1. PSR
    psr = probabilistic_sharpe_ratio(returns)
    details["psr"] = {"value": psr, "threshold": 0.95, "passed": psr > 0.95}
    tests_total += 1
    if psr > 0.95:
        tests_passed += 1

    # 2. DSR
    dsr = deflated_sharpe_ratio(returns, n_trials)
    details["dsr"] = {"value": dsr, "threshold": 0.95, "passed": dsr > 0.95}
    tests_total += 1
    if dsr > 0.95:
        tests_passed += 1

    # 3. Permutation test
    perm_p = 1.0
    if positions is not None and raw_returns is not None:
        perm = permutation_test(raw_returns, positions, n_permutations, seed)
        perm_p = perm["p_value"]
        details["permutation"] = perm
        tests_total += 1
        if perm_p < 0.05:
            tests_passed += 1

    # 4. Block bootstrap
    boot = block_bootstrap_ci(returns, n_bootstrap, seed=seed)
    details["bootstrap"] = boot
    tests_total += 1
    if boot["ci_lower"] > 0:
        tests_passed += 1

    # 5. Walk-forward stability
    wf_cv = float("inf")
    if fold_sharpes is not None and len(fold_sharpes) >= 2:
        wf = walk_forward_stability(fold_sharpes)
        wf_cv = wf["cv"]
        details["walk_forward_stability"] = wf
        tests_total += 1
        if wf["stable"]:
            tests_passed += 1

    # Verdict
    if tests_total == 0:
        verdict = Verdict.INCONCLUSIVE
    elif tests_passed >= tests_total * 0.6:
        verdict = Verdict.GENUINE_EDGE
    elif tests_passed >= tests_total * 0.3:
        verdict = Verdict.INCONCLUSIVE
    else:
        verdict = Verdict.LIKELY_NOISE

    result = ValidationResult(
        verdict=verdict,
        psr=psr,
        dsr=dsr,
        permutation_p_value=perm_p,
        bootstrap_ci_lower=boot["ci_lower"],
        bootstrap_ci_upper=boot["ci_upper"],
        wf_stability_cv=wf_cv,
        tests_passed=tests_passed,
        tests_total=tests_total,
        details=details,
    )

    logger.info(
        "validation_complete",
        verdict=verdict.value,
        tests_passed=tests_passed,
        tests_total=tests_total,
        psr=psr,
        dsr=dsr,
    )

    return result
