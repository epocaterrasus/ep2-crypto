"""
Statistical Validation Tests for Trading Strategy Edge Detection
================================================================

Comprehensive suite of tests to determine if a trading strategy has genuine
edge versus luck. Designed for the ep2-crypto 5-min BTC prediction system.

References:
- Lo (2002) "The Statistics of Sharpe Ratios"
- Bailey & Lopez de Prado (2014) "The Deflated Sharpe Ratio"
- Bailey, Borwein, Lopez de Prado, Zhu (2015) "Probability of Backtest Overfitting"
- White (2000) "A Reality Check for Data Snooping"
- Hansen (2005) "A Test for Superior Predictive Ability"
- Lopez de Prado (2018) "Advances in Financial Machine Learning"

Dependencies:
    uv add numpy scipy pandas statsmodels arch scikit-learn
"""

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, binom_test, binomtest

logger = logging.getLogger(__name__)


# =============================================================================
# 1. T-TEST ON STRATEGY RETURNS (Newey-West adjusted)
# =============================================================================
#
# QUESTION: Is the mean return statistically different from zero?
#
# FORMULA (standard t-test):
#   t = mean(r) / (std(r) / sqrt(N))
#
# PROBLEM: Strategy returns are autocorrelated. Standard errors are biased
# downward, inflating the t-statistic. Solution: Newey-West HAC standard errors.
#
# NEWEY-WEST ADJUSTMENT:
#   The variance estimator accounts for autocorrelation up to L lags:
#   Var_NW(mean) = (1/T^2) * sum_{j=-L}^{L} w(j) * sum_{t} e_t * e_{t-j}
#   where w(j) = 1 - |j|/(L+1) (Bartlett kernel)
#   Optimal lag: L = floor(4*(T/100)^(2/9))  [Andrews 1991]
#
# P-VALUE THRESHOLD for trading:
#   - Academic standard: p < 0.05 (two-tailed)
#   - For trading strategies: p < 0.01 recommended (higher bar due to
#     multiple testing, survivorship bias, implementation slippage)
#   - Harvey, Liu & Zhu (2016) argue for t > 3.0 (roughly p < 0.003)
#     for new factor/strategy discovery
#
# MINIMUM SAMPLE SIZE:
#   - Rule of thumb: N > 30 for CLT to kick in
#   - For autocorrelated returns: N > 100 trades minimum
#   - For reliable inference: N > 250 trades (1 year of daily)
#   - For 5-min BTC: ~750 trades minimum (robust to regime shifts)


@dataclass
class TTestResult:
    t_statistic: float
    p_value: float
    mean_return: float
    std_error: float  # Newey-West adjusted
    n_observations: int
    newey_west_lags: int
    ci_lower: float  # 95% CI
    ci_upper: float
    significant_at_005: bool
    significant_at_001: bool


def ttest_strategy_returns(
    returns: np.ndarray,
    max_lags: Optional[int] = None,
    alpha: float = 0.05,
) -> TTestResult:
    """
    T-test on strategy returns with Newey-West HAC standard errors.

    Parameters
    ----------
    returns : array of per-trade or per-period returns
    max_lags : Newey-West truncation lag. None = auto (Andrews rule).
    alpha : significance level for confidence intervals

    Returns
    -------
    TTestResult with t-stat, p-value, and confidence intervals.

    Interpretation
    --------------
    - p < 0.01: Strong evidence of non-zero mean return
    - p < 0.05: Moderate evidence (but be skeptical for strategies)
    - t > 3.0: Harvey et al. threshold for strategy discovery
    - CI not containing 0: mean return is statistically significant
    """
    import statsmodels.api as sm

    T = len(returns)
    if T < 30:
        logger.warning(
            "Only %d observations. T-test unreliable with < 30 samples.", T
        )

    # Optimal lag selection: Andrews (1991) rule
    if max_lags is None:
        max_lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
        max_lags = max(1, max_lags)

    # Regress returns on a constant to get Newey-West adjusted inference
    X = np.ones(T)
    model = sm.OLS(returns, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": max_lags},
    )

    mean_ret = model.params[0]
    nw_stderr = model.bse[0]
    t_stat = model.tvalues[0]
    p_val = model.pvalues[0]

    z = norm.ppf(1 - alpha / 2)
    ci_lower = mean_ret - z * nw_stderr
    ci_upper = mean_ret + z * nw_stderr

    return TTestResult(
        t_statistic=t_stat,
        p_value=p_val,
        mean_return=mean_ret,
        std_error=nw_stderr,
        n_observations=T,
        newey_west_lags=max_lags,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        significant_at_005=p_val < 0.05,
        significant_at_001=p_val < 0.01,
    )


# =============================================================================
# 2. SHARPE RATIO CONFIDENCE INTERVALS (Lo 2002)
# =============================================================================
#
# FORMULA (Lo 2002):
#   SE(SR) = sqrt( (1 + 0.5 * SR^2 - gamma3 * SR + (gamma4 - 3)/4 * SR^2) / T )
#
#   where:
#     SR = annualized Sharpe ratio (or per-period, but be consistent)
#     gamma3 = skewness of returns
#     gamma4 = kurtosis of returns (excess kurtosis + 3)
#     T = number of observations
#
# SIMPLIFIED (IID case, gamma3=0, gamma4=3):
#   SE(SR) = sqrt( (1 + 0.5 * SR^2) / T )
#
# WITH AUTOCORRELATION (Lo 2002, eq. 11):
#   SE(SR_annual) = SE(SR_period) * sqrt(q) * eta(q)
#   where q = annualization factor (e.g., 252 for daily)
#   eta(q) = sqrt(q) * adjustment factor for serial correlation
#   The adjustment inflates SE when returns are positively autocorrelated.
#
#   For returns with autocorrelation rho at lag k:
#   eta(q) = sqrt( q + 2 * sum_{k=1}^{q-1} (q-k)*rho_k ) / sqrt(q)
#
# WHEN CAN YOU TRUST A SHARPE RATIO?
#   - SR < 0.5: Very hard to distinguish from noise unless T >> 1000
#   - SR = 1.0: Need ~400 observations for 95% CI to exclude 0
#   - SR = 2.0: Need ~100 observations for 95% CI to exclude 0
#   - Rule of thumb: T > (2/SR)^2 for significance at p < 0.05
#
# MINIMUM SAMPLE SIZE (to achieve SE < 0.5*SR, i.e., 95% CI excludes 0):
#   T_min ≈ (1.96 / SR)^2 * (1 + 0.5*SR^2)
#   SR=0.5 → T_min ≈ 16 (periods, not annualized!)
#   SR=1.0 → T_min ≈ 5 ... but this is PER-PERIOD Sharpe
#   For annualized SR=1.0 with daily data: need ~4 years
#   For annualized SR=2.0 with daily data: need ~1 year


@dataclass
class SharpeResult:
    sharpe_ratio: float
    standard_error: float  # Lo (2002) formula
    ci_lower: float
    ci_upper: float
    p_value: float  # H0: SR = 0
    n_observations: int
    skewness: float
    kurtosis: float
    annualization_factor: int
    autocorrelation_adjusted: bool


def sharpe_confidence_interval(
    returns: np.ndarray,
    annualization_factor: int = 1,
    alpha: float = 0.05,
    adjust_autocorrelation: bool = True,
    max_autocorr_lags: int = 6,
) -> SharpeResult:
    """
    Compute Sharpe ratio with Lo (2002) confidence intervals.

    Parameters
    ----------
    returns : array of per-period returns (not annualized)
    annualization_factor : periods per year (252 daily, 52 weekly,
                          105120 for 5-min crypto [365*24*12])
    alpha : significance level
    adjust_autocorrelation : if True, inflate SE for serial correlation
    max_autocorr_lags : lags to include in autocorrelation adjustment

    Returns
    -------
    SharpeResult with Sharpe, SE, CI, and p-value.

    Interpretation
    --------------
    - If CI excludes 0: Sharpe is statistically significant
    - SE > |SR|: very uncertain, need more data
    - p < 0.05 with autocorrelation adjustment: trustworthy signal
    """
    T = len(returns)
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)

    if sigma == 0:
        logger.warning("Zero standard deviation in returns.")
        sr_per_period = 0.0
    else:
        sr_per_period = mu / sigma

    gamma3 = float(stats.skew(returns))
    gamma4 = float(stats.kurtosis(returns, fisher=False))  # non-excess

    # Lo (2002) standard error with skewness/kurtosis correction
    se_sr = np.sqrt(
        (1 + 0.5 * sr_per_period**2
         - gamma3 * sr_per_period
         + (gamma4 - 3) / 4 * sr_per_period**2)
        / T
    )

    # Autocorrelation adjustment (Lo 2002, Section III)
    if adjust_autocorrelation and T > max_autocorr_lags + 1:
        rho = np.array([
            np.corrcoef(returns[k:], returns[:-k])[0, 1]
            if k > 0 else 1.0
            for k in range(1, max_autocorr_lags + 1)
        ])
        # Replace NaN correlations with 0
        rho = np.nan_to_num(rho, nan=0.0)

        q = annualization_factor
        if q > 1:
            # eta(q) correction factor
            correction = 0.0
            for k in range(1, min(max_autocorr_lags + 1, q)):
                correction += (q - k) * rho[k - 1]
            eta_sq = (q + 2 * correction) / q
            eta_sq = max(eta_sq, 1.0)  # Cannot reduce SE below IID case
            se_sr = se_sr * np.sqrt(eta_sq)

    # Annualize
    sr_annual = sr_per_period * np.sqrt(annualization_factor)
    se_annual = se_sr * np.sqrt(annualization_factor)

    z = norm.ppf(1 - alpha / 2)
    ci_lower = sr_annual - z * se_annual
    ci_upper = sr_annual + z * se_annual

    # p-value for H0: SR = 0
    z_stat = sr_annual / se_annual if se_annual > 0 else 0.0
    p_val = 2 * (1 - norm.cdf(abs(z_stat)))

    return SharpeResult(
        sharpe_ratio=sr_annual,
        standard_error=se_annual,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value=p_val,
        n_observations=T,
        skewness=gamma3,
        kurtosis=gamma4,
        annualization_factor=annualization_factor,
        autocorrelation_adjusted=adjust_autocorrelation,
    )


# =============================================================================
# 3. DEFLATED SHARPE RATIO (Bailey & Lopez de Prado, 2014)
# =============================================================================
#
# PROBLEM: You tried N strategies and picked the best one. The observed Sharpe
# is inflated by selection bias. DSR asks: "What is the probability that the
# best strategy's true Sharpe exceeds what we'd expect from N random trials?"
#
# FORMULA:
#   Step 1 - Expected Maximum Sharpe Ratio under null (all strategies have SR=0):
#     E[max(SR)] ≈ mean(SR) + std(SR) * [
#         (1-gamma) * Phi^{-1}(1 - 1/N) + gamma * Phi^{-1}(1 - 1/(N*e))
#     ]
#     where gamma = 0.5772... (Euler-Mascheroni constant)
#           Phi^{-1} = inverse normal CDF
#           N = number of strategies tested
#
#   Step 2 - DSR (probability that true SR exceeds SR0):
#     DSR = Phi[
#         (SR_observed - SR0) * sqrt(T-1)
#         / sqrt(1 - gamma3*SR + (gamma4-1)/4 * SR^2)
#     ]
#     where SR0 = E[max(SR)] from Step 1
#           gamma3 = skewness of the selected strategy's returns
#           gamma4 = kurtosis of the selected strategy's returns
#           T = number of observations
#
# INTERPRETATION:
#   - DSR > 0.95: Strategy likely has genuine edge (analogous to p < 0.05)
#   - DSR < 0.50: Strategy's Sharpe is fully explained by selection bias
#   - DSR decreases as N increases (more strategies tested = higher bar)
#
# WHEN DOES DSR REJECT SEEMINGLY GOOD STRATEGIES?
#   - High N (tried many strategies): SR=2.0 can be rejected with N=1000
#   - High kurtosis (fat tails): inflates denominator, reduces DSR
#   - Negative skew (common in trading): reduces DSR
#   - Short track record (small T): reduces DSR
#
# EXAMPLE: SR=1.5, N=100 trials, T=500 obs, skew=-0.3, kurt=5 → DSR ≈ 0.72
#          Not significant! The 1.5 Sharpe is explainable by trying 100 things.


EULER_MASCHERONI = 0.5772156649015328606


@dataclass
class DeflatedSharpeResult:
    dsr: float  # Probability that true SR exceeds SR0
    observed_sharpe: float
    threshold_sharpe: float  # SR0: expected max under null
    n_trials: int
    n_observations: int
    skewness: float
    kurtosis: float
    is_significant: bool  # DSR > 0.95


def expected_max_sharpe(
    mean_sharpe: float,
    var_sharpe: float,
    n_trials: int,
) -> float:
    """
    Expected maximum Sharpe ratio from N independent trials under null.

    Uses the approximation from Bailey & Lopez de Prado (2014), eq. 6:
    E[max(SR)] ≈ SR_mean + SR_std * [(1-gamma)*Z_{1-1/N} + gamma*Z_{1-1/(Ne)}]

    Parameters
    ----------
    mean_sharpe : mean of all strategies' Sharpe ratios (use 0 under null)
    var_sharpe : variance of all strategies' Sharpe ratios
    n_trials : number of strategies tested

    Returns
    -------
    Expected maximum Sharpe ratio (SR0 threshold)
    """
    e = np.e
    gamma = EULER_MASCHERONI

    return mean_sharpe + np.sqrt(var_sharpe) * (
        (1 - gamma) * norm.ppf(1 - 1.0 / n_trials)
        + gamma * norm.ppf(1 - 1.0 / (n_trials * e))
    )


def deflated_sharpe_ratio(
    observed_sharpe: float,
    all_sharpes: np.ndarray,
    n_observations: int,
    skewness: float,
    kurtosis: float,
    n_trials: Optional[int] = None,
) -> DeflatedSharpeResult:
    """
    Compute the Deflated Sharpe Ratio.

    Parameters
    ----------
    observed_sharpe : Sharpe ratio of the selected (best) strategy.
                      MUST be non-annualized (per-period).
    all_sharpes : array of Sharpe ratios from ALL strategies tested.
                  MUST be non-annualized (per-period).
    n_observations : number of return observations (T)
    skewness : skewness of the selected strategy's returns
    kurtosis : kurtosis of the selected strategy's returns (non-excess)
    n_trials : number of strategies tested. If None, uses len(all_sharpes).

    Returns
    -------
    DeflatedSharpeResult

    Interpretation
    --------------
    - DSR > 0.95: genuine edge survives selection bias correction
    - DSR in [0.50, 0.95]: inconclusive, need more data or fewer trials
    - DSR < 0.50: observed SR is fully explained by multiple testing
    """
    if n_trials is None:
        n_trials = len(all_sharpes)

    mean_sr = float(np.mean(all_sharpes))
    var_sr = float(np.var(all_sharpes, ddof=1))

    sr0 = expected_max_sharpe(mean_sr, var_sr, max(n_trials, 2))

    # DSR formula (Bailey & Lopez de Prado 2014, eq. 15)
    numerator = (observed_sharpe - sr0) * np.sqrt(n_observations - 1)
    denominator = np.sqrt(
        1
        - skewness * observed_sharpe
        + (kurtosis - 1) / 4 * observed_sharpe**2
    )

    # Protect against negative denominator (extreme kurtosis/skew)
    if denominator <= 0:
        logger.warning(
            "DSR denominator non-positive (skew=%.2f, kurt=%.2f). "
            "Returns are too non-normal for DSR.",
            skewness,
            kurtosis,
        )
        dsr_value = 0.0
    else:
        dsr_value = float(norm.cdf(numerator / denominator))

    return DeflatedSharpeResult(
        dsr=dsr_value,
        observed_sharpe=observed_sharpe,
        threshold_sharpe=sr0,
        n_trials=n_trials,
        n_observations=n_observations,
        skewness=skewness,
        kurtosis=kurtosis,
        is_significant=dsr_value > 0.95,
    )


# =============================================================================
# 4. PROBABILITY OF BACKTEST OVERFITTING (PBO) via CSCV
# =============================================================================
#
# PROBLEM: You optimized strategy parameters on historical data. PBO estimates
# the probability that the "optimal" in-sample configuration underperforms
# the median out-of-sample.
#
# ALGORITHM: Combinatorially Symmetric Cross-Validation (CSCV)
#
#   Input: Matrix M of shape (T, N) where:
#     T = number of time periods
#     N = number of strategy configurations (parameter combinations)
#     M[t, n] = return of configuration n at time t
#
#   Step 1: Partition T rows into S contiguous blocks of size T/S each.
#           S must be even. Typical: S = 8, 10, 16.
#
#   Step 2: Generate all C(S, S/2) combinations of S/2 blocks.
#           For S=8: C(8,4) = 70 combinations.
#           For S=16: C(16,8) = 12,870 combinations.
#
#   Step 3: For each combination c:
#     a) IS = union of selected S/2 blocks (in-sample)
#     b) OOS = union of remaining S/2 blocks (out-of-sample)
#     c) Compute performance metric (e.g., Sharpe) for each of N configs
#        on IS data → rank them → find best config n*
#     d) Compute performance metric for n* on OOS data
#     e) Compute rank of n* in OOS (relative to all N configs on OOS)
#     f) lambda_c = rank(n*, OOS) / N  (normalized: 0=worst, 1=best)
#        If lambda_c < 0.5, the IS-best config is below median OOS.
#
#   Step 4: PBO = (# combinations where lambda_c < 0.5) / total combinations
#
# INTERPRETATION:
#   - PBO < 0.10: Low overfitting risk. Strategy likely has real edge.
#   - PBO 0.10-0.30: Moderate risk. Proceed with caution, use robust params.
#   - PBO 0.30-0.50: High risk. Strategy may be overfit.
#   - PBO > 0.50: Likely overfit. IS-best is more likely to underperform OOS.
#   - PBO > 0.80: Almost certainly overfit. Abandon or fundamentally redesign.
#
# HOW MANY STRATEGY VARIANTS TRIGGER CONCERN?
#   - N < 10: PBO analysis has low power (too few configs to rank)
#   - N = 50-200: Sweet spot for meaningful PBO analysis
#   - N > 1000: Almost guaranteed high PBO (too many degrees of freedom)
#   - The more parameters you tuned, the more N grows combinatorially
#   - Example: 5 params with 10 values each → N = 100,000 → PBO will be high


@dataclass
class PBOResult:
    pbo: float  # Probability of backtest overfitting
    n_configurations: int
    n_partitions: int  # S
    n_combinations: int  # C(S, S/2)
    lambdas: np.ndarray  # All lambda_c values
    mean_lambda: float
    median_lambda: float
    is_overfit: bool  # PBO > 0.50


def compute_pbo(
    returns_matrix: np.ndarray,
    n_partitions: int = 8,
    metric_func: Optional[object] = None,
) -> PBOResult:
    """
    Compute Probability of Backtest Overfitting via CSCV.

    Parameters
    ----------
    returns_matrix : array of shape (T, N) where T=time periods, N=strategy configs.
                     Each column is one configuration's return series.
    n_partitions : S, number of blocks to partition time axis. Must be even. 8-16 typical.
    metric_func : callable(returns_array) -> float. Performance metric.
                  Default: Sharpe ratio (mean/std).

    Returns
    -------
    PBOResult

    Minimum requirements
    --------------------
    - N >= 10 strategy configurations for meaningful analysis
    - T >= 200 time periods (for S=8, each block needs >= 25 periods)
    - Ideally T >= 500 and N in [50, 500]
    """
    T, N = returns_matrix.shape

    if n_partitions % 2 != 0:
        raise ValueError("n_partitions (S) must be even.")

    if T < n_partitions * 10:
        logger.warning(
            "T=%d with S=%d gives blocks of %d periods. "
            "Recommend T >= %d for reliable PBO.",
            T, n_partitions, T // n_partitions, n_partitions * 25,
        )

    if metric_func is None:
        def metric_func(r: np.ndarray) -> float:
            """Default: Sharpe ratio."""
            if len(r) < 2 or np.std(r) == 0:
                return 0.0
            return float(np.mean(r) / np.std(r, ddof=1))

    # Step 1: Partition into S blocks
    block_size = T // n_partitions
    # Trim excess rows to make even partitioning
    trimmed_T = block_size * n_partitions
    M = returns_matrix[:trimmed_T, :]

    blocks = []
    for s in range(n_partitions):
        blocks.append(M[s * block_size : (s + 1) * block_size, :])

    # Step 2: Generate all C(S, S/2) combinations
    half = n_partitions // 2
    all_combos = list(combinations(range(n_partitions), half))
    n_combos = len(all_combos)

    logger.info(
        "PBO: S=%d, C(%d,%d)=%d combinations, N=%d configs, block_size=%d",
        n_partitions, n_partitions, half, n_combos, N, block_size,
    )

    lambdas = np.zeros(n_combos)

    # Step 3-4: For each combination, compute IS-best rank in OOS
    for idx, is_indices in enumerate(all_combos):
        oos_indices = tuple(s for s in range(n_partitions) if s not in is_indices)

        # Assemble IS and OOS data
        is_data = np.vstack([blocks[s] for s in is_indices])
        oos_data = np.vstack([blocks[s] for s in oos_indices])

        # Compute metric for each config on IS
        is_metrics = np.array([metric_func(is_data[:, n]) for n in range(N)])

        # Find best config in-sample
        best_config = int(np.argmax(is_metrics))

        # Compute metric for each config on OOS
        oos_metrics = np.array([metric_func(oos_data[:, n]) for n in range(N)])

        # Rank of IS-best in OOS (0 = worst, N-1 = best)
        oos_rank = int(np.sum(oos_metrics <= oos_metrics[best_config]))
        # Normalize to [0, 1] where 1 = best
        lambdas[idx] = oos_rank / N

    # Step 5: PBO = fraction where IS-best is below median OOS
    pbo = float(np.mean(lambdas < 0.5))

    return PBOResult(
        pbo=pbo,
        n_configurations=N,
        n_partitions=n_partitions,
        n_combinations=n_combos,
        lambdas=lambdas,
        mean_lambda=float(np.mean(lambdas)),
        median_lambda=float(np.median(lambdas)),
        is_overfit=pbo > 0.50,
    )


# =============================================================================
# 5. WHITE'S REALITY CHECK
# =============================================================================
#
# PROBLEM: You tested K strategies against a benchmark. The best one beat it.
# But if K is large, the best will beat the benchmark by chance.
#
# NULL HYPOTHESIS: No strategy is superior to the benchmark.
#   H0: max_k E[d_k] <= 0
#   where d_k,t = loss(benchmark,t) - loss(strategy_k,t)
#
# ALGORITHM (Block Bootstrap):
#   1. Compute d_k,t for each strategy k and time t
#   2. Compute test statistic: V = max_k (sqrt(T) * mean(d_k))
#   3. Bootstrap:
#      a) Resample d_k,t using block bootstrap (preserves autocorrelation)
#      b) Center the resampled series (enforce null hypothesis)
#      c) Compute V* = max_k (sqrt(T) * mean(d_k*))
#      d) Repeat B times
#   4. p-value = fraction of V* >= V
#
# BLOCK BOOTSTRAP FOR TIME SERIES:
#   - Standard bootstrap (IID resampling) destroys autocorrelation structure
#   - Block bootstrap resamples contiguous blocks of length l
#   - Moving Block Bootstrap (MBB): blocks of fixed length l
#   - Circular Block Bootstrap (CBB): wraps around at boundaries
#   - Stationary Bootstrap (SB): random block lengths, geometric(1/l)
#   - Optimal block length: l ≈ T^(1/3) for variance estimation
#     For T=500: l ≈ 8; for T=1000: l ≈ 10; for T=5000: l ≈ 17
#
# IMPLEMENTATION NOTE: The `arch` library provides all of this.
# Use arch.bootstrap.SPA for the improved Hansen (2005) version.


@dataclass
class RealityCheckResult:
    test_statistic: float
    p_value: float
    n_strategies: int
    n_observations: int
    n_bootstrap: int
    block_length: int
    best_strategy_idx: int
    best_strategy_mean_excess: float
    rejects_null: bool  # True = at least one strategy beats benchmark


def whites_reality_check(
    benchmark_losses: np.ndarray,
    strategy_losses: np.ndarray,
    n_bootstrap: int = 1000,
    block_length: Optional[int] = None,
) -> RealityCheckResult:
    """
    White's Reality Check via block bootstrap.

    Parameters
    ----------
    benchmark_losses : array of shape (T,) - benchmark loss at each time
    strategy_losses : array of shape (T, K) - loss of each strategy at each time
    n_bootstrap : number of bootstrap replications
    block_length : block length for moving block bootstrap. None = auto.

    Returns
    -------
    RealityCheckResult

    Note
    ----
    For the improved version (Hansen SPA), use `hansens_spa_test()` below.
    White's RC is conservative (biased toward not rejecting) because it
    includes the maximum over ALL strategies, including bad ones.
    """
    T = len(benchmark_losses)
    K = strategy_losses.shape[1]

    if block_length is None:
        block_length = max(1, int(np.ceil(T ** (1 / 3))))

    # Excess performance: d_k,t = loss_benchmark - loss_strategy_k
    # Positive d means strategy is better than benchmark
    d = benchmark_losses[:, np.newaxis] - strategy_losses  # (T, K)

    # Test statistic: max over strategies of sqrt(T) * mean(d_k)
    mean_d = np.mean(d, axis=0)  # (K,)
    V_observed = np.sqrt(T) * np.max(mean_d)
    best_idx = int(np.argmax(mean_d))

    # Block bootstrap
    n_blocks = int(np.ceil(T / block_length))
    V_bootstrap = np.zeros(n_bootstrap)

    rng = np.random.default_rng(42)

    for b in range(n_bootstrap):
        # Resample block start indices
        starts = rng.integers(0, T - block_length + 1, size=n_blocks)
        indices = np.concatenate([
            np.arange(s, s + block_length) for s in starts
        ])[:T]

        # Centered resampled excess returns (enforce null)
        d_star = d[indices, :] - mean_d[np.newaxis, :]
        V_bootstrap[b] = np.sqrt(T) * np.max(np.mean(d_star, axis=0))

    p_value = float(np.mean(V_bootstrap >= V_observed))

    return RealityCheckResult(
        test_statistic=V_observed,
        p_value=p_value,
        n_strategies=K,
        n_observations=T,
        n_bootstrap=n_bootstrap,
        block_length=block_length,
        best_strategy_idx=best_idx,
        best_strategy_mean_excess=float(mean_d[best_idx]),
        rejects_null=p_value < 0.05,
    )


# =============================================================================
# 6. HANSEN'S SPA TEST (Superior Predictive Ability)
# =============================================================================
#
# IMPROVEMENT OVER WHITE'S RC:
#   White's RC is conservative because it takes the maximum over ALL strategies,
#   including ones that are clearly bad. Hansen (2005) improves by:
#   1. Studentizing the test statistic (dividing by strategy-specific std)
#   2. Only including strategies with positive mean in the bootstrap null
#   3. Providing three p-values: lower, consistent, upper
#
# The `arch` library implements this as `arch.bootstrap.SPA`.
#
# USAGE:
#   from arch.bootstrap import SPA
#   spa = SPA(benchmark_losses, strategy_losses, reps=1000)
#   spa.compute()
#   print(spa.pvalues)
#   # Returns: lower_pval, consistent_pval, upper_pval
#   # Use consistent_pval for decisions
#
# INTERPRETATION:
#   - consistent_pval < 0.05: reject null → at least one strategy is superior
#   - consistent_pval > 0.10: fail to reject → no evidence of superiority
#   - lower_pval: most liberal (easiest to reject)
#   - upper_pval: most conservative
#
# PARAMETERS:
#   - bootstrap: 'stationary' (default), 'circular', 'moving block'
#   - reps: bootstrap replications (1000 default, use 5000+ for publication)
#   - nested: True for studentized version (slower but more powerful)


def hansens_spa_test(
    benchmark_losses: np.ndarray,
    strategy_losses: np.ndarray,
    reps: int = 1000,
    bootstrap_method: str = "stationary",
) -> dict:
    """
    Hansen's SPA test using the arch library.

    Parameters
    ----------
    benchmark_losses : array of shape (T,) — loss series for benchmark
    strategy_losses : array of shape (T, K) — loss series for K strategies
    reps : number of bootstrap replications
    bootstrap_method : 'stationary', 'circular', or 'moving block'

    Returns
    -------
    dict with keys:
        'lower_pvalue': most liberal p-value
        'consistent_pvalue': recommended p-value
        'upper_pvalue': most conservative p-value
        'rejects_null': bool (using consistent_pvalue < 0.05)

    Note
    ----
    Requires: pip install arch
    Loss convention: LOWER loss is BETTER. If using returns, negate them.
    """
    from arch.bootstrap import SPA

    spa = SPA(
        benchmark_losses,
        strategy_losses,
        reps=reps,
        bootstrap=bootstrap_method,
    )
    spa.compute()

    pvals = spa.pvalues
    return {
        "lower_pvalue": float(pvals.iloc[0]),
        "consistent_pvalue": float(pvals.iloc[1]),
        "upper_pvalue": float(pvals.iloc[2]),
        "rejects_null": float(pvals.iloc[1]) < 0.05,
    }


# =============================================================================
# 7. PERMUTATION TEST (Shuffle Timing)
# =============================================================================
#
# IDEA: If your strategy has timing skill, the ORDER of trades matters.
# Shuffling trade timing should destroy performance. If shuffled equity curves
# look similar to the real one, your strategy has no timing edge.
#
# ALGORITHM:
#   1. Record per-trade returns: [r1, r2, ..., rN]
#   2. For b = 1 to B permutations:
#      a) Shuffle the return series randomly
#      b) Compute equity curve and performance metric on shuffled series
#   3. p-value = fraction of shuffled metrics >= observed metric
#
# WHY THIS WORKS:
#   - Shuffling preserves: mean return, volatility, distribution shape
#   - Shuffling destroys: autocorrelation, timing, serial dependence
#   - If the strategy profits from WHEN it enters (timing), shuffling hurts
#   - If the strategy profits from market bias (drift), shuffling doesn't hurt
#
# INTERPRETATION:
#   - p < 0.05: Timing matters. Strategy has genuine entry/exit skill.
#   - p > 0.10: No timing skill. Returns are due to drift or luck.
#   - Check multiple metrics: Sharpe, max drawdown, longest winning streak
#
# MINIMUM SAMPLE: N >= 100 trades, B >= 1000 permutations


@dataclass
class PermutationResult:
    observed_metric: float
    permuted_mean: float
    permuted_std: float
    p_value: float
    n_permutations: int
    n_trades: int
    has_timing_skill: bool  # p < 0.05
    percentile: float  # where observed falls in permuted distribution


def permutation_test(
    trade_returns: np.ndarray,
    metric_func: Optional[object] = None,
    n_permutations: int = 5000,
    seed: int = 42,
) -> PermutationResult:
    """
    Permutation test: does trade ordering/timing matter?

    Parameters
    ----------
    trade_returns : array of per-trade returns in chronological order
    metric_func : callable(returns) -> float. Default: Sharpe ratio.
    n_permutations : number of random shuffles
    seed : random seed

    Returns
    -------
    PermutationResult

    Interpretation
    --------------
    - p < 0.05: Your strategy has genuine timing skill
    - p > 0.10: Performance is explained by return distribution alone
                (no timing edge, just market drift or luck)
    """
    if metric_func is None:
        def metric_func(r: np.ndarray) -> float:
            if len(r) < 2 or np.std(r) == 0:
                return 0.0
            return float(np.mean(r) / np.std(r, ddof=1))

    observed = metric_func(trade_returns)
    rng = np.random.default_rng(seed)

    permuted_metrics = np.zeros(n_permutations)
    for b in range(n_permutations):
        shuffled = rng.permutation(trade_returns)
        permuted_metrics[b] = metric_func(shuffled)

    p_value = float(np.mean(permuted_metrics >= observed))
    percentile = float(np.mean(permuted_metrics <= observed) * 100)

    return PermutationResult(
        observed_metric=observed,
        permuted_mean=float(np.mean(permuted_metrics)),
        permuted_std=float(np.std(permuted_metrics)),
        p_value=p_value,
        n_permutations=n_permutations,
        n_trades=len(trade_returns),
        has_timing_skill=p_value < 0.05,
        percentile=percentile,
    )


# =============================================================================
# 8. BINOMIAL TEST ON WIN RATE
# =============================================================================
#
# QUESTION: Given N trades with W wins, is this better than random?
#
# NULL HYPOTHESIS: Win rate = p0 (default: 0.50 for fair coin)
#
# FORMULA:
#   P(X >= W | N, p0) = sum_{k=W}^{N} C(N,k) * p0^k * (1-p0)^(N-k)
#   This is 1 - CDF of Binomial(N, p0) evaluated at W-1.
#
# ADJUSTING FOR TRANSACTION COSTS:
#   If your average win = $A and average loss = $L:
#   Break-even win rate = L / (A + L)
#   Example: avg_win=50bps, avg_loss=40bps → break-even = 40/(50+40) = 44.4%
#   Use this as p0 instead of 0.50.
#
#   If costs are C bps per trade:
#   Adjusted avg_win = A - C
#   Adjusted avg_loss = L + C
#   Break-even = (L+C) / ((A-C) + (L+C))
#
# MINIMUM SAMPLE:
#   To detect a win rate of p vs p0 at power 0.80, alpha 0.05:
#   N ≈ [Z_{alpha} * sqrt(p0*(1-p0)) + Z_{beta} * sqrt(p*(1-p))]^2 / (p-p0)^2
#   p=0.55, p0=0.50 → N ≈ 784
#   p=0.52, p0=0.50 → N ≈ 4900
#   p=0.60, p0=0.50 → N ≈ 200


@dataclass
class BinomialTestResult:
    n_trades: int
    n_wins: int
    observed_win_rate: float
    null_win_rate: float  # p0
    p_value: float
    ci_lower: float  # Wilson CI
    ci_upper: float
    significant: bool
    cost_adjusted_null: Optional[float]


def binomial_test_win_rate(
    n_wins: int,
    n_trades: int,
    null_win_rate: float = 0.50,
    avg_win: Optional[float] = None,
    avg_loss: Optional[float] = None,
    cost_per_trade: float = 0.0,
    alpha: float = 0.05,
) -> BinomialTestResult:
    """
    Binomial test on win rate, optionally adjusted for transaction costs.

    Parameters
    ----------
    n_wins : number of winning trades
    n_trades : total number of trades
    null_win_rate : null hypothesis win rate (0.50 = fair coin)
    avg_win : average winning trade return (in same units as avg_loss)
    avg_loss : average losing trade return (positive number)
    cost_per_trade : transaction cost per trade (in same units)
    alpha : significance level

    Returns
    -------
    BinomialTestResult

    Usage: cost-adjusted null
    -------------------------
    If avg_win=50bps, avg_loss=40bps, cost=8bps:
    break_even = (40+8) / ((50-8) + (40+8)) = 48/90 = 53.3%
    Your win rate must beat 53.3% to be profitable, so use p0=0.533.
    """
    # Cost-adjusted null hypothesis
    cost_adjusted_null = None
    if avg_win is not None and avg_loss is not None:
        adj_win = avg_win - cost_per_trade
        adj_loss = avg_loss + cost_per_trade
        if adj_win + adj_loss > 0:
            cost_adjusted_null = adj_loss / (adj_win + adj_loss)
            null_win_rate = cost_adjusted_null

    observed_wr = n_wins / n_trades

    # Exact binomial test (one-sided: is win rate > null?)
    result = binomtest(n_wins, n_trades, null_win_rate, alternative="greater")
    p_value = result.pvalue

    # Wilson confidence interval (better than Wald for proportions)
    ci = result.proportion_ci(confidence_level=1 - alpha, method="wilson")

    return BinomialTestResult(
        n_trades=n_trades,
        n_wins=n_wins,
        observed_win_rate=observed_wr,
        null_win_rate=null_win_rate,
        p_value=p_value,
        ci_lower=ci.low,
        ci_upper=ci.high,
        significant=p_value < alpha,
        cost_adjusted_null=cost_adjusted_null,
    )


# =============================================================================
# 9. STRATEGY ROBUSTNESS TESTS
# =============================================================================
#
# These tests check if performance is fragile or robust.
#
# 9a. PARAMETER SENSITIVITY:
#   Perturb each parameter by +/- 10%, 20%, 50%.
#   If Sharpe drops by more than 50% with a 10% perturbation → fragile.
#   A robust strategy has a "performance plateau" in parameter space.
#
# 9b. DATA PERTURBATION:
#   Add Gaussian noise to features: X' = X + N(0, sigma*std(X))
#   sigma = 0.01, 0.05, 0.10
#   If Sharpe drops > 30% with sigma=0.05 → overfitting to noise patterns.
#
# 9c. WALK-FORWARD STABILITY:
#   Run walk-forward with K folds. Compute Sharpe for each fold.
#   Coefficient of variation: CV = std(Sharpes) / mean(Sharpes)
#   CV < 0.5: reasonably stable. CV > 1.0: unstable.
#   Also check: what fraction of folds have Sharpe > 0?
#
# 9d. REGIME STABILITY:
#   Compute performance in each market regime separately.
#   A good strategy: positive Sharpe in at least 2/3 of regimes.
#   A dangerous strategy: huge Sharpe in one regime, negative in others.


@dataclass
class ParameterSensitivityResult:
    base_metric: float
    perturbations: dict  # {param_name: {delta: metric_value, ...}}
    max_degradation: float  # worst-case % drop
    mean_degradation: float
    is_robust: bool  # max degradation < 50%


@dataclass
class WalkForwardStabilityResult:
    fold_sharpes: np.ndarray
    mean_sharpe: float
    std_sharpe: float
    cv: float  # coefficient of variation
    pct_positive: float  # fraction of folds with Sharpe > 0
    min_sharpe: float
    max_sharpe: float
    is_stable: bool  # CV < 0.5 and pct_positive > 0.6


def walk_forward_stability(
    fold_sharpes: np.ndarray,
) -> WalkForwardStabilityResult:
    """
    Assess consistency of Sharpe ratio across walk-forward folds.

    Parameters
    ----------
    fold_sharpes : array of Sharpe ratios from K walk-forward folds

    Returns
    -------
    WalkForwardStabilityResult

    Interpretation
    --------------
    - CV < 0.3: Very stable across time periods
    - CV 0.3-0.5: Reasonably stable
    - CV 0.5-1.0: Concerning variability
    - CV > 1.0: Unstable; edge may be ephemeral or regime-dependent
    - pct_positive > 0.8: Strong consistency
    - pct_positive < 0.6: Strategy may only work in certain conditions
    """
    mean_sr = float(np.mean(fold_sharpes))
    std_sr = float(np.std(fold_sharpes, ddof=1))
    cv = abs(std_sr / mean_sr) if mean_sr != 0 else float("inf")
    pct_pos = float(np.mean(fold_sharpes > 0))

    return WalkForwardStabilityResult(
        fold_sharpes=fold_sharpes,
        mean_sharpe=mean_sr,
        std_sharpe=std_sr,
        cv=cv,
        pct_positive=pct_pos,
        min_sharpe=float(np.min(fold_sharpes)),
        max_sharpe=float(np.max(fold_sharpes)),
        is_stable=cv < 0.5 and pct_pos > 0.6,
    )


def data_perturbation_test(
    features: np.ndarray,
    returns: np.ndarray,
    model_predict_func: object,
    noise_levels: tuple = (0.01, 0.05, 0.10),
    metric_func: Optional[object] = None,
    n_trials: int = 20,
    seed: int = 42,
) -> dict:
    """
    Test strategy robustness by adding noise to features.

    Parameters
    ----------
    features : array of shape (T, F), feature matrix
    returns : array of shape (T,), actual returns
    model_predict_func : callable(features) -> predictions (signal array)
    noise_levels : standard deviations of noise as fraction of feature std
    metric_func : callable(actual_returns, predictions) -> float
    n_trials : number of noisy trials per noise level
    seed : random seed

    Returns
    -------
    dict: {noise_level: {'mean_metric': float, 'degradation_pct': float}}

    Interpretation
    --------------
    - Degradation < 10% at noise=0.05: Very robust
    - Degradation 10-30% at noise=0.05: Acceptable
    - Degradation > 30% at noise=0.05: Overfitting to noise patterns
    """
    if metric_func is None:
        def metric_func(actual: np.ndarray, preds: np.ndarray) -> float:
            strat_returns = actual * np.sign(preds)
            if np.std(strat_returns) == 0:
                return 0.0
            return float(np.mean(strat_returns) / np.std(strat_returns, ddof=1))

    rng = np.random.default_rng(seed)

    # Baseline metric (no noise)
    base_preds = model_predict_func(features)
    base_metric = metric_func(returns, base_preds)

    results = {}
    feature_stds = np.std(features, axis=0, keepdims=True)
    feature_stds = np.where(feature_stds == 0, 1.0, feature_stds)  # avoid div by 0

    for sigma in noise_levels:
        trial_metrics = []
        for _ in range(n_trials):
            noise = rng.normal(0, sigma, size=features.shape) * feature_stds
            noisy_features = features + noise
            noisy_preds = model_predict_func(noisy_features)
            trial_metrics.append(metric_func(returns, noisy_preds))

        mean_metric = float(np.mean(trial_metrics))
        degradation = (
            (base_metric - mean_metric) / abs(base_metric) * 100
            if base_metric != 0
            else 0.0
        )
        results[sigma] = {
            "mean_metric": mean_metric,
            "std_metric": float(np.std(trial_metrics)),
            "degradation_pct": degradation,
        }

    return {"base_metric": base_metric, "noise_results": results}


# =============================================================================
# 10. MULTIPLE HYPOTHESIS CORRECTION
# =============================================================================
#
# When you test N hypotheses, the probability of at least one false positive is:
#   P(at least one false positive) = 1 - (1 - alpha)^N ≈ N * alpha for small alpha
#   With N=20 and alpha=0.05: P ≈ 64%!
#
# METHODS (from most conservative to least):
#
# 1. BONFERRONI:
#   Adjusted alpha = alpha / N
#   Reject H_i if p_i < alpha/N
#   Controls FWER (family-wise error rate)
#   Very conservative. Use when false positives are catastrophic.
#   For N=100, alpha=0.05: threshold = 0.0005
#
# 2. HOLM-BONFERRONI (step-down):
#   Sort p-values: p_(1) <= p_(2) <= ... <= p_(N)
#   Reject H_(i) if p_(i) < alpha / (N - i + 1)
#   Uniformly more powerful than Bonferroni (always use Holm over Bonferroni)
#   Still controls FWER.
#
# 3. BENJAMINI-HOCHBERG (FDR):
#   Sort p-values: p_(1) <= p_(2) <= ... <= p_(N)
#   Find largest k where p_(k) <= k/N * alpha
#   Reject H_(1), ..., H_(k)
#   Controls FDR (expected fraction of false discoveries among rejections)
#   More powerful than FWER methods. Best for EXPLORATORY analysis.
#
# WHICH IS BEST FOR STRATEGY TESTING?
#   - Bonferroni: Too conservative. You'll reject everything.
#   - Holm-Bonferroni: Good for a small number of strategies (N < 20)
#   - Benjamini-Hochberg: BEST for strategy testing with many candidates
#     Rationale: You accept some false positives in exchange for more discoveries.
#     A 5% FDR means ~1 in 20 "significant" strategies is actually noise.
#
# HOW TO COUNT HYPOTHESES:
#   N = total number of distinct strategies evaluated, INCLUDING:
#   - Different models (XGBoost, GRU, ensemble variants)
#   - Different feature sets tried
#   - Different hyperparameter configurations (if hand-selected)
#   - Different time horizons (1-min, 5-min, 15-min)
#   - Different stop-loss/take-profit levels
#   EXCLUDING (if properly cross-validated):
#   - Automated Optuna trials (they're internal optimization, not hypotheses)
#   - Walk-forward folds (they're validation, not separate hypotheses)
#   Honest count: "How many strategies did I look at and could have selected?"
#   When in doubt, OVERCOUNT. N=50-200 is typical for a research project.


@dataclass
class MultipleTestResult:
    method: str
    n_hypotheses: int
    original_alpha: float
    adjusted_pvalues: np.ndarray
    rejected: np.ndarray  # boolean mask
    n_rejected: int


def bonferroni_correction(p_values: np.ndarray, alpha: float = 0.05) -> MultipleTestResult:
    """Bonferroni correction: adjusted_p = p * N, reject if < alpha."""
    N = len(p_values)
    adjusted = np.minimum(p_values * N, 1.0)
    rejected = adjusted < alpha
    return MultipleTestResult(
        method="bonferroni",
        n_hypotheses=N,
        original_alpha=alpha,
        adjusted_pvalues=adjusted,
        rejected=rejected,
        n_rejected=int(np.sum(rejected)),
    )


def holm_bonferroni_correction(p_values: np.ndarray, alpha: float = 0.05) -> MultipleTestResult:
    """Holm-Bonferroni step-down correction."""
    N = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    adjusted = np.zeros(N)
    rejected = np.zeros(N, dtype=bool)

    running_max = 0.0
    for i in range(N):
        adj_p = sorted_p[i] * (N - i)
        running_max = max(running_max, adj_p)
        adjusted[sorted_idx[i]] = min(running_max, 1.0)

    rejected = adjusted < alpha

    return MultipleTestResult(
        method="holm_bonferroni",
        n_hypotheses=N,
        original_alpha=alpha,
        adjusted_pvalues=adjusted,
        rejected=rejected,
        n_rejected=int(np.sum(rejected)),
    )


def benjamini_hochberg_correction(p_values: np.ndarray, alpha: float = 0.05) -> MultipleTestResult:
    """Benjamini-Hochberg FDR correction. RECOMMENDED for strategy testing."""
    N = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    adjusted = np.zeros(N)

    # Work backwards from largest p-value
    running_min = 1.0
    for i in range(N - 1, -1, -1):
        adj_p = sorted_p[i] * N / (i + 1)
        running_min = min(running_min, adj_p)
        adjusted[sorted_idx[i]] = min(running_min, 1.0)

    rejected = adjusted < alpha

    return MultipleTestResult(
        method="benjamini_hochberg",
        n_hypotheses=N,
        original_alpha=alpha,
        adjusted_pvalues=adjusted,
        rejected=rejected,
        n_rejected=int(np.sum(rejected)),
    )


# =============================================================================
# 11. BOOTSTRAP CONFIDENCE INTERVALS FOR ALL METRICS
# =============================================================================
#
# For any metric that doesn't have a clean analytical formula for SE,
# use bootstrap CIs. This covers: Sharpe, max drawdown, win rate, profit factor,
# Calmar ratio, etc.
#
# BLOCK BOOTSTRAP vs CIRCULAR BOOTSTRAP:
#   - Block bootstrap (MBB): resample contiguous blocks of length l
#     Blocks at the end of the series are truncated → edge bias
#   - Circular bootstrap (CBB): wrap the series: after observation T comes
#     observation 1. Eliminates edge bias. PREFERRED for most cases.
#   - Stationary bootstrap (SB): random block lengths from Geometric(1/l).
#     Most theoretically sound. Block length = expected length.
#
# OPTIMAL BLOCK LENGTH:
#   Use Politis & Romano (2004) or the arch library's `optimal_block_length()`.
#   Rule of thumb: l ≈ T^(1/3) for variance-type statistics.
#
# HOW MANY BOOTSTRAP SAMPLES:
#   - B = 1000: standard for exploratory analysis
#   - B = 5000: recommended for publication-quality CIs
#   - B = 10000: diminishing returns beyond this for most metrics
#   - For p-value precision to 0.01: need B >= 1/0.01 - 1 = 99 (minimum)
#   - For p-value precision to 0.001: need B >= 999
#
# CONFIDENCE INTERVAL TYPES:
#   - Percentile: [Q_{alpha/2}, Q_{1-alpha/2}] of bootstrap distribution
#     Simple but biased for small samples.
#   - Bias-corrected accelerated (BCa): corrects for bias and skewness.
#     RECOMMENDED. More accurate CI coverage.
#   - Basic/pivotal: 2*observed - Q_{1-alpha/2}, 2*observed - Q_{alpha/2}


@dataclass
class BootstrapCIResult:
    observed: float
    ci_lower: float
    ci_upper: float
    bootstrap_mean: float
    bootstrap_std: float
    n_bootstrap: int
    block_length: int
    ci_method: str  # 'percentile' or 'bca'


def _circular_block_resample(
    data: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Circular block bootstrap resample."""
    T = len(data)
    n_blocks = int(np.ceil(T / block_length))
    starts = rng.integers(0, T, size=n_blocks)

    indices = np.concatenate([
        np.arange(s, s + block_length) % T for s in starts
    ])[:T]

    return data[indices]


def bootstrap_confidence_interval(
    data: np.ndarray,
    metric_func: object,
    n_bootstrap: int = 5000,
    block_length: Optional[int] = None,
    alpha: float = 0.05,
    method: str = "percentile",
    seed: int = 42,
) -> BootstrapCIResult:
    """
    Bootstrap confidence interval for any metric using circular block bootstrap.

    Parameters
    ----------
    data : 1D array of returns or observations
    metric_func : callable(data) -> float. The statistic to bootstrap.
    n_bootstrap : number of bootstrap replications
    block_length : circular block length. None = auto (T^(1/3)).
    alpha : significance level (0.05 = 95% CI)
    method : 'percentile' or 'bca' (bias-corrected accelerated)
    seed : random seed

    Returns
    -------
    BootstrapCIResult

    Examples
    --------
    # Sharpe ratio CI
    sharpe_ci = bootstrap_confidence_interval(
        returns,
        metric_func=lambda r: np.mean(r)/np.std(r, ddof=1) if np.std(r)>0 else 0,
    )

    # Max drawdown CI
    def max_drawdown(r):
        cum = np.cumsum(r)
        running_max = np.maximum.accumulate(cum)
        dd = running_max - cum
        return np.max(dd) if len(dd) > 0 else 0.0

    dd_ci = bootstrap_confidence_interval(returns, metric_func=max_drawdown)

    # Profit factor CI
    def profit_factor(r):
        gains = np.sum(r[r > 0])
        losses = abs(np.sum(r[r < 0]))
        return gains / losses if losses > 0 else float('inf')

    pf_ci = bootstrap_confidence_interval(returns, metric_func=profit_factor)
    """
    T = len(data)
    if block_length is None:
        block_length = max(1, int(np.ceil(T ** (1 / 3))))

    observed = metric_func(data)
    rng = np.random.default_rng(seed)

    boot_stats = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        resampled = _circular_block_resample(data, block_length, rng)
        boot_stats[b] = metric_func(resampled)

    # Remove any inf/nan
    valid = np.isfinite(boot_stats)
    boot_stats_clean = boot_stats[valid]

    if len(boot_stats_clean) < n_bootstrap * 0.9:
        logger.warning(
            "%.1f%% of bootstrap samples produced non-finite metrics.",
            (1 - len(boot_stats_clean) / n_bootstrap) * 100,
        )

    if method == "bca" and len(boot_stats_clean) > 100:
        # Bias-corrected accelerated (BCa) CI
        # Bias correction factor
        z0 = norm.ppf(np.mean(boot_stats_clean < observed))

        # Acceleration factor via jackknife
        jackknife_stats = np.zeros(T)
        for i in range(T):
            jack_data = np.concatenate([data[:i], data[i + 1 :]])
            jackknife_stats[i] = metric_func(jack_data)
        jack_mean = np.mean(jackknife_stats)
        numerator = np.sum((jack_mean - jackknife_stats) ** 3)
        denominator = 6 * (np.sum((jack_mean - jackknife_stats) ** 2)) ** 1.5
        a = numerator / denominator if denominator != 0 else 0.0

        # Adjusted percentiles
        z_alpha = norm.ppf(alpha / 2)
        z_1alpha = norm.ppf(1 - alpha / 2)

        p_lower = norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
        p_upper = norm.cdf(z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha)))

        ci_lower = float(np.percentile(boot_stats_clean, p_lower * 100))
        ci_upper = float(np.percentile(boot_stats_clean, p_upper * 100))
    else:
        # Percentile method
        ci_lower = float(np.percentile(boot_stats_clean, alpha / 2 * 100))
        ci_upper = float(np.percentile(boot_stats_clean, (1 - alpha / 2) * 100))

    return BootstrapCIResult(
        observed=observed,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        bootstrap_mean=float(np.mean(boot_stats_clean)),
        bootstrap_std=float(np.std(boot_stats_clean)),
        n_bootstrap=n_bootstrap,
        block_length=block_length,
        ci_method=method,
    )


# =============================================================================
# 12. POWER ANALYSIS
# =============================================================================
#
# QUESTION: How many trades/days do you need to detect a real edge of X bps?
#
# SETUP:
#   H0: mean return = 0
#   H1: mean return = delta (the edge, in bps)
#   alpha = significance level (typically 0.05)
#   beta = Type II error rate (1-beta = power, typically 0.80)
#   sigma = standard deviation of per-trade returns
#
# FORMULA (two-sided test):
#   N = ((z_{alpha/2} + z_{beta}) * sigma / delta)^2
#
#   where z_{alpha/2} = 1.96, z_{beta} = 0.84 for alpha=0.05, power=0.80
#
# EXAMPLES FOR CRYPTO TRADING (sigma ≈ 30-50 bps per 5-min trade):
#
#   Edge = 5 bps, sigma = 40 bps:
#   N = ((1.96 + 0.84) * 40 / 5)^2 = (2.80 * 8)^2 = 22.4^2 ≈ 502 trades
#
#   Edge = 10 bps, sigma = 40 bps:
#   N = ((2.80) * 40 / 10)^2 = (11.2)^2 ≈ 126 trades
#
#   Edge = 2 bps, sigma = 40 bps:
#   N = ((2.80) * 40 / 2)^2 = (56)^2 ≈ 3136 trades
#
#   Edge = 15 bps (strong edge), sigma = 50 bps:
#   N = ((2.80) * 50 / 15)^2 = (9.33)^2 ≈ 87 trades
#
# AT 5-MIN FREQUENCY (288 candles/day, assuming ~30% signal rate ≈ 86 trades/day):
#   5 bps edge: 502 / 86 ≈ 6 days of trading
#   10 bps edge: 126 / 86 ≈ 1.5 days
#   2 bps edge: 3136 / 86 ≈ 36 days
#
# ADJUSTMENTS FOR REALISM:
#   - Autocorrelated returns: multiply N by eta^2 (Lo's correction, typically 1.2-2.0x)
#   - Non-normal returns: multiply N by 1.1-1.5x (kurtosis penalty)
#   - Multiple testing: use adjusted alpha (Bonferroni or BH), which increases N
#   - Regime changes: need samples from EACH regime, so multiply by n_regimes


@dataclass
class PowerAnalysisResult:
    required_trades: int
    required_days: float  # at given trade frequency
    edge_bps: float
    volatility_bps: float
    alpha: float
    power: float
    trades_per_day: float
    autocorrelation_multiplier: float


def power_analysis(
    edge_bps: float,
    volatility_bps: float,
    alpha: float = 0.05,
    power: float = 0.80,
    trades_per_day: float = 86.0,
    autocorrelation_multiplier: float = 1.0,
    two_sided: bool = True,
) -> PowerAnalysisResult:
    """
    How many trades needed to detect a real edge?

    Parameters
    ----------
    edge_bps : expected per-trade edge in basis points
    volatility_bps : standard deviation of per-trade returns in basis points
    alpha : significance level
    power : statistical power (1 - Type II error rate)
    trades_per_day : expected number of trades per day
    autocorrelation_multiplier : inflate N for autocorrelated returns (1.0-2.0)
    two_sided : True for two-sided test (more conservative)

    Returns
    -------
    PowerAnalysisResult

    Interpretation
    --------------
    - required_trades: minimum trades for statistical significance
    - required_days: how long you need to run the strategy
    - If required_days > 60: edge may be too small to validate in reasonable time
    - If required_days < 5: edge is large and should be detectable quickly

    Typical crypto parameters
    -------------------------
    5-min BTC: volatility ≈ 30-50 bps per trade
    Trade frequency: 20-100 trades per day depending on signal rate
    Realistic edge: 2-15 bps per trade after costs
    """
    if edge_bps <= 0:
        raise ValueError("edge_bps must be positive.")

    z_alpha = norm.ppf(1 - alpha / (2 if two_sided else 1))
    z_beta = norm.ppf(power)

    N = ((z_alpha + z_beta) * volatility_bps / edge_bps) ** 2
    N *= autocorrelation_multiplier
    N = int(np.ceil(N))

    days = N / trades_per_day if trades_per_day > 0 else float("inf")

    return PowerAnalysisResult(
        required_trades=N,
        required_days=days,
        edge_bps=edge_bps,
        volatility_bps=volatility_bps,
        alpha=alpha,
        power=power,
        trades_per_day=trades_per_day,
        autocorrelation_multiplier=autocorrelation_multiplier,
    )


# =============================================================================
# COMPREHENSIVE VALIDATION PIPELINE
# =============================================================================
# Run all tests together on a strategy's results.


@dataclass
class ValidationSummary:
    ttest: TTestResult
    sharpe_ci: SharpeResult
    dsr: Optional[DeflatedSharpeResult]
    pbo: Optional[PBOResult]
    permutation: PermutationResult
    binomial: BinomialTestResult
    wf_stability: Optional[WalkForwardStabilityResult]
    bootstrap_sharpe_ci: BootstrapCIResult
    power: PowerAnalysisResult
    verdict: str  # "GENUINE_EDGE", "INCONCLUSIVE", "LIKELY_NOISE"


def full_validation_pipeline(
    trade_returns: np.ndarray,
    n_wins: int,
    n_trades: int,
    all_strategy_sharpes: Optional[np.ndarray] = None,
    returns_matrix: Optional[np.ndarray] = None,
    fold_sharpes: Optional[np.ndarray] = None,
    avg_win_bps: float = 20.0,
    avg_loss_bps: float = 15.0,
    cost_per_trade_bps: float = 8.0,
    trades_per_day: float = 86.0,
    annualization_factor: int = 105120,
) -> ValidationSummary:
    """
    Run all statistical validation tests on a strategy.

    Parameters
    ----------
    trade_returns : per-trade returns (chronological order)
    n_wins : number of winning trades
    n_trades : total trades
    all_strategy_sharpes : Sharpe ratios of ALL strategies tested (for DSR)
    returns_matrix : (T, N) matrix of all strategy configs (for PBO)
    fold_sharpes : Sharpe ratios from walk-forward folds
    avg_win_bps / avg_loss_bps : average win/loss in bps
    cost_per_trade_bps : round-trip cost in bps
    trades_per_day : for power analysis
    annualization_factor : 105120 for 5-min crypto (365*24*12)

    Returns
    -------
    ValidationSummary with all test results and overall verdict.
    """
    # 1. T-test with Newey-West
    ttest = ttest_strategy_returns(trade_returns)

    # 2. Sharpe CI (Lo 2002)
    sharpe_ci = sharpe_confidence_interval(
        trade_returns,
        annualization_factor=annualization_factor,
    )

    # 3. DSR (if multiple strategies tested)
    dsr_result = None
    if all_strategy_sharpes is not None and len(all_strategy_sharpes) > 1:
        skew = float(stats.skew(trade_returns))
        kurt = float(stats.kurtosis(trade_returns, fisher=False))
        per_period_sr = (
            np.mean(trade_returns) / np.std(trade_returns, ddof=1)
            if np.std(trade_returns) > 0
            else 0.0
        )
        dsr_result = deflated_sharpe_ratio(
            observed_sharpe=per_period_sr,
            all_sharpes=all_strategy_sharpes,
            n_observations=len(trade_returns),
            skewness=skew,
            kurtosis=kurt,
        )

    # 4. PBO (if returns matrix available)
    pbo_result = None
    if returns_matrix is not None and returns_matrix.shape[1] >= 10:
        pbo_result = compute_pbo(returns_matrix)

    # 5. Permutation test
    perm = permutation_test(trade_returns)

    # 6. Binomial test
    binom = binomial_test_win_rate(
        n_wins=n_wins,
        n_trades=n_trades,
        avg_win=avg_win_bps,
        avg_loss=avg_loss_bps,
        cost_per_trade=cost_per_trade_bps,
    )

    # 7. Walk-forward stability
    wf_result = None
    if fold_sharpes is not None and len(fold_sharpes) >= 3:
        wf_result = walk_forward_stability(fold_sharpes)

    # 8. Bootstrap CI on Sharpe
    def sharpe_func(r: np.ndarray) -> float:
        if len(r) < 2 or np.std(r) == 0:
            return 0.0
        return float(np.mean(r) / np.std(r, ddof=1))

    boot_ci = bootstrap_confidence_interval(
        trade_returns,
        metric_func=sharpe_func,
        method="bca",
    )

    # 9. Power analysis
    vol_bps = float(np.std(trade_returns)) * 10000 if np.std(trade_returns) < 1 else float(np.std(trade_returns))
    edge_bps = float(np.mean(trade_returns)) * 10000 if abs(np.mean(trade_returns)) < 1 else float(np.mean(trade_returns))
    edge_bps = max(abs(edge_bps), 0.1)  # avoid zero

    power_result = power_analysis(
        edge_bps=edge_bps,
        volatility_bps=vol_bps,
        trades_per_day=trades_per_day,
    )

    # Verdict logic
    genuine_signals = 0
    total_tests = 0

    # T-test significant at 1%?
    total_tests += 1
    if ttest.significant_at_001:
        genuine_signals += 1

    # Sharpe CI excludes 0?
    total_tests += 1
    if sharpe_ci.ci_lower > 0:
        genuine_signals += 1

    # DSR significant?
    if dsr_result is not None:
        total_tests += 1
        if dsr_result.is_significant:
            genuine_signals += 1

    # PBO not overfit?
    if pbo_result is not None:
        total_tests += 1
        if not pbo_result.is_overfit:
            genuine_signals += 1

    # Permutation test shows timing skill?
    total_tests += 1
    if perm.has_timing_skill:
        genuine_signals += 1

    # Win rate significant?
    total_tests += 1
    if binom.significant:
        genuine_signals += 1

    # Walk-forward stable?
    if wf_result is not None:
        total_tests += 1
        if wf_result.is_stable:
            genuine_signals += 1

    # Bootstrap CI excludes 0?
    total_tests += 1
    if boot_ci.ci_lower > 0:
        genuine_signals += 1

    ratio = genuine_signals / total_tests if total_tests > 0 else 0

    if ratio >= 0.75:
        verdict = "GENUINE_EDGE"
    elif ratio >= 0.50:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "LIKELY_NOISE"

    return ValidationSummary(
        ttest=ttest,
        sharpe_ci=sharpe_ci,
        dsr=dsr_result,
        pbo=pbo_result,
        permutation=perm,
        binomial=binom,
        wf_stability=wf_result,
        bootstrap_sharpe_ci=boot_ci,
        power=power_result,
        verdict=verdict,
    )


def print_validation_report(summary: ValidationSummary) -> None:
    """Print a human-readable validation report."""
    print("=" * 72)
    print("STRATEGY STATISTICAL VALIDATION REPORT")
    print("=" * 72)

    print(f"\nOVERALL VERDICT: {summary.verdict}")
    print("-" * 72)

    # T-test
    t = summary.ttest
    print(f"\n1. T-TEST (Newey-West, {t.newey_west_lags} lags)")
    print(f"   Mean return: {t.mean_return:.6f}")
    print(f"   t-statistic: {t.t_statistic:.3f}")
    print(f"   p-value: {t.p_value:.4f}")
    print(f"   95% CI: [{t.ci_lower:.6f}, {t.ci_upper:.6f}]")
    print(f"   Significant at 1%: {t.significant_at_001}")

    # Sharpe
    s = summary.sharpe_ci
    print(f"\n2. SHARPE RATIO (Lo 2002)")
    print(f"   Sharpe: {s.sharpe_ratio:.3f}")
    print(f"   SE: {s.standard_error:.3f}")
    print(f"   95% CI: [{s.ci_lower:.3f}, {s.ci_upper:.3f}]")
    print(f"   p-value: {s.p_value:.4f}")
    print(f"   Skewness: {s.skewness:.3f}, Kurtosis: {s.kurtosis:.3f}")

    # DSR
    if summary.dsr is not None:
        d = summary.dsr
        print(f"\n3. DEFLATED SHARPE RATIO")
        print(f"   DSR: {d.dsr:.4f}")
        print(f"   Threshold SR (SR0): {d.threshold_sharpe:.4f}")
        print(f"   N trials: {d.n_trials}")
        print(f"   Significant (DSR > 0.95): {d.is_significant}")

    # PBO
    if summary.pbo is not None:
        p = summary.pbo
        print(f"\n4. PROBABILITY OF BACKTEST OVERFITTING")
        print(f"   PBO: {p.pbo:.3f}")
        print(f"   Mean lambda: {p.mean_lambda:.3f}")
        print(f"   Overfit (PBO > 0.50): {p.is_overfit}")
        print(f"   N configs: {p.n_configurations}, Combinations: {p.n_combinations}")

    # Permutation
    pm = summary.permutation
    print(f"\n5. PERMUTATION TEST")
    print(f"   Observed metric: {pm.observed_metric:.4f}")
    print(f"   Permuted mean: {pm.permuted_mean:.4f} +/- {pm.permuted_std:.4f}")
    print(f"   p-value: {pm.p_value:.4f}")
    print(f"   Timing skill: {pm.has_timing_skill}")
    print(f"   Percentile: {pm.percentile:.1f}%")

    # Binomial
    b = summary.binomial
    print(f"\n6. BINOMIAL TEST")
    print(f"   Win rate: {b.observed_win_rate:.3f} ({b.n_wins}/{b.n_trades})")
    print(f"   Null rate: {b.null_win_rate:.3f}")
    if b.cost_adjusted_null is not None:
        print(f"   Cost-adjusted null: {b.cost_adjusted_null:.3f}")
    print(f"   p-value: {b.p_value:.4f}")
    print(f"   95% CI: [{b.ci_lower:.3f}, {b.ci_upper:.3f}]")
    print(f"   Significant: {b.significant}")

    # Walk-forward
    if summary.wf_stability is not None:
        w = summary.wf_stability
        print(f"\n7. WALK-FORWARD STABILITY")
        print(f"   Mean Sharpe: {w.mean_sharpe:.3f} +/- {w.std_sharpe:.3f}")
        print(f"   CV: {w.cv:.3f}")
        print(f"   % positive folds: {w.pct_positive:.1%}")
        print(f"   Range: [{w.min_sharpe:.3f}, {w.max_sharpe:.3f}]")
        print(f"   Stable: {w.is_stable}")

    # Bootstrap
    bc = summary.bootstrap_sharpe_ci
    print(f"\n8. BOOTSTRAP SHARPE CI ({bc.ci_method})")
    print(f"   Observed: {bc.observed:.4f}")
    print(f"   95% CI: [{bc.ci_lower:.4f}, {bc.ci_upper:.4f}]")
    print(f"   Boot mean: {bc.bootstrap_mean:.4f} +/- {bc.bootstrap_std:.4f}")

    # Power
    pw = summary.power
    print(f"\n9. POWER ANALYSIS")
    print(f"   Edge: {pw.edge_bps:.1f} bps, Vol: {pw.volatility_bps:.1f} bps")
    print(f"   Required trades: {pw.required_trades}")
    print(f"   Required days: {pw.required_days:.1f}")
    print(f"   (at {pw.trades_per_day:.0f} trades/day)")

    print("\n" + "=" * 72)


# =============================================================================
# QUICK REFERENCE TABLE
# =============================================================================
#
# | Test                  | What it tests                      | Min samples | Key threshold          |
# |-----------------------|------------------------------------|-------------|------------------------|
# | t-test (NW)           | Mean return != 0                   | 100 trades  | p < 0.01, t > 3.0     |
# | Sharpe CI (Lo)        | SR significantly > 0               | 250 obs     | CI excludes 0          |
# | Deflated Sharpe       | SR survives multiple testing       | 500 obs     | DSR > 0.95             |
# | PBO (CSCV)            | Strategy not overfit               | 200 periods | PBO < 0.30             |
# | White's RC            | Best beats benchmark               | 500 obs     | p < 0.05               |
# | Hansen's SPA          | Improved White's RC                | 500 obs     | consistent p < 0.05    |
# | Permutation           | Timing/ordering matters            | 100 trades  | p < 0.05               |
# | Binomial              | Win rate > chance                  | 200 trades  | p < 0.05 (cost-adj)    |
# | WF stability          | Consistent across time             | 5 folds     | CV < 0.5, >60% pos     |
# | Parameter sensitivity | Robust to param changes            | N/A         | <50% degradation @10%  |
# | Data perturbation     | Robust to feature noise            | N/A         | <30% degradation @5%   |
# | Bootstrap CI          | Metric uncertainty                 | 100 obs     | CI excludes 0          |
# | Power analysis        | How many trades needed             | N/A         | Depends on edge size   |
# | Bonferroni            | FWER control (conservative)        | N/A         | Adjusted p < 0.05      |
# | Holm-Bonferroni       | FWER control (better)              | N/A         | Adjusted p < 0.05      |
# | Benjamini-Hochberg    | FDR control (RECOMMENDED)          | N/A         | Adjusted p < 0.05      |
#
# DECISION FRAMEWORK:
# 1. Start with t-test + Sharpe CI: basic sanity check
# 2. Run permutation test: does timing matter at all?
# 3. If multiple strategies tested: DSR + BH correction
# 4. If parameters were optimized: PBO
# 5. If comparing to benchmark: Hansen's SPA
# 6. Always: bootstrap CIs, walk-forward stability, power analysis
# 7. Final verdict: majority of tests must pass
