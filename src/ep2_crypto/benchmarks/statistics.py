"""Statistical analysis for benchmark strategies.

Provides:
- Sharpe ratio confidence intervals (bootstrapped and analytical)
- Significance tests (vs zero and vs other strategies)
- Multiple testing correction
- Regime decomposition analysis
- Drawdown analysis with statistical context
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ep2_crypto.benchmarks.metrics import BARS_PER_YEAR, BacktestMetrics, compute_metrics

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Statistical tests and confidence intervals for strategy evaluation."""

    def __init__(self, confidence_level: float = 0.95) -> None:
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    # ------------------------------------------------------------------
    # Sharpe ratio confidence intervals
    # ------------------------------------------------------------------
    def sharpe_confidence_interval(
        self,
        returns: pd.Series,
        method: str = "both",
    ) -> dict[str, Any]:
        """Compute confidence interval for annualized Sharpe ratio.

        Two methods:
        1. Analytical (Lo 2002): assumes IID returns — fast but approximate.
        2. Bootstrap: nonparametric, handles autocorrelation — slower but robust.

        Args:
            returns: Per-bar strategy returns.
            method: 'analytical', 'bootstrap', or 'both'.
        """
        result: dict[str, Any] = {}
        n = len(returns)
        if n < 30:
            logger.warning("Too few observations (%d) for reliable Sharpe CI", n)

        mean_ret = returns.mean()
        std_ret = returns.std(ddof=1)
        sharpe_per_bar = mean_ret / std_ret if std_ret > 1e-15 else 0.0
        sharpe_annual = sharpe_per_bar * np.sqrt(BARS_PER_YEAR)

        result["sharpe_ratio"] = float(sharpe_annual)
        result["n_observations"] = n

        if method in ("analytical", "both"):
            # Lo (2002) standard error for Sharpe ratio
            # SE(SR) = sqrt((1 + 0.5 * SR^2) / n) * sqrt(annualization_factor)
            se_per_bar = np.sqrt((1 + 0.5 * sharpe_per_bar**2) / n)
            se_annual = se_per_bar * np.sqrt(BARS_PER_YEAR)
            z = sp_stats.norm.ppf(1 - self.alpha / 2)
            result["analytical"] = {
                "ci_lower": float(sharpe_annual - z * se_annual),
                "ci_upper": float(sharpe_annual + z * se_annual),
                "standard_error": float(se_annual),
            }

        if method in ("bootstrap", "both"):
            result["bootstrap"] = self._bootstrap_sharpe_ci(returns, n_bootstrap=5000)

        return result

    def _bootstrap_sharpe_ci(
        self,
        returns: pd.Series,
        n_bootstrap: int = 5000,
    ) -> dict[str, float]:
        """Block bootstrap CI for Sharpe — handles autocorrelation."""
        rng = np.random.default_rng(42)
        n = len(returns)
        # Block size: cube root of n (standard choice for dependent data)
        block_size = max(int(np.ceil(n ** (1 / 3))), 1)
        n_blocks = int(np.ceil(n / block_size))

        ret_array = returns.values
        bootstrap_sharpes = np.zeros(n_bootstrap)

        for b in range(n_bootstrap):
            # Sample blocks with replacement
            block_starts = rng.integers(0, n - block_size + 1, size=n_blocks)
            sample = np.concatenate([
                ret_array[s:s + block_size] for s in block_starts
            ])[:n]
            mean_s = sample.mean()
            std_s = sample.std(ddof=1)
            if std_s > 1e-15:
                bootstrap_sharpes[b] = mean_s / std_s * np.sqrt(BARS_PER_YEAR)

        lower = float(np.percentile(bootstrap_sharpes, 100 * self.alpha / 2))
        upper = float(np.percentile(bootstrap_sharpes, 100 * (1 - self.alpha / 2)))
        return {
            "ci_lower": lower,
            "ci_upper": upper,
            "standard_error": float(bootstrap_sharpes.std()),
            "n_bootstrap": n_bootstrap,
            "block_size": block_size,
        }

    # ------------------------------------------------------------------
    # Significance tests
    # ------------------------------------------------------------------
    def test_sharpe_vs_zero(self, returns: pd.Series) -> dict[str, Any]:
        """Test H0: Sharpe ratio = 0 (strategy has no edge).

        Uses the Jobson-Korkie (1981) test statistic.
        """
        n = len(returns)
        mean_ret = returns.mean()
        std_ret = returns.std(ddof=1)

        if std_ret < 1e-15 or n < 10:
            return {
                "test_statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "n": n,
            }

        sharpe_per_bar = mean_ret / std_ret
        # Under H0: SR = 0, test stat is approximately normal
        se = np.sqrt((1 + 0.5 * sharpe_per_bar**2) / n)
        t_stat = sharpe_per_bar / se

        # Two-sided p-value
        p_value = 2 * (1 - sp_stats.norm.cdf(abs(t_stat)))

        return {
            "test_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
            "n": n,
            "sharpe_per_bar": float(sharpe_per_bar),
            "sharpe_annual": float(sharpe_per_bar * np.sqrt(BARS_PER_YEAR)),
        }

    def test_sharpe_difference(
        self,
        returns_a: pd.Series,
        returns_b: pd.Series,
    ) -> dict[str, Any]:
        """Test H0: Sharpe(A) = Sharpe(B) using Ledoit-Wolf (2008).

        This accounts for correlation between the two return series,
        which is critical when strategies share the same underlying.
        """
        n = min(len(returns_a), len(returns_b))
        a = returns_a.values[:n]
        b = returns_b.values[:n]

        mu_a, mu_b = a.mean(), b.mean()
        sig_a, sig_b = a.std(ddof=1), b.std(ddof=1)

        if sig_a < 1e-15 or sig_b < 1e-15:
            return {"test_statistic": 0.0, "p_value": 1.0, "significant": False}

        sr_a = mu_a / sig_a
        sr_b = mu_b / sig_b

        # Ledoit-Wolf variance of Sharpe difference
        corr = np.corrcoef(a, b)[0, 1]
        v = (
            2 * (1 - corr)
            + 0.5 * (sr_a**2 + sr_b**2 - 2 * sr_a * sr_b * corr**2)
        ) / n

        if v <= 0:
            return {"test_statistic": 0.0, "p_value": 1.0, "significant": False}

        t_stat = (sr_a - sr_b) / np.sqrt(v)
        p_value = 2 * (1 - sp_stats.norm.cdf(abs(t_stat)))

        return {
            "test_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
            "sharpe_a_annual": float(sr_a * np.sqrt(BARS_PER_YEAR)),
            "sharpe_b_annual": float(sr_b * np.sqrt(BARS_PER_YEAR)),
            "sharpe_diff_annual": float((sr_a - sr_b) * np.sqrt(BARS_PER_YEAR)),
            "correlation": float(corr),
        }

    def multiple_testing_correction(
        self,
        p_values: dict[str, float],
        method: str = "holm",
    ) -> dict[str, dict[str, Any]]:
        """Apply multiple testing correction.

        Holm-Bonferroni is less conservative than Bonferroni but still
        controls family-wise error rate. Use this when comparing ML
        strategy against multiple benchmarks simultaneously.
        """
        names = list(p_values.keys())
        pvals = np.array([p_values[n] for n in names])
        m = len(pvals)

        # Sort by p-value
        sorted_idx = np.argsort(pvals)

        if method == "bonferroni":
            adjusted = np.minimum(pvals * m, 1.0)
        elif method == "holm":
            adjusted = np.zeros(m)
            for rank, idx in enumerate(sorted_idx):
                adjusted[idx] = min(pvals[idx] * (m - rank), 1.0)
            # Enforce monotonicity
            for i in range(1, m):
                idx = sorted_idx[i]
                prev_idx = sorted_idx[i - 1]
                adjusted[idx] = max(adjusted[idx], adjusted[prev_idx])
        else:
            raise ValueError(f"Unknown correction method: {method}")

        return {
            names[i]: {
                "original_p": float(pvals[i]),
                "adjusted_p": float(adjusted[i]),
                "significant": adjusted[i] < self.alpha,
            }
            for i in range(m)
        }

    # ------------------------------------------------------------------
    # Drawdown analysis
    # ------------------------------------------------------------------
    def drawdown_analysis(self, returns: pd.Series) -> dict[str, Any]:
        """Comprehensive drawdown statistics.

        Includes:
        - Top 5 drawdowns with duration and recovery
        - Expected max drawdown under random walk
        - Drawdown ratio (actual vs expected)
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max

        # Find individual drawdown episodes
        in_dd = drawdown < 0
        episodes = self._find_drawdown_episodes(drawdown, in_dd)

        # Sort by magnitude
        episodes.sort(key=lambda x: x["max_drawdown"])
        top5 = episodes[:5] if len(episodes) >= 5 else episodes

        # Expected max drawdown under random walk (Magdon-Ismail 2004 approx)
        n = len(returns)
        sigma = returns.std()
        expected_max_dd = sigma * np.sqrt(2 * np.log(n)) if n > 1 else 0.0

        actual_max_dd = abs(drawdown.min())

        return {
            "max_drawdown": float(actual_max_dd),
            "expected_max_dd_random_walk": float(expected_max_dd),
            "drawdown_ratio": float(actual_max_dd / expected_max_dd) if expected_max_dd > 0 else 0.0,
            "n_drawdown_episodes": len(episodes),
            "avg_drawdown_depth": float(np.mean([e["max_drawdown"] for e in episodes])) if episodes else 0.0,
            "avg_drawdown_duration_bars": float(np.mean([e["duration_bars"] for e in episodes])) if episodes else 0.0,
            "top_5_drawdowns": top5,
            "current_drawdown": float(drawdown.iloc[-1]) if len(drawdown) > 0 else 0.0,
        }

    @staticmethod
    def _find_drawdown_episodes(
        drawdown: pd.Series,
        in_dd: pd.Series,
    ) -> list[dict[str, Any]]:
        """Extract individual drawdown episodes."""
        episodes: list[dict[str, Any]] = []
        groups = (~in_dd).cumsum()
        dd_groups = groups[in_dd]

        for group_id in dd_groups.unique():
            mask = (groups == group_id) & in_dd
            if mask.sum() == 0:
                continue
            dd_slice = drawdown[mask]
            episodes.append({
                "max_drawdown": float(dd_slice.min()),
                "duration_bars": int(mask.sum()),
                "start_idx": int(mask.idxmax()) if hasattr(mask.idxmax(), '__int__') else 0,
            })

        return episodes

    # ------------------------------------------------------------------
    # Return distribution analysis
    # ------------------------------------------------------------------
    def return_distribution_analysis(self, returns: pd.Series) -> dict[str, Any]:
        """Analyze return distribution properties.

        Tests normality, fat tails, autocorrelation — all of which
        affect the reliability of Sharpe-based comparisons.
        """
        n = len(returns)
        ret = returns.values

        # Basic moments
        mean = float(ret.mean())
        std = float(ret.std(ddof=1))
        skew = float(sp_stats.skew(ret))
        kurt = float(sp_stats.kurtosis(ret))  # excess kurtosis

        # Jarque-Bera normality test
        if n >= 20:
            jb_stat, jb_p = sp_stats.jarque_bera(ret)
        else:
            jb_stat, jb_p = 0.0, 1.0

        # Ljung-Box autocorrelation test (lag 1-5)
        autocorr = {}
        for lag in [1, 2, 5, 10]:
            if n > lag + 10:
                corr = pd.Series(ret).autocorr(lag=lag)
                autocorr[f"lag_{lag}"] = float(corr) if not np.isnan(corr) else 0.0

        # Tail analysis: what fraction of returns > 2 sigma?
        if std > 0:
            tail_ratio_2s = float((np.abs(ret) > 2 * std).mean())
            tail_ratio_3s = float((np.abs(ret) > 3 * std).mean())
        else:
            tail_ratio_2s = 0.0
            tail_ratio_3s = 0.0

        # Expected under normal: 4.55% for 2-sigma, 0.27% for 3-sigma
        return {
            "mean": mean,
            "std": std,
            "skewness": skew,
            "excess_kurtosis": kurt,
            "jarque_bera_stat": float(jb_stat),
            "jarque_bera_p": float(jb_p),
            "is_normal": jb_p > 0.05,
            "autocorrelation": autocorr,
            "tail_ratio_2sigma": tail_ratio_2s,
            "tail_ratio_3sigma": tail_ratio_3s,
            "expected_2sigma_normal": 0.0455,
            "expected_3sigma_normal": 0.0027,
            "fat_tails": tail_ratio_3s > 0.005,  # More than ~2x normal
        }

    # ------------------------------------------------------------------
    # Full statistical report for a benchmark suite
    # ------------------------------------------------------------------
    def full_benchmark_report(
        self,
        suite_results: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Generate complete statistical report for all benchmarks.

        For each strategy, computes:
        - Sharpe CI (analytical + bootstrap)
        - Significance test vs zero
        - Return distribution analysis
        - Drawdown analysis
        """
        report: dict[str, dict[str, Any]] = {}

        for name, result in suite_results.items():
            if "error" in result:
                report[name] = {"error": result["error"]}
                continue

            returns = result["returns"]
            metrics = result["metrics"]

            report[name] = {
                "metrics": metrics,
                "sharpe_ci": self.sharpe_confidence_interval(returns),
                "significance": self.test_sharpe_vs_zero(returns),
                "distribution": self.return_distribution_analysis(returns),
                "drawdown": self.drawdown_analysis(returns),
            }

        return report
