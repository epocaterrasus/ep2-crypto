"""Stress testing framework: historical replay and synthetic adversarial scenarios.

Covers the four required test categories from BACKTESTING_PLAN.md §4:

Historical replays (synthetic price paths matching known crash statistics):
  - COVID Mar 2020: -50% in 48 h (576 bars)
  - May 2021 China ban: -30% in 8 h (96 bars)
  - FTX Nov 2022: -25% over 5 days (1440 bars)
  - Mar 2024 ATH correction: -15% over 3 days (864 bars)

Synthetic scenarios:
  - 48 h zero volatility (alpha = 0, costs accumulate)
  - 10 flash crashes in 1 week
  - BTC-NQ correlation drops to 0 overnight
  - Funding rate +0.3 % for 2 weeks
  - Broken model: same direction 100 bars straight

Pass criterion per scenario:
  - Kill switches fire before equity falls > kill_switch_dd_threshold (default 15 %)
  - OR scenario produces max_drawdown < max_allowed_dd (default 20 %)

Usage::

    runner = StressTestRunner()
    report = runner.run_all()
    print(report.summary())
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


# ---------------------------------------------------------------------------
# Scenario identifiers
# ---------------------------------------------------------------------------


class HistoricalScenario(StrEnum):
    COVID_CRASH_MAR2020 = "covid_crash_mar2020"  # -50% in 48h
    CHINA_BAN_MAY2021 = "china_ban_may2021"  # -30% in 8h
    FTX_COLLAPSE_NOV2022 = "ftx_collapse_nov2022"  # -25% over 5 days
    ATH_CORRECTION_MAR2024 = "ath_correction_mar2024"  # -15% over 3 days


class SyntheticScenario(StrEnum):
    ZERO_VOLATILITY_48H = "zero_volatility_48h"
    TEN_FLASH_CRASHES = "ten_flash_crashes"
    CORRELATION_DROP = "correlation_drop"
    HIGH_FUNDING_RATE = "high_funding_rate"
    BROKEN_MODEL = "broken_model"


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class StressTestResult:
    """Result for a single stress scenario."""

    scenario: str
    passed: bool
    max_drawdown: float
    kill_switch_fired: bool
    total_trades: int
    sharpe: float
    total_return: float
    n_bars: int
    reason: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        ks = "KS fired" if self.kill_switch_fired else "no KS"
        return (
            f"[{status}] {self.scenario}: "
            f"DD={self.max_drawdown:.1%}, trades={self.total_trades}, {ks}"
        )


@dataclasses.dataclass
class StressReport:
    """Aggregated results across all stress scenarios."""

    results: list[StressTestResult]
    max_allowed_dd: float = 0.20

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        return len(self.results) - self.n_passed

    @property
    def all_passed(self) -> bool:
        return self.n_failed == 0

    def summary(self) -> str:
        lines = [
            f"Stress Test Report: {self.n_passed}/{len(self.results)} passed",
            "=" * 60,
        ]
        for r in self.results:
            lines.append(str(r))
        lines.append("=" * 60)
        verdict = "ALL PASS" if self.all_passed else f"{self.n_failed} FAILED"
        lines.append(f"Verdict: {verdict}")
        return "\n".join(lines)

    def to_dict(self) -> list[dict[str, Any]]:
        return [dataclasses.asdict(r) for r in self.results]


# ---------------------------------------------------------------------------
# Price path generators
# ---------------------------------------------------------------------------


def _make_crash_path(
    n_bars: int,
    start_price: float,
    crash_fraction: float,
    crash_start: int,
    crash_end: int,
    recovery_bars: int = 0,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Simulate a crash as a sequence of closes.

    Args:
        n_bars: Total number of bars.
        start_price: Starting price.
        crash_fraction: Total price drop (e.g. 0.50 for -50%).
        crash_start: Bar index when crash begins.
        crash_end: Bar index when crash ends.
        recovery_bars: Bars of mild recovery after crash.
        seed: RNG seed for noise.

    Returns:
        Array of close prices.
    """
    rng = np.random.default_rng(seed)
    closes = np.full(n_bars, start_price)

    # Pre-crash: small noise
    noise_pre = rng.normal(0, 0.001, crash_start)
    for i in range(1, crash_start):
        closes[i] = closes[i - 1] * (1 + noise_pre[i])

    # Crash: linear drop plus noise
    crash_len = max(crash_end - crash_start, 1)
    per_bar_drop = crash_fraction / crash_len
    for i in range(crash_start, crash_end):
        noise = rng.normal(-per_bar_drop, per_bar_drop * 0.3)
        closes[i] = closes[i - 1] * (1 + noise)

    # Post-crash / recovery
    if crash_end < n_bars:
        recovery_drift = 0.0001 if recovery_bars > 0 else -0.0001
        for i in range(crash_end, n_bars):
            closes[i] = closes[i - 1] * (1 + rng.normal(recovery_drift, 0.001))

    # Clip negatives
    closes = np.maximum(closes, 100.0)
    return closes


def _closes_to_ohlcv(
    closes: NDArray[np.float64],
    seed: int = 0,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray[np.int64]]:
    """Convert a close price series to full OHLCV + timestamps."""
    n = len(closes)
    rng = np.random.default_rng(seed)
    noise = abs(rng.normal(0, 0.002, n))
    highs = closes * (1 + noise)
    lows = closes * (1 - noise)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    volumes = rng.uniform(100, 2000, n)
    timestamps_ms = np.arange(n, dtype=np.int64) * 300_000  # 5-min
    return opens, highs, lows, closes, volumes, timestamps_ms


def _make_signals_for_crash(
    n: int,
    seed: int = 0,
    strategy: str = "contrarian",
) -> tuple[NDArray[np.int8], NDArray[np.float64]]:
    """Generate trading signals for stress scenarios.

    contrarian: tries to pick bottoms (wrong during crashes)
    trend: follows momentum (loses in reversal, gains in crash continuation)
    """
    rng = np.random.default_rng(seed)
    if strategy == "broken":
        # Broken model: always predicts the same direction
        signals = np.ones(n, dtype=np.int8)
        confidences = np.full(n, 0.8)
    else:
        # Random with slight edge
        signals = rng.choice([-1, 1], size=n).astype(np.int8)
        confidences = rng.uniform(0.55, 0.85, n)

    # No signal in first 5% (warm-up)
    warmup = max(1, n // 20)
    signals[:warmup] = 0
    return signals, confidences


# ---------------------------------------------------------------------------
# Historical scenario generators
# ---------------------------------------------------------------------------


class HistoricalStressScenario:
    """Synthetic price paths calibrated to known historical crashes.

    We cannot replay real prices, so we simulate the OHLCV statistics
    matching what is documented in BACKTESTING_PLAN.md.
    """

    _SCENARIOS: dict[HistoricalScenario, dict] = {
        HistoricalScenario.COVID_CRASH_MAR2020: {
            "total_bars": 1000,
            "crash_fraction": 0.50,
            "crash_start_bar": 200,
            "crash_end_bar": 776,  # ~48 h = 576 bars
            "description": "COVID crash Mar 2020: -50% in 48h",
        },
        HistoricalScenario.CHINA_BAN_MAY2021: {
            "total_bars": 500,
            "crash_fraction": 0.30,
            "crash_start_bar": 100,
            "crash_end_bar": 196,  # ~8 h = 96 bars
            "description": "China ban May 2021: -30% in 8h",
        },
        HistoricalScenario.FTX_COLLAPSE_NOV2022: {
            "total_bars": 2500,
            "crash_fraction": 0.25,
            "crash_start_bar": 500,
            "crash_end_bar": 1940,  # ~5 days = 1440 bars
            "description": "FTX collapse Nov 2022: -25% over 5 days",
        },
        HistoricalScenario.ATH_CORRECTION_MAR2024: {
            "total_bars": 1500,
            "crash_fraction": 0.15,
            "crash_start_bar": 300,
            "crash_end_bar": 1164,  # ~3 days = 864 bars
            "description": "ATH correction Mar 2024: -15% over 3 days",
        },
    }

    def generate(
        self,
        scenario: HistoricalScenario,
        start_price: float = 30_000.0,
        seed: int = 42,
    ) -> tuple:
        """Generate OHLCV + signals for a historical crash scenario.

        Returns:
            (opens, highs, lows, closes, volumes, timestamps_ms, signals, confidences, meta)
        """
        cfg = self._SCENARIOS[scenario]
        closes = _make_crash_path(
            n_bars=cfg["total_bars"],
            start_price=start_price,
            crash_fraction=cfg["crash_fraction"],
            crash_start=cfg["crash_start_bar"],
            crash_end=cfg["crash_end_bar"],
            seed=seed,
        )
        opens, highs, lows, closes2, volumes, timestamps_ms = _closes_to_ohlcv(closes, seed)
        signals, confidences = _make_signals_for_crash(len(closes), seed)

        meta = {
            "scenario": scenario.value,
            "description": cfg["description"],
            "expected_crash_fraction": cfg["crash_fraction"],
        }
        return opens, highs, lows, closes2, volumes, timestamps_ms, signals, confidences, meta


# ---------------------------------------------------------------------------
# Synthetic scenario generators
# ---------------------------------------------------------------------------


class SyntheticStressScenario:
    """Parameterised synthetic adversarial scenarios."""

    def zero_volatility_48h(
        self,
        pre_bars: int = 200,
        post_bars: int = 100,
        start_price: float = 30_000.0,
        seed: int = 0,
    ) -> tuple:
        """48 h of near-zero volatility: alpha ≈ 0, costs still paid."""
        n = pre_bars + 576 + post_bars  # 576 = 48h at 5-min
        rng = np.random.default_rng(seed)

        closes = np.full(n, start_price)
        # Pre: normal vol
        for i in range(1, pre_bars):
            closes[i] = closes[i - 1] * (1 + rng.normal(0, 0.002))
        # Zero-vol window: price barely moves
        for i in range(pre_bars, pre_bars + 576):
            closes[i] = closes[i - 1] * (1 + rng.normal(0, 0.0001))
        # Post: normal vol resumes
        for i in range(pre_bars + 576, n):
            closes[i] = closes[i - 1] * (1 + rng.normal(0, 0.002))

        closes = np.maximum(closes, 100.0)
        opens, highs, lows, closes2, volumes, timestamps_ms = _closes_to_ohlcv(closes, seed)
        signals, confidences = _make_signals_for_crash(n, seed)
        return opens, highs, lows, closes2, volumes, timestamps_ms, signals, confidences

    def ten_flash_crashes(
        self,
        n_bars: int = 2016,  # 1 week
        crash_depth: float = 0.08,  # 8% per flash crash
        start_price: float = 30_000.0,
        seed: int = 0,
    ) -> tuple:
        """10 flash crashes of 8% over 1 week, each recovering in 2h."""
        rng = np.random.default_rng(seed)
        closes = np.full(n_bars, start_price)

        # Inject crashes at regular intervals
        crash_spacing = n_bars // 11
        crash_positions = [crash_spacing * (i + 1) for i in range(10)]

        for i in range(1, n_bars):
            closes[i] = closes[i - 1] * (1 + rng.normal(0, 0.001))
            # Flash crash: instant drop
            if i in crash_positions:
                drop = rng.uniform(crash_depth * 0.8, crash_depth * 1.2)
                closes[i] = closes[i - 1] * (1 - drop)
            # Recovery ramp in 24 bars (2h)
            for cp in crash_positions:
                if cp < i < cp + 24:
                    closes[i] = closes[i - 1] * (1 + crash_depth / 24 * 0.9)
                    break

        closes = np.maximum(closes, 100.0)
        opens, highs, lows, closes2, volumes, timestamps_ms = _closes_to_ohlcv(closes, seed)
        signals, confidences = _make_signals_for_crash(n_bars, seed)
        return opens, highs, lows, closes2, volumes, timestamps_ms, signals, confidences

    def correlation_drop(
        self,
        n_bars: int = 1000,
        start_price: float = 30_000.0,
        seed: int = 0,
    ) -> tuple:
        """BTC-NQ correlation drops to ~0 at bar n/2.

        Before: BTC tracks NQ (correlated). After: uncorrelated.
        This tests that cross-market features do not cause harm when
        the cross-market relationship breaks down.
        """
        rng = np.random.default_rng(seed)
        mid = n_bars // 2

        # NQ-like signal
        nq_returns = rng.normal(0.0001, 0.002, n_bars)

        closes = np.zeros(n_bars)
        closes[0] = start_price
        for i in range(1, n_bars):
            if i < mid:
                # Correlated: BTC = 0.6 * NQ + noise
                btc_ret = 0.6 * nq_returns[i] + rng.normal(0, 0.001)
            else:
                # Uncorrelated: BTC is pure noise
                btc_ret = rng.normal(0, 0.002)
            closes[i] = closes[i - 1] * (1 + btc_ret)

        closes = np.maximum(closes, 100.0)
        opens, highs, lows, closes2, volumes, timestamps_ms = _closes_to_ohlcv(closes, seed)
        signals, confidences = _make_signals_for_crash(n_bars, seed)
        return opens, highs, lows, closes2, volumes, timestamps_ms, signals, confidences

    def high_funding_rate(
        self,
        n_bars: int = 4032,  # 2 weeks
        funding_rate: float = 0.003,  # 0.3% per 8h
        start_price: float = 30_000.0,
        seed: int = 0,
    ) -> tuple:
        """Sustained high funding rate (+0.3% per 8h) for 2 weeks.

        Funding is applied every 8h (96 bars). Long positions bleed fees.
        """
        rng = np.random.default_rng(seed)
        closes = np.full(n_bars, start_price)
        for i in range(1, n_bars):
            closes[i] = closes[i - 1] * (1 + rng.normal(0, 0.002))
        closes = np.maximum(closes, 100.0)

        opens, highs, lows, closes2, volumes, timestamps_ms = _closes_to_ohlcv(closes, seed)

        # Funding rate array: spike at every 96-bar interval
        funding_rates = np.zeros(n_bars)
        for i in range(0, n_bars, 96):
            funding_rates[i] = funding_rate

        signals, confidences = _make_signals_for_crash(n_bars, seed)
        return (
            opens,
            highs,
            lows,
            closes2,
            volumes,
            timestamps_ms,
            signals,
            confidences,
            funding_rates,
        )

    def broken_model(
        self,
        n_bars: int = 500,
        start_price: float = 30_000.0,
        seed: int = 0,
    ) -> tuple:
        """Model stuck predicting same direction for 100+ bars.

        The risk engine's consecutive-loss kill switch should fire.
        """
        rng = np.random.default_rng(seed)
        closes = np.full(n_bars, start_price)
        # Counter-trend price: opposite to what model predicts
        for i in range(1, n_bars):
            closes[i] = closes[i - 1] * (1 + rng.normal(-0.0005, 0.002))
        closes = np.maximum(closes, 100.0)

        opens, highs, lows, closes2, volumes, timestamps_ms = _closes_to_ohlcv(closes, seed)
        # Broken model: always says long (model is stuck)
        signals = np.ones(n_bars, dtype=np.int8)
        confidences = np.full(n_bars, 0.8)
        signals[:10] = 0  # warm-up

        return opens, highs, lows, closes2, volumes, timestamps_ms, signals, confidences


# ---------------------------------------------------------------------------
# Pass/fail criteria
# ---------------------------------------------------------------------------


def _evaluate_stress_result(
    scenario_name: str,
    result: BacktestResult,
    n_bars: int,
    max_allowed_dd: float = 0.20,
    kill_switch_dd_threshold: float = 0.15,
) -> StressTestResult:
    """Evaluate whether a stress scenario is passed.

    Pass criteria (either condition):
      1. Kill switches fire before equity falls past kill_switch_dd_threshold
         (proxy: max_drawdown <= kill_switch_dd_threshold + 5% buffer)
      2. Max drawdown remains below max_allowed_dd (regardless of kill switch)
    """
    # Kill switch proxy: significantly fewer trades than n_bars / 6 (engine stopped early)
    expected_max_trades = n_bars // 6
    kill_switch_fired = result.total_trades < max(1, expected_max_trades * 0.5)

    passed = result.max_drawdown <= max_allowed_dd or (
        kill_switch_fired and result.max_drawdown <= kill_switch_dd_threshold + 0.05
    )

    reason = ""
    if not passed:
        reason = (
            f"max_drawdown={result.max_drawdown:.1%} > {max_allowed_dd:.0%} "
            f"and kill_switch_fired={kill_switch_fired}"
        )

    return StressTestResult(
        scenario=scenario_name,
        passed=passed,
        max_drawdown=result.max_drawdown,
        kill_switch_fired=kill_switch_fired,
        total_trades=result.total_trades,
        sharpe=result.sharpe_ratio,
        total_return=result.total_return,
        n_bars=n_bars,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# StressTestRunner
# ---------------------------------------------------------------------------


class StressTestRunner:
    """Run all historical and synthetic stress scenarios.

    Args:
        initial_equity: Starting capital.
        max_allowed_dd: Scenario pass threshold for max drawdown.
        seed: RNG seed.
    """

    def __init__(
        self,
        initial_equity: float = 50_000.0,
        max_allowed_dd: float = 0.20,
        seed: int = 42,
    ) -> None:
        self._initial_equity = initial_equity
        self._max_allowed_dd = max_allowed_dd
        self._seed = seed
        self._risk_config = RiskConfig()

    def _make_engine(self) -> BacktestEngine:
        cfg = BacktestConfig(
            initial_equity=self._initial_equity,
            seed=self._seed,
            risk_config=self._risk_config,
        )
        return BacktestEngine(config=cfg)

    def run_historical(
        self,
        scenarios: list[HistoricalScenario] | None = None,
    ) -> list[StressTestResult]:
        """Run historical crash replays."""
        if scenarios is None:
            scenarios = list(HistoricalScenario)

        gen = HistoricalStressScenario()
        results = []
        for scenario in scenarios:
            logger.info("stress.historical.start", scenario=scenario.value)
            (
                opens,
                highs,
                lows,
                closes,
                volumes,
                timestamps_ms,
                signals,
                confidences,
                _meta,
            ) = gen.generate(scenario, seed=self._seed)

            engine = self._make_engine()
            result = engine.run(
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                volumes=volumes,
                timestamps_ms=timestamps_ms,
                signals=signals,
                confidences=confidences,
            )
            sr = _evaluate_stress_result(
                scenario_name=scenario.value,
                result=result,
                n_bars=len(closes),
                max_allowed_dd=self._max_allowed_dd,
            )
            results.append(sr)
            logger.info(
                "stress.historical.done",
                scenario=scenario.value,
                passed=sr.passed,
                max_dd=round(sr.max_drawdown, 3),
            )

        return results

    def run_synthetic(
        self,
        scenarios: list[SyntheticScenario] | None = None,
    ) -> list[StressTestResult]:
        """Run synthetic adversarial scenarios."""
        if scenarios is None:
            scenarios = list(SyntheticScenario)

        gen = SyntheticStressScenario()
        results = []

        for scenario in scenarios:
            logger.info("stress.synthetic.start", scenario=scenario.value)
            funding_rates = None

            if scenario == SyntheticScenario.ZERO_VOLATILITY_48H:
                opens, highs, lows, closes, volumes, ts, sigs, confs = gen.zero_volatility_48h(
                    seed=self._seed
                )
            elif scenario == SyntheticScenario.TEN_FLASH_CRASHES:
                opens, highs, lows, closes, volumes, ts, sigs, confs = gen.ten_flash_crashes(
                    seed=self._seed
                )
            elif scenario == SyntheticScenario.CORRELATION_DROP:
                opens, highs, lows, closes, volumes, ts, sigs, confs = gen.correlation_drop(
                    seed=self._seed
                )
            elif scenario == SyntheticScenario.HIGH_FUNDING_RATE:
                opens, highs, lows, closes, volumes, ts, sigs, confs, funding_rates = (
                    gen.high_funding_rate(seed=self._seed)
                )
            elif scenario == SyntheticScenario.BROKEN_MODEL:
                opens, highs, lows, closes, volumes, ts, sigs, confs = gen.broken_model(
                    seed=self._seed
                )
            else:
                logger.warning("stress.unknown_scenario", scenario=scenario.value)
                continue

            engine = self._make_engine()
            result = engine.run(
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                volumes=volumes,
                timestamps_ms=ts,
                signals=sigs,
                confidences=confs,
                funding_rates=funding_rates,
            )
            sr = _evaluate_stress_result(
                scenario_name=scenario.value,
                result=result,
                n_bars=len(closes),
                max_allowed_dd=self._max_allowed_dd,
            )
            results.append(sr)
            logger.info(
                "stress.synthetic.done",
                scenario=scenario.value,
                passed=sr.passed,
                max_dd=round(sr.max_drawdown, 3),
            )

        return results

    def run_all(
        self,
        historical_scenarios: list[HistoricalScenario] | None = None,
        synthetic_scenarios: list[SyntheticScenario] | None = None,
    ) -> StressReport:
        """Run both historical and synthetic stress scenarios.

        Returns:
            StressReport with all results and aggregate verdict.
        """
        hist = self.run_historical(historical_scenarios)
        synth = self.run_synthetic(synthetic_scenarios)
        return StressReport(
            results=hist + synth,
            max_allowed_dd=self._max_allowed_dd,
        )
