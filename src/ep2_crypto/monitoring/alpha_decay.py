"""Multi-timescale alpha decay detection.

Monitors strategy performance using four complementary methods:
- CUSUM: Fast detection of mean shift in returns (2-5 bars)
- Rolling Sharpe: Medium-term performance decline (7-14 days)
- ADWIN: Adaptive windowing for distribution change
- SPRT: Sequential probability ratio test on win rate

Response protocol:
  NORMAL → WARNING → CAUTION → STOP → EMERGENCY
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

ANNUALIZATION_FACTOR = math.sqrt(105_120)  # 24/7 crypto, 5-min bars


class AlertLevel(IntEnum):
    """Monotonically increasing severity levels."""

    NORMAL = 0
    WARNING = 1
    CAUTION = 2
    STOP = 3
    EMERGENCY = 4


@dataclass
class AlphaDecayState:
    """Aggregate state from all detectors."""

    level: AlertLevel = AlertLevel.NORMAL
    cusum_signal: bool = False
    sharpe_declining: bool = False
    adwin_detected: bool = False
    sprt_rejected: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def should_trade(self) -> bool:
        return self.level < AlertLevel.STOP


class CUSUMDetector:
    """Cumulative sum control chart for fast mean-shift detection.

    Detects when returns shift away from the expected mean.
    Sensitivity controlled by `threshold` (h) and `drift` (k).
    """

    def __init__(
        self,
        threshold: float = 0.005,
        drift: float = 0.001,
        target_mean: float = 0.0,
    ) -> None:
        self._threshold = threshold
        self._drift = drift
        self._target_mean = target_mean
        self._s_pos = 0.0
        self._s_neg = 0.0
        self._n = 0
        self._signal = False

    def update(self, value: float) -> bool:
        """Update with new return value. Returns True if alarm fires."""
        z = value - self._target_mean
        self._s_pos = max(0.0, self._s_pos + z - self._drift)
        self._s_neg = max(0.0, self._s_neg - z - self._drift)
        self._n += 1
        self._signal = self._s_pos > self._threshold or self._s_neg > self._threshold
        if self._signal:
            logger.warning(
                "cusum_alarm",
                s_pos=self._s_pos,
                s_neg=self._s_neg,
                threshold=self._threshold,
                n=self._n,
            )
        return self._signal

    def reset(self) -> None:
        """Reset accumulators (after model retrain or regime change)."""
        self._s_pos = 0.0
        self._s_neg = 0.0
        self._signal = False

    @property
    def signal(self) -> bool:
        return self._signal

    @property
    def s_pos(self) -> float:
        return self._s_pos

    @property
    def s_neg(self) -> float:
        return self._s_neg


class RollingSharpeMonitor:
    """Monitors rolling Sharpe ratio for medium-term performance decline.

    Compares short-window Sharpe vs long-window baseline.
    """

    def __init__(
        self,
        short_window: int = 2016,  # ~7 days of 5-min bars
        long_window: int = 4032,  # ~14 days
        decline_threshold: float = 0.5,  # Alert when short < 50% of long
    ) -> None:
        self._short_window = short_window
        self._long_window = long_window
        self._decline_threshold = decline_threshold
        self._returns: deque[float] = deque(maxlen=long_window)
        self._declining = False

    def update(self, ret: float) -> bool:
        """Update with new return. Returns True if Sharpe is declining."""
        self._returns.append(ret)
        if len(self._returns) < self._short_window:
            self._declining = False
            return False

        short_sharpe = self._compute_sharpe(list(self._returns)[-self._short_window :])
        long_sharpe = self._compute_sharpe(list(self._returns))

        if long_sharpe > 0 and short_sharpe < long_sharpe * self._decline_threshold:
            self._declining = True
            logger.warning(
                "sharpe_declining",
                short_sharpe=round(short_sharpe, 4),
                long_sharpe=round(long_sharpe, 4),
                ratio=round(short_sharpe / long_sharpe, 4) if long_sharpe != 0 else 0,
            )
        else:
            self._declining = False
        return self._declining

    @staticmethod
    def _compute_sharpe(returns: list[float]) -> float:
        """Annualized Sharpe ratio for 5-min bars."""
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        std = math.sqrt(var) if var > 0 else 1e-10
        return (mean / std) * ANNUALIZATION_FACTOR

    @property
    def declining(self) -> bool:
        return self._declining

    @property
    def current_sharpe(self) -> float:
        if len(self._returns) < 2:
            return 0.0
        return self._compute_sharpe(list(self._returns))

    def reset(self) -> None:
        self._returns.clear()
        self._declining = False


class ADWINDetector:
    """Adaptive Windowing (ADWIN) for distribution change detection.

    Maintains a variable-length window and detects when older and newer
    sub-windows have statistically different means. Uses the simplified
    ADWIN algorithm with epsilon-cut.
    """

    def __init__(self, delta: float = 0.002) -> None:
        self._delta = delta
        self._window: deque[float] = deque()
        self._sum = 0.0
        self._n = 0
        self._detected = False

    def update(self, value: float) -> bool:
        """Add value and check for distribution change."""
        self._window.append(value)
        self._sum += value
        self._n += 1
        self._detected = False

        if self._n < 10:
            return False

        # Check for cut point
        self._detected = self._check_cut()
        if self._detected:
            logger.warning(
                "adwin_change_detected",
                window_size=self._n,
            )
        return self._detected

    def _check_cut(self) -> bool:
        """Check if any split of the window shows a significant mean difference."""
        total = self._sum
        n = self._n
        prefix_sum = 0.0

        items = list(self._window)
        for i in range(1, n):
            prefix_sum += items[i - 1]
            n0 = i
            n1 = n - i
            if n0 < 5 or n1 < 5:
                continue

            mean0 = prefix_sum / n0
            mean1 = (total - prefix_sum) / n1
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            epsilon = math.sqrt(math.log(2.0 / self._delta) / (2.0 * m))
            if abs(mean0 - mean1) >= epsilon:
                # Shrink window: drop older half
                for _ in range(n0):
                    dropped = self._window.popleft()
                    self._sum -= dropped
                    self._n -= 1
                return True
        return False

    @property
    def detected(self) -> bool:
        return self._detected

    @property
    def window_size(self) -> int:
        return self._n

    def reset(self) -> None:
        self._window.clear()
        self._sum = 0.0
        self._n = 0
        self._detected = False


class SPRTMonitor:
    """Sequential Probability Ratio Test on win rate.

    Tests H0: win_rate = p0 vs H1: win_rate = p1 (one-sided).
    Rejects H0 (alpha decay detected) when log-likelihood ratio exceeds threshold.
    """

    def __init__(
        self,
        p0: float = 0.52,  # Expected win rate under H0 (strategy works)
        p1: float = 0.48,  # Win rate under H1 (alpha decayed)
        alpha: float = 0.05,  # Type I error
        beta: float = 0.10,  # Type II error
        min_trades: int = 50,
    ) -> None:
        if p0 <= p1:
            raise ValueError("p0 must be > p1 for decay detection")
        self._p0 = p0
        self._p1 = p1
        # SPRT boundaries (Wald):
        #   Reject H0 (decay detected) when LLR >= upper = log((1-beta)/alpha)
        #   Accept H0 (strategy fine) when LLR <= lower = log(beta/(1-alpha))
        # Losses push LLR positive (evidence for H1: lower win rate).
        self._upper = math.log((1.0 - beta) / alpha)
        self._lower = math.log(beta / (1.0 - alpha))
        self._min_trades = min_trades
        self._llr = 0.0  # Log-likelihood ratio
        self._n = 0
        self._wins = 0
        self._rejected = False  # H0 rejected = alpha decay detected

    def update(self, win: bool) -> bool:
        """Update with trade outcome. Returns True if H0 rejected (decay detected)."""
        self._n += 1
        if win:
            self._wins += 1
            self._llr += math.log(self._p1 / self._p0)
        else:
            self._llr += math.log((1.0 - self._p1) / (1.0 - self._p0))

        if self._n < self._min_trades:
            self._rejected = False
            return False

        if self._llr >= self._upper:
            # Reject H0 — alpha decay detected
            self._rejected = True
            logger.warning(
                "sprt_h0_rejected",
                llr=round(self._llr, 4),
                n=self._n,
                win_rate=round(self._wins / self._n, 4),
            )
            return True

        if self._llr <= self._lower:
            # Accept H0 — strategy is fine, reset
            self._rejected = False
            self.reset()
            return False

        self._rejected = False
        return False

    @property
    def rejected(self) -> bool:
        return self._rejected

    @property
    def llr(self) -> float:
        return self._llr

    @property
    def win_rate(self) -> float | None:
        if self._n == 0:
            return None
        return self._wins / self._n

    def reset(self) -> None:
        self._llr = 0.0
        self._n = 0
        self._wins = 0
        self._rejected = False


class AlphaDecayMonitor:
    """Orchestrates all four detectors and determines the aggregate alert level.

    Response protocol:
      NORMAL:    All clear
      WARNING:   1 detector fires
      CAUTION:   2 detectors fire
      STOP:      3 detectors fire — halt new trades
      EMERGENCY: 4 detectors fire or CUSUM + SPRT both fire — halt everything
    """

    def __init__(
        self,
        cusum_threshold: float = 0.005,
        cusum_drift: float = 0.001,
        sharpe_short_window: int = 2016,
        sharpe_long_window: int = 4032,
        sharpe_decline_threshold: float = 0.5,
        adwin_delta: float = 0.002,
        sprt_p0: float = 0.52,
        sprt_p1: float = 0.48,
        sprt_min_trades: int = 50,
    ) -> None:
        self.cusum = CUSUMDetector(threshold=cusum_threshold, drift=cusum_drift)
        self.rolling_sharpe = RollingSharpeMonitor(
            short_window=sharpe_short_window,
            long_window=sharpe_long_window,
            decline_threshold=sharpe_decline_threshold,
        )
        self.adwin = ADWINDetector(delta=adwin_delta)
        self.sprt = SPRTMonitor(p0=sprt_p0, p1=sprt_p1, min_trades=sprt_min_trades)
        self._state = AlphaDecayState()

    def on_bar(self, bar_return: float) -> AlphaDecayState:
        """Update bar-level detectors (CUSUM, Sharpe, ADWIN)."""
        self.cusum.update(bar_return)
        self.rolling_sharpe.update(bar_return)
        self.adwin.update(bar_return)
        return self._compute_state()

    def on_trade(self, pnl: float) -> AlphaDecayState:
        """Update trade-level detector (SPRT)."""
        self.sprt.update(win=pnl > 0)
        return self._compute_state()

    def _compute_state(self) -> AlphaDecayState:
        """Determine aggregate alert level from all detectors."""
        signals = [
            self.cusum.signal,
            self.rolling_sharpe.declining,
            self.adwin.detected,
            self.sprt.rejected,
        ]
        active_count = sum(signals)

        # CUSUM + SPRT together = emergency (fast + statistical confirmation)
        if (self.cusum.signal and self.sprt.rejected) or active_count >= 4:
            level = AlertLevel.EMERGENCY
        elif active_count == 3:
            level = AlertLevel.STOP
        elif active_count == 2:
            level = AlertLevel.CAUTION
        elif active_count == 1:
            level = AlertLevel.WARNING
        else:
            level = AlertLevel.NORMAL

        self._state = AlphaDecayState(
            level=level,
            cusum_signal=self.cusum.signal,
            sharpe_declining=self.rolling_sharpe.declining,
            adwin_detected=self.adwin.detected,
            sprt_rejected=self.sprt.rejected,
            details={
                "cusum_s_pos": self.cusum.s_pos,
                "cusum_s_neg": self.cusum.s_neg,
                "rolling_sharpe": self.rolling_sharpe.current_sharpe,
                "adwin_window_size": self.adwin.window_size,
                "sprt_llr": self.sprt.llr,
                "sprt_win_rate": self.sprt.win_rate,
                "active_detectors": active_count,
            },
        )

        if level >= AlertLevel.WARNING:
            logger.warning(
                "alpha_decay_alert",
                level=level.name,
                active_count=active_count,
                cusum=self.cusum.signal,
                sharpe=self.rolling_sharpe.declining,
                adwin=self.adwin.detected,
                sprt=self.sprt.rejected,
            )

        return self._state

    @property
    def state(self) -> AlphaDecayState:
        return self._state

    def reset(self) -> None:
        """Reset all detectors (e.g., after model retrain)."""
        self.cusum.reset()
        self.rolling_sharpe.reset()
        self.adwin.reset()
        self.sprt.reset()
        self._state = AlphaDecayState()
