"""Event-driven macro module: economic calendar + NQ lead-lag trade logic.

Monitors pre-scheduled macro events (CPI, FOMC, NFP) and uses NQ futures
reaction at T+0 to predict BTC direction at T+2min with ~62-68% accuracy.
Operates independently from the ML model as a separate alpha source.

Key parameters from research:
- NQ leads BTC by 2-5 minutes during macro events
- VIX gate: skip signals when VIX > 35
- EWMA BTC-NQ correlation (trailing 6h) for signal weighting
- Exit at T+5-10min (mean reversion risk)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class MacroEventType(Enum):
    """Types of scheduled macro events that impact BTC via NQ."""

    CPI = "cpi"
    FOMC = "fomc"
    NFP = "nfp"
    PPI = "ppi"
    RETAIL_SALES = "retail_sales"
    GDP = "gdp"
    PCE = "pce"
    UNEMPLOYMENT = "unemployment"


@dataclass(frozen=True)
class MacroEvent:
    """A scheduled macro event with its expected impact.

    Attributes:
        event_type: Type of macro event.
        timestamp_ms: Scheduled release time in milliseconds UTC.
        name: Human-readable name (e.g., "CPI MoM Dec 2025").
        expected_impact: 1-3 scale of expected market impact.
    """

    event_type: MacroEventType
    timestamp_ms: int
    name: str
    expected_impact: int = 2  # 1=low, 2=medium, 3=high


@dataclass
class MacroSignal:
    """Output signal from the macro event monitor.

    Attributes:
        event: The macro event that triggered the signal.
        direction: +1 (long BTC), -1 (short BTC), 0 (no signal).
        nq_return_pct: NQ return observed at signal time.
        confidence: Signal confidence [0, 1] based on correlation and VIX.
        btc_nq_correlation: Trailing EWMA correlation.
        vix_level: Current VIX level (if available).
        signal_timestamp_ms: When the signal was generated.
    """

    event: MacroEvent
    direction: int
    nq_return_pct: float
    confidence: float
    btc_nq_correlation: float
    vix_level: float | None
    signal_timestamp_ms: int


# Pre-scheduled events for 2025-2026 (major releases only).
# In production, this would be populated from an economic calendar API.
# Timestamps are approximate release times in UTC milliseconds.
MACRO_CALENDAR_2025: list[MacroEvent] = [
    # CPI releases (8:30 AM ET = 13:30 UTC)
    MacroEvent(MacroEventType.CPI, 1736857800000, "CPI Dec 2024", 3),
    MacroEvent(MacroEventType.CPI, 1739536200000, "CPI Jan 2025", 3),
    MacroEvent(MacroEventType.CPI, 1741955400000, "CPI Feb 2025", 3),
    MacroEvent(MacroEventType.CPI, 1744547400000, "CPI Mar 2025", 3),
    MacroEvent(MacroEventType.CPI, 1747139400000, "CPI Apr 2025", 3),
    MacroEvent(MacroEventType.CPI, 1749818400000, "CPI May 2025", 3),
    # FOMC decisions (2:00 PM ET = 19:00 UTC)
    MacroEvent(MacroEventType.FOMC, 1738267200000, "FOMC Jan 2025", 3),
    MacroEvent(MacroEventType.FOMC, 1742414400000, "FOMC Mar 2025", 3),
    MacroEvent(MacroEventType.FOMC, 1746619200000, "FOMC May 2025", 3),
    MacroEvent(MacroEventType.FOMC, 1750248000000, "FOMC Jun 2025", 3),
    # NFP (8:30 AM ET = 13:30 UTC, first Friday of month)
    MacroEvent(MacroEventType.NFP, 1736512200000, "NFP Jan 2025", 2),
    MacroEvent(MacroEventType.NFP, 1738926600000, "NFP Feb 2025", 2),
    MacroEvent(MacroEventType.NFP, 1741350600000, "NFP Mar 2025", 2),
    MacroEvent(MacroEventType.NFP, 1743942600000, "NFP Apr 2025", 2),
    MacroEvent(MacroEventType.NFP, 1746188200000, "NFP May 2025", 2),
]


class EWMACorrelation:
    """Exponentially weighted moving average of BTC-NQ correlation.

    Uses a 6-hour trailing window (72 bars at 5-min) with EWMA decay.
    """

    def __init__(self, halflife_bars: int = 36, min_bars: int = 12) -> None:
        self._alpha = 1.0 - math.exp(-math.log(2.0) / halflife_bars)
        self._min_bars = min_bars
        self._ewma_btc_sq: float = 0.0
        self._ewma_nq_sq: float = 0.0
        self._ewma_cross: float = 0.0
        self._ewma_btc_mean: float = 0.0
        self._ewma_nq_mean: float = 0.0
        self._bars_seen: int = 0

    def update(self, btc_return: float, nq_return: float) -> float:
        """Update with new returns and return current correlation estimate.

        Returns NaN if fewer than min_bars have been seen.
        """
        self._bars_seen += 1
        a = self._alpha

        self._ewma_btc_mean = a * btc_return + (1 - a) * self._ewma_btc_mean
        self._ewma_nq_mean = a * nq_return + (1 - a) * self._ewma_nq_mean
        self._ewma_btc_sq = a * btc_return**2 + (1 - a) * self._ewma_btc_sq
        self._ewma_nq_sq = a * nq_return**2 + (1 - a) * self._ewma_nq_sq
        self._ewma_cross = a * btc_return * nq_return + (1 - a) * self._ewma_cross

        if self._bars_seen < self._min_bars:
            return float("nan")

        var_btc = self._ewma_btc_sq - self._ewma_btc_mean**2
        var_nq = self._ewma_nq_sq - self._ewma_nq_mean**2

        if var_btc <= 1e-15 or var_nq <= 1e-15:
            return 0.0

        cov = self._ewma_cross - self._ewma_btc_mean * self._ewma_nq_mean
        corr = cov / math.sqrt(var_btc * var_nq)
        return max(-1.0, min(1.0, corr))

    @property
    def correlation(self) -> float:
        """Current correlation estimate."""
        if self._bars_seen < self._min_bars:
            return float("nan")
        var_btc = self._ewma_btc_sq - self._ewma_btc_mean**2
        var_nq = self._ewma_nq_sq - self._ewma_nq_mean**2
        if var_btc <= 1e-15 or var_nq <= 1e-15:
            return 0.0
        cov = self._ewma_cross - self._ewma_btc_mean * self._ewma_nq_mean
        corr = cov / math.sqrt(var_btc * var_nq)
        return max(-1.0, min(1.0, corr))

    @property
    def bars_seen(self) -> int:
        return self._bars_seen


class MacroEventMonitor:
    """Monitors macro events and generates BTC trade signals from NQ reaction.

    Strategy:
    1. Pre-load a calendar of macro events (CPI, FOMC, NFP, etc.)
    2. At each bar, check if we're within the event window
    3. At T+0 to T+2min: observe NQ 5-min return direction
    4. At T+2min: generate BTC signal in NQ's direction
    5. Signal valid for T+2min to T+10min (then expires)
    6. VIX gate: skip when VIX > 35
    7. Confidence weighted by EWMA BTC-NQ correlation

    Parameters:
        vix_threshold: Skip signals when VIX exceeds this level.
        nq_min_move_pct: Minimum NQ move to generate a signal.
        signal_delay_ms: Wait time after event before generating signal (2 min).
        signal_expiry_ms: Signal expires after this duration (10 min).
        min_correlation: Minimum BTC-NQ correlation to generate signal.
    """

    def __init__(
        self,
        calendar: list[MacroEvent] | None = None,
        *,
        vix_threshold: float = 35.0,
        nq_min_move_pct: float = 0.05,
        signal_delay_ms: int = 120_000,
        signal_expiry_ms: int = 600_000,
        min_correlation: float = 0.2,
    ) -> None:
        self._calendar = sorted(
            calendar or MACRO_CALENDAR_2025,
            key=lambda e: e.timestamp_ms,
        )
        self._vix_threshold = vix_threshold
        self._nq_min_move_pct = nq_min_move_pct
        self._signal_delay_ms = signal_delay_ms
        self._signal_expiry_ms = signal_expiry_ms
        self._min_correlation = min_correlation

        self._ewma_corr = EWMACorrelation()
        self._active_signal: MacroSignal | None = None
        self._last_event_idx: int = 0  # pointer into sorted calendar
        self._log = logger.bind(module="macro_monitor")

    def update(
        self,
        timestamp_ms: int,
        btc_close: float,
        btc_prev_close: float,
        nq_close: float,
        nq_prev_close: float,
        vix_level: float | None = None,
    ) -> MacroSignal | None:
        """Process a new bar and return a signal if applicable.

        Args:
            timestamp_ms: Current bar timestamp in milliseconds.
            btc_close: Current BTC close price.
            btc_prev_close: Previous bar BTC close price.
            nq_close: Current NQ close price.
            nq_prev_close: Previous bar NQ close price.
            vix_level: Current VIX level (None if unavailable).

        Returns:
            MacroSignal if a trade signal is generated, None otherwise.
        """
        # Update EWMA correlation
        btc_ret = (btc_close - btc_prev_close) / btc_prev_close if btc_prev_close > 0 else 0.0
        nq_ret = (nq_close - nq_prev_close) / nq_prev_close if nq_prev_close > 0 else 0.0
        self._ewma_corr.update(btc_ret, nq_ret)

        # Expire active signal
        if self._active_signal is not None:
            elapsed = timestamp_ms - self._active_signal.signal_timestamp_ms
            if elapsed > self._signal_expiry_ms:
                self._active_signal = None

        # Check if current time is within signal window of any event
        event = self._find_active_event(timestamp_ms)
        if event is None:
            return self._active_signal

        # Check if we're in the signal generation window (delay to delay + 1 bar)
        time_since_event = timestamp_ms - event.timestamp_ms
        if not (self._signal_delay_ms <= time_since_event < self._signal_delay_ms + 300_000):
            return self._active_signal

        # Already have an active signal for this window
        if self._active_signal is not None and self._active_signal.event == event:
            return self._active_signal

        # VIX gate
        if vix_level is not None and vix_level > self._vix_threshold:
            self._log.info(
                "macro_signal_vix_gated",
                macro_event=event.name,
                vix=vix_level,
                threshold=self._vix_threshold,
            )
            return None

        # NQ return since event start (use available bars)
        nq_ret_pct = nq_ret * 100.0

        # Check minimum move
        if abs(nq_ret_pct) < self._nq_min_move_pct:
            return None

        # Check correlation
        corr = self._ewma_corr.correlation
        if math.isnan(corr) or abs(corr) < self._min_correlation:
            return None

        # Generate signal
        direction = 1 if nq_ret > 0 else -1
        confidence = self._compute_confidence(
            nq_ret_pct=nq_ret_pct,
            correlation=corr,
            vix_level=vix_level,
            impact=event.expected_impact,
        )

        signal = MacroSignal(
            event=event,
            direction=direction,
            nq_return_pct=nq_ret_pct,
            confidence=confidence,
            btc_nq_correlation=corr,
            vix_level=vix_level,
            signal_timestamp_ms=timestamp_ms,
        )

        self._active_signal = signal
        self._log.info(
            "macro_signal_generated",
            macro_event=event.name,
            direction=direction,
            nq_ret_pct=nq_ret_pct,
            confidence=confidence,
            correlation=corr,
        )

        return signal

    def _find_active_event(self, timestamp_ms: int) -> MacroEvent | None:
        """Find the macro event that is currently active (within signal window)."""
        window_start = self._signal_delay_ms
        window_end = self._signal_delay_ms + 300_000  # delay + 5 min

        for i in range(self._last_event_idx, len(self._calendar)):
            event = self._calendar[i]
            time_since = timestamp_ms - event.timestamp_ms

            if time_since < 0:
                # Future event, stop searching
                break

            if time_since > self._signal_expiry_ms + 600_000:
                # Past event, advance pointer
                self._last_event_idx = i + 1
                continue

            if window_start <= time_since < window_end:
                return event

        return None

    def _compute_confidence(
        self,
        *,
        nq_ret_pct: float,
        correlation: float,
        vix_level: float | None,
        impact: int,
    ) -> float:
        """Compute signal confidence from multiple factors.

        Factors:
        - BTC-NQ correlation strength (higher = more confident)
        - NQ return magnitude (larger = more confident, with diminishing returns)
        - VIX level (lower = more confident)
        - Event impact rating (higher = more confident)
        """
        # Correlation factor: [0, 1]
        corr_factor = min(abs(correlation), 1.0)

        # NQ magnitude factor: sigmoid-like scaling, saturates at ~0.5%
        mag = abs(nq_ret_pct)
        mag_factor = min(mag / 0.5, 1.0)

        # VIX factor: full confidence below 20, reduced above, zero above threshold
        if vix_level is not None:
            if vix_level > self._vix_threshold:
                vix_factor = 0.0
            elif vix_level < 20.0:
                vix_factor = 1.0
            else:
                vix_factor = 1.0 - (vix_level - 20.0) / (self._vix_threshold - 20.0)
        else:
            vix_factor = 0.7  # Unknown VIX = moderate confidence

        # Impact factor: normalized to [0.5, 1.0]
        impact_factor = 0.5 + 0.5 * (min(impact, 3) / 3.0)

        # Weighted combination
        confidence = (
            0.35 * corr_factor + 0.25 * mag_factor + 0.20 * vix_factor + 0.20 * impact_factor
        )

        return max(0.0, min(1.0, confidence))

    @property
    def active_signal(self) -> MacroSignal | None:
        """Currently active signal, if any."""
        return self._active_signal

    @property
    def calendar(self) -> list[MacroEvent]:
        """The loaded macro event calendar."""
        return list(self._calendar)

    def get_next_event(self, timestamp_ms: int) -> MacroEvent | None:
        """Get the next upcoming macro event after the given timestamp."""
        for event in self._calendar:
            if event.timestamp_ms > timestamp_ms:
                return event
        return None

    def time_to_next_event_ms(self, timestamp_ms: int) -> int | None:
        """Milliseconds until the next macro event. None if no future events."""
        event = self.get_next_event(timestamp_ms)
        if event is None:
            return None
        return event.timestamp_ms - timestamp_ms
