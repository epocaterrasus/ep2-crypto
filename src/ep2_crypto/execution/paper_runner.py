"""Paper trading runner: connects live prediction signals to PaperExchange.

PaperRunner is a thin orchestration layer. It:
  1. Receives prediction dicts from the live prediction loop
  2. Applies the confidence threshold gate
  3. Converts directional signals to OrderRequests
  4. Routes orders to PaperExchange (or any VenueAdapter)
  5. Logs fills and tracks cumulative PnL

This is intentionally kept thin — risk management is the RiskManager's job,
position sizing is the PositionSizer's job. PaperRunner just connects them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from ep2_crypto.execution.paper_exchange import PaperExchange
from ep2_crypto.execution.venue import (
    OrderRequest,
    OrderResult,
    OrderSide,
    VenueAdapter,
)

logger = structlog.get_logger(__name__)

# Default confidence threshold for trading (can be overridden via config)
DEFAULT_CONFIDENCE_THRESHOLD = 0.60


@dataclass
class TradeSignal:
    """Normalized trade signal produced by the prediction loop."""

    direction: str  # "up", "down", "flat"
    confidence: float
    regime: str
    position_size_btc: float  # Pre-sized by PositionSizer
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    @property
    def is_tradeable(self) -> bool:
        return self.direction in ("up", "down")


class PaperRunner:
    """Connects live prediction signals to a VenueAdapter for paper execution.

    Usage:
        runner = PaperRunner(exchange=PaperExchange())
        await runner.on_signal(signal)
        summary = runner.get_summary()
    """

    def __init__(
        self,
        exchange: VenueAdapter,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        max_position_btc: float = 0.1,
    ) -> None:
        self._exchange = exchange
        self._confidence_threshold = confidence_threshold
        self._max_position_btc = max_position_btc
        self._signals_received = 0
        self._signals_traded = 0
        self._signals_skipped = 0
        self._last_direction: str = "flat"
        self._order_history: list[dict[str, Any]] = []

    async def on_signal(self, signal: TradeSignal) -> OrderResult | None:
        """Process a prediction signal and optionally place a paper order.

        Returns:
            OrderResult if an order was placed, None if signal was skipped.
        """
        self._signals_received += 1

        if not signal.is_tradeable:
            self._signals_skipped += 1
            logger.debug(
                "signal_skipped_flat",
                direction=signal.direction,
                confidence=round(signal.confidence, 4),
            )
            return None

        if signal.confidence < self._confidence_threshold:
            self._signals_skipped += 1
            logger.debug(
                "signal_skipped_low_confidence",
                direction=signal.direction,
                confidence=round(signal.confidence, 4),
                threshold=self._confidence_threshold,
            )
            return None

        size = min(signal.position_size_btc, self._max_position_btc)
        side = OrderSide.BUY if signal.direction == "up" else OrderSide.SELL

        # Skip if we'd be doubling up in the same direction
        if self._last_direction == signal.direction:
            self._signals_skipped += 1
            logger.debug(
                "signal_skipped_same_direction",
                direction=signal.direction,
            )
            return None

        request = OrderRequest(side=side, size=size)
        result = await self._exchange.place_order(request)

        self._signals_traded += 1
        self._last_direction = signal.direction if result.is_filled else self._last_direction

        entry = {
            "timestamp_ms": signal.timestamp_ms,
            "direction": signal.direction,
            "confidence": signal.confidence,
            "regime": signal.regime,
            "order_id": result.order_id,
            "fill_price": result.fill_price,
            "fill_quantity": result.fill_quantity,
            "fee": result.fee,
            "status": result.status.value,
        }
        self._order_history.append(entry)

        logger.info(
            "paper_signal_executed",
            direction=signal.direction,
            confidence=round(signal.confidence, 4),
            side=side.value,
            size=round(size, 6),
            fill_price=round(result.fill_price, 2),
            status=result.status.value,
        )

        return result

    async def close_position(self) -> OrderResult | None:
        """Close any open position (called at end of session or on kill switch)."""
        pos = await self._exchange.get_position()
        if not pos.is_open:
            return None

        # Opposite side to close
        close_side = OrderSide.SELL if pos.side == OrderSide.BUY else OrderSide.BUY
        request = OrderRequest(side=close_side, size=pos.size)
        result = await self._exchange.place_order(request)

        self._last_direction = "flat"
        logger.info(
            "paper_position_closed",
            close_side=close_side.value,
            size=pos.size,
            fill_price=result.fill_price,
        )
        return result

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of runner activity."""
        base: dict[str, Any] = {
            "signals_received": self._signals_received,
            "signals_traded": self._signals_traded,
            "signals_skipped": self._signals_skipped,
            "trade_rate": round(self._signals_traded / max(self._signals_received, 1), 4),
            "confidence_threshold": self._confidence_threshold,
        }
        # Add exchange-level summary if PaperExchange
        if isinstance(self._exchange, PaperExchange):
            base.update(self._exchange.get_summary())
        return base

    @property
    def exchange(self) -> VenueAdapter:
        return self._exchange

    @property
    def order_history(self) -> list[dict[str, Any]]:
        return list(self._order_history)
