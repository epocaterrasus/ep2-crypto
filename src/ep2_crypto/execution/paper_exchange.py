"""Paper trading exchange: simulates fills against real orderbook data.

PaperExchange implements VenueAdapter so the live trading pipeline can swap
between paper and live execution by changing a single config flag. All logic
above the execution layer (signal generation, risk management, monitoring)
remains identical.

Fill simulation:
  - Market orders walk the orderbook levels until filled or exhausted
  - Taker fee 4 bps applied on the notional value
  - SlippageEstimator adds market impact noise
  - Latency sampled but not blocking (paper is always instant locally)
  - Position and balance tracked in memory, logged to SQLite via structlog
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from ep2_crypto.execution.venue import (
    OrderBookLevel,
    OrderBookSnapshot,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    PositionInfo,
    VenueAdapter,
    VenueType,
)

logger = structlog.get_logger(__name__)

BPS = 1e-4
TAKER_FEE_BPS = 4.0


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class PaperTrade:
    """A completed paper trade."""

    trade_id: str
    order_id: str
    side: OrderSide
    size: float  # BTC
    fill_price: float
    fee_usd: float
    slippage_bps: float
    notional_usd: float
    timestamp_ms: int
    pnl_usd: float = 0.0  # Realised PnL (set on position close)


@dataclass
class PaperPosition:
    """Current paper position state."""

    side: OrderSide | None = None
    size: float = 0.0
    entry_price: float = 0.0
    entry_notional: float = 0.0
    unrealised_pnl: float = 0.0

    def update_unrealised(self, mark_price: float) -> None:
        if self.side is None or self.size == 0.0:
            self.unrealised_pnl = 0.0
            return
        price_move = mark_price - self.entry_price
        if self.side == OrderSide.SELL:
            price_move = -price_move
        self.unrealised_pnl = price_move * self.size

    @property
    def is_open(self) -> bool:
        return self.size > 0.0


# ---------------------------------------------------------------------------
# Fill simulator: walks orderbook levels
# ---------------------------------------------------------------------------


@dataclass
class FillSimulator:
    """Walk the orderbook to simulate market order fills.

    For a BUY order we consume ask levels (ascending price).
    For a SELL order we consume bid levels (descending price).
    Returns (avg_fill_price, filled_size, unfilled_size).
    """

    taker_fee_bps: float = TAKER_FEE_BPS
    slippage_noise_bps: float = 0.5
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))

    def simulate(
        self,
        request: OrderRequest,
        snapshot: OrderBookSnapshot,
    ) -> tuple[float, float, float]:
        """Simulate fill against the provided orderbook snapshot.

        Returns:
            (avg_fill_price, filled_size, unfilled_size)
        """
        levels: list[OrderBookLevel] = (
            snapshot.asks if request.side == OrderSide.BUY else snapshot.bids
        )

        if not levels:
            return 0.0, 0.0, request.size

        remaining = request.size
        total_cost = 0.0
        total_filled = 0.0

        for level in levels:
            if remaining <= 0:
                break
            fill_at_level = min(remaining, level.size)
            total_cost += fill_at_level * level.price
            total_filled += fill_at_level
            remaining -= fill_at_level

        if total_filled == 0.0:
            return 0.0, 0.0, request.size

        avg_price = total_cost / total_filled
        # Add small noise representing residual market impact
        noise = self.rng.normal(0.0, self.slippage_noise_bps * BPS)
        if request.side == OrderSide.BUY:
            avg_price *= 1.0 + abs(noise)
        else:
            avg_price *= 1.0 - abs(noise)

        return avg_price, total_filled, remaining


# ---------------------------------------------------------------------------
# PaperExchange
# ---------------------------------------------------------------------------


class PaperExchange(VenueAdapter):
    """Paper trading exchange implementing VenueAdapter.

    Drop-in replacement for a live exchange. All methods are async-compatible
    but execute synchronously since no network IO is required.

    Usage:
        exchange = PaperExchange(initial_balance_usd=10_000.0)
        await exchange.connect()
        order = await exchange.place_order(request)
    """

    def __init__(
        self,
        initial_balance_usd: float = 10_000.0,
        taker_fee_bps: float = TAKER_FEE_BPS,
        slippage_noise_bps: float = 0.5,
        seed: int = 42,
    ) -> None:
        self._initial_balance = initial_balance_usd
        self._balance = initial_balance_usd
        self._position = PaperPosition()
        self._fill_sim = FillSimulator(
            taker_fee_bps=taker_fee_bps,
            slippage_noise_bps=slippage_noise_bps,
            rng=np.random.default_rng(seed),
        )
        self._orders: dict[str, OrderResult] = {}
        self._trades: list[PaperTrade] = []
        self._connected = False
        self._last_snapshot: OrderBookSnapshot | None = None
        self._last_price: float = 0.0

    # ------------------------------------------------------------------
    # VenueAdapter properties
    # ------------------------------------------------------------------

    @property
    def venue_type(self) -> VenueType:
        return VenueType.PAPER

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        self._connected = True
        logger.info(
            "paper_exchange_connected",
            initial_balance_usd=self._initial_balance,
        )

    async def disconnect(self) -> None:
        self._connected = False
        logger.info(
            "paper_exchange_disconnected",
            total_trades=len(self._trades),
            final_balance=self._balance,
        )

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def place_order(self, request: OrderRequest) -> OrderResult:
        """Simulate an order fill against the current orderbook snapshot."""
        order_id = str(uuid.uuid4())
        ts_ms = int(time.time() * 1000)

        if not self._connected:
            result = OrderResult(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                raw_response={"reason": "not_connected"},
            )
            self._orders[order_id] = result
            return result

        # Use last snapshot if available, otherwise fall back to a single-level
        # synthetic book using last_price
        snapshot = self._last_snapshot
        if snapshot is None or not snapshot.bids or not snapshot.asks:
            # No real orderbook — use a synthetic 1-level book
            if self._last_price > 0:
                synthetic_size = 1000.0  # large enough to fill any reasonable size
                snapshot = OrderBookSnapshot(
                    venue=VenueType.PAPER,
                    symbol="BTC/USDT",
                    bids=[OrderBookLevel(price=self._last_price * 0.9999, size=synthetic_size)],
                    asks=[OrderBookLevel(price=self._last_price * 1.0001, size=synthetic_size)],
                    timestamp_ms=ts_ms,
                )
            else:
                result = OrderResult(
                    order_id=order_id,
                    status=OrderStatus.REJECTED,
                    raw_response={"reason": "no_price_data"},
                )
                self._orders[order_id] = result
                logger.warning("paper_order_rejected_no_price", order_id=order_id)
                return result

        # Simulate fill
        avg_price, filled_size, unfilled = self._fill_sim.simulate(request, snapshot)

        if filled_size == 0.0:
            result = OrderResult(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                raw_response={"reason": "no_liquidity"},
            )
            self._orders[order_id] = result
            return result

        notional = avg_price * filled_size
        fee_usd = notional * self._fill_sim.taker_fee_bps * BPS
        slippage_bps = abs((avg_price - self._last_price) / max(self._last_price, 1e-9)) / BPS

        # Check sufficient balance
        cost = notional + fee_usd
        if request.side == OrderSide.BUY and cost > self._balance:
            result = OrderResult(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                raw_response={
                    "reason": "insufficient_balance",
                    "required": cost,
                    "available": self._balance,
                },
            )
            self._orders[order_id] = result
            logger.warning(
                "paper_order_rejected_balance",
                required=cost,
                available=self._balance,
            )
            return result

        # Apply trade to state
        realised_pnl = self._apply_trade(request, filled_size, avg_price, fee_usd)

        trade = PaperTrade(
            trade_id=str(uuid.uuid4()),
            order_id=order_id,
            side=request.side,
            size=filled_size,
            fill_price=avg_price,
            fee_usd=fee_usd,
            slippage_bps=slippage_bps,
            notional_usd=notional,
            timestamp_ms=ts_ms,
            pnl_usd=realised_pnl,
        )
        self._trades.append(trade)

        status = OrderStatus.FILLED if unfilled == 0.0 else OrderStatus.PARTIALLY_FILLED
        result = OrderResult(
            order_id=order_id,
            status=status,
            fill_price=avg_price,
            fill_quantity=filled_size,
            fee=fee_usd,
            fee_currency="USDT",
            venue=VenueType.PAPER,
        )
        self._orders[order_id] = result

        logger.info(
            "paper_order_filled",
            order_id=order_id,
            side=request.side.value,
            size=filled_size,
            avg_price=avg_price,
            fee_usd=round(fee_usd, 4),
            slippage_bps=round(slippage_bps, 3),
            realised_pnl=round(realised_pnl, 4),
            balance=round(self._balance, 2),
        )
        return result

    async def cancel_order(self, order_id: str) -> OrderResult:
        """Paper orders fill instantly — cancel is always a no-op."""
        if order_id in self._orders:
            result = self._orders[order_id]
            if result.is_terminal:
                return result
            # Mark as cancelled
            cancelled = OrderResult(
                order_id=order_id,
                status=OrderStatus.CANCELLED,
                fill_price=result.fill_price,
                fill_quantity=result.fill_quantity,
                fee=result.fee,
                fee_currency=result.fee_currency,
                venue=VenueType.PAPER,
            )
            self._orders[order_id] = cancelled
            return cancelled
        return OrderResult(
            order_id=order_id,
            status=OrderStatus.REJECTED,
            raw_response={"reason": "order_not_found"},
        )

    async def get_order(self, order_id: str) -> OrderResult:
        if order_id in self._orders:
            return self._orders[order_id]
        return OrderResult(
            order_id=order_id,
            status=OrderStatus.REJECTED,
            raw_response={"reason": "order_not_found"},
        )

    async def get_position(self) -> PositionInfo:
        if self._last_price > 0:
            self._position.update_unrealised(self._last_price)
        return PositionInfo(
            venue=VenueType.PAPER,
            symbol="BTC/USDT",
            side=self._position.side,
            size=self._position.size,
            entry_price=self._position.entry_price,
            unrealized_pnl=self._position.unrealised_pnl,
            cost_basis=self._position.entry_notional,
        )

    async def get_orderbook(self) -> OrderBookSnapshot:
        if self._last_snapshot is not None:
            return self._last_snapshot
        # Return empty book if no snapshot has been fed yet
        return OrderBookSnapshot(
            venue=VenueType.PAPER,
            symbol="BTC/USDT",
            bids=[],
            asks=[],
            timestamp_ms=int(time.time() * 1000),
        )

    async def get_balance(self) -> float:
        return self._balance

    async def health_check(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # State update methods (called by the data pipeline)
    # ------------------------------------------------------------------

    def update_orderbook(self, snapshot: OrderBookSnapshot) -> None:
        """Feed a fresh orderbook snapshot for fill simulation."""
        self._last_snapshot = snapshot
        if snapshot.mid_price is not None:
            self._last_price = snapshot.mid_price

    def update_price(self, price: float) -> None:
        """Update the last known price (used for unrealised PnL)."""
        self._last_price = price
        self._position.update_unrealised(price)

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    @property
    def trades(self) -> list[PaperTrade]:
        return list(self._trades)

    @property
    def total_pnl_usd(self) -> float:
        return sum(t.pnl_usd for t in self._trades)

    @property
    def total_fees_usd(self) -> float:
        return sum(t.fee_usd for t in self._trades)

    def get_summary(self) -> dict[str, Any]:
        """Return a snapshot of current paper trading state."""
        winning = [t for t in self._trades if t.pnl_usd > 0]
        losing = [t for t in self._trades if t.pnl_usd < 0]
        win_rate = len(winning) / max(len(self._trades), 1)
        return {
            "balance_usd": round(self._balance, 2),
            "initial_balance_usd": round(self._initial_balance, 2),
            "total_pnl_usd": round(self.total_pnl_usd, 4),
            "total_fees_usd": round(self.total_fees_usd, 4),
            "total_trades": len(self._trades),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": round(win_rate, 4),
            "position_open": self._position.is_open,
            "position_side": self._position.side.value if self._position.side else None,
            "position_size": round(self._position.size, 6),
            "unrealised_pnl_usd": round(self._position.unrealised_pnl, 4),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_trade(
        self,
        request: OrderRequest,
        filled_size: float,
        avg_price: float,
        fee_usd: float,
    ) -> float:
        """Update balance and position; return realised PnL."""
        notional = avg_price * filled_size
        realised_pnl = 0.0

        if request.side == OrderSide.BUY:
            if self._position.side == OrderSide.SELL and self._position.size > 0:
                # Closing a short (partially or fully)
                close_size = min(filled_size, self._position.size)
                realised_pnl = (self._position.entry_price - avg_price) * close_size
                self._position.size -= close_size
                self._balance += realised_pnl - fee_usd
                remaining_buy = filled_size - close_size
                if remaining_buy > 0:
                    # Flip to long
                    self._position.side = OrderSide.BUY
                    self._position.size = remaining_buy
                    self._position.entry_price = avg_price
                    self._position.entry_notional = avg_price * remaining_buy
                    self._balance -= avg_price * remaining_buy
                elif self._position.size == 0:
                    self._position.side = None
                    self._position.entry_price = 0.0
                    self._position.entry_notional = 0.0
            else:
                # Opening or adding to a long
                if self._position.side is None or self._position.size == 0:
                    self._position.side = OrderSide.BUY
                    self._position.size = filled_size
                    self._position.entry_price = avg_price
                    self._position.entry_notional = notional
                else:
                    # Average up
                    total_size = self._position.size + filled_size
                    total_cost = self._position.entry_price * self._position.size + notional
                    self._position.entry_price = total_cost / total_size
                    self._position.entry_notional = total_cost
                    self._position.size = total_size
                self._balance -= notional + fee_usd

        else:  # SELL
            if self._position.side == OrderSide.BUY and self._position.size > 0:
                # Closing a long (partially or fully)
                close_size = min(filled_size, self._position.size)
                realised_pnl = (avg_price - self._position.entry_price) * close_size
                self._position.size -= close_size
                self._balance += notional - fee_usd
                remaining_sell = filled_size - close_size
                if remaining_sell > 0:
                    # Flip to short
                    self._position.side = OrderSide.SELL
                    self._position.size = remaining_sell
                    self._position.entry_price = avg_price
                    self._position.entry_notional = avg_price * remaining_sell
                elif self._position.size == 0:
                    self._position.side = None
                    self._position.entry_price = 0.0
                    self._position.entry_notional = 0.0
            else:
                # Opening or adding to a short
                if self._position.side is None or self._position.size == 0:
                    self._position.side = OrderSide.SELL
                    self._position.size = filled_size
                    self._position.entry_price = avg_price
                    self._position.entry_notional = notional
                else:
                    total_size = self._position.size + filled_size
                    total_cost = self._position.entry_price * self._position.size + notional
                    self._position.entry_price = total_cost / total_size
                    self._position.entry_notional = total_cost
                    self._position.size = total_size
                self._balance -= fee_usd  # Short doesn't cost cash, just fees

        return realised_pnl
