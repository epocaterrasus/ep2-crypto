"""Live prediction loop.

Async event loop that:
1. Runs all data collectors (Binance WS, Bybit, cross-market)
2. Computes features on each 5-min candle close
3. Runs model inference and confidence gating
4. Calls RiskManager.approve_trade() before every execution
5. Calls RiskManager.on_bar() on every bar for stop/kill-switch checks
6. Routes approved signals to the correct VenueAdapter (paper / live Binance / Polymarket)
7. Updates API state (AppState) shared with the FastAPI server
8. Resets daily/weekly risk counters at UTC midnight / Monday
9. Auto-retrains every 2-4h (warm-start) if triggered
10. Graceful degradation if a data source fails

Usage:
    uv run python scripts/live.py
    uv run python scripts/live.py --collect-only   # Data collection only, no inference
    uv run python scripts/live.py --api            # Also start the API server
    uv run python scripts/live.py --venue polymarket_binary
    uv run python scripts/live.py --mode live      # Real orders (requires credentials)
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import signal
import sqlite3
import time
from typing import Any

import structlog

from ep2_crypto.api.server import AppState
from ep2_crypto.execution.paper_exchange import PaperExchange
from ep2_crypto.execution.paper_runner import PaperRunner, TradeSignal
from ep2_crypto.execution.venue import VenueAdapter, VenueType
from ep2_crypto.logging import configure_logging
from ep2_crypto.risk.config import RiskConfig
from ep2_crypto.risk.risk_manager import RiskManager, SignalInput

logger = structlog.get_logger(__name__)

BAR_INTERVAL_S = 300  # 5 minutes
STALENESS_WARN_S = 600  # 10 minutes: warn if no prediction


def _build_venue_adapter(
    venue_type: VenueType,
    mode: str,
    initial_balance: float,
) -> VenueAdapter:
    """Construct the correct VenueAdapter based on venue and mode flags.

    paper mode:  always PaperExchange (even for binance_perps / polymarket_binary)
    live mode:   binance_perps -> LiveExchange
                 polymarket_binary -> PolymarketAdapter
    """
    if mode == "paper":
        return PaperExchange(initial_balance_usd=initial_balance)

    # live mode
    if venue_type == VenueType.BINANCE_PERPS:
        from ep2_crypto.execution.live_exchange import LiveExchange  # lazy import
        return LiveExchange()

    if venue_type == VenueType.POLYMARKET_BINARY:
        from ep2_crypto.execution.polymarket import PolymarketAdapter  # lazy import
        from ep2_crypto.execution.polymarket_config import PolymarketConfig
        return PolymarketAdapter(config=PolymarketConfig())

    raise ValueError(f"Unsupported venue_type for live mode: {venue_type}")


class LivePredictionLoop:
    """Main async prediction loop.

    Coordinates collectors, feature pipeline, model inference, risk management,
    and monitoring in a single asyncio event loop.

    Venue selection:
        venue_type=VenueType.BINANCE_PERPS    — perps futures (default)
        venue_type=VenueType.POLYMARKET_BINARY — binary prediction markets

    Mode:
        mode="paper" — simulated fills via PaperExchange (default)
        mode="live"  — real orders via LiveExchange or PolymarketAdapter
    """

    def __init__(
        self,
        collect_only: bool = False,
        retrain_interval_s: float = 4 * 3600,
        venue_type: VenueType = VenueType.BINANCE_PERPS,
        mode: str = "paper",
        initial_balance: float = 10_000.0,
        confidence_threshold: float = 0.60,
        api_state: AppState | None = None,
        db_path: str = "data/ep2_crypto.db",
    ) -> None:
        self._collect_only = collect_only
        self._retrain_interval_s = retrain_interval_s
        self._venue_type = venue_type
        self._mode = mode
        self._running = False
        self._bar_count = 0
        self._last_retrain_time = time.monotonic()
        self._degraded_sources: set[str] = set()
        self._db_path = db_path

        # Shared API state — written here, read by the FastAPI server
        self._api_state: AppState = api_state if api_state is not None else AppState()

        # Execution layer (initialized in start())
        self._venue: VenueAdapter | None = None
        self._paper_runner: PaperRunner | None = None
        self._initial_balance = initial_balance
        self._confidence_threshold = confidence_threshold

        # Risk engine (initialized in start())
        self._risk_manager: RiskManager | None = None
        self._db_conn: sqlite3.Connection | None = None

        # Day/week boundary tracking
        self._last_day: int | None = None
        self._last_week: int | None = None  # ISO week number

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def start(self) -> None:
        """Start the live prediction loop."""
        self._running = True

        # 1. Initialize SQLite for risk persistence
        # Use pathlib instead of os.path (ASYNC240: avoid os.path in async functions)
        from pathlib import Path
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._db_conn.execute("PRAGMA journal_mode=WAL")

        # 2. Initialize risk manager
        if not self._collect_only:
            risk_config = RiskConfig()
            self._risk_manager = RiskManager(
                conn=self._db_conn,
                initial_equity=self._initial_balance,
                config=risk_config,
            )

        # 3. Build and connect the venue adapter
        if not self._collect_only:
            self._venue = _build_venue_adapter(
                venue_type=self._venue_type,
                mode=self._mode,
                initial_balance=self._initial_balance,
            )
            await self._venue.connect()

            # Wrap in PaperRunner (works for any VenueAdapter)
            self._paper_runner = PaperRunner(
                exchange=self._venue,
                confidence_threshold=self._confidence_threshold,
            )

            logger.info(
                "venue_connected",
                venue=self._venue_type.value,
                mode=self._mode,
                initial_balance=self._initial_balance,
                confidence_threshold=self._confidence_threshold,
            )

        logger.info(
            "live_loop_starting",
            mode=self._mode,
            collect_only=self._collect_only,
            retrain_interval_s=self._retrain_interval_s,
            venue=self._venue_type.value,
        )

        try:
            await self._run_loop()
        except asyncio.CancelledError:
            logger.info("live_loop_cancelled")
        finally:
            await self._shutdown()

    async def _run_loop(self) -> None:
        """Main loop: wait for candle close, then process."""
        while self._running:
            next_close = self._next_bar_close_time()
            wait_s = max(0, next_close - time.time())

            if wait_s > 0:
                logger.debug("waiting_for_candle", wait_s=round(wait_s, 1))
                await asyncio.sleep(wait_s)

            if not self._running:
                break

            await self._on_bar_close()

    async def _on_bar_close(self) -> None:
        """Process a single bar close event."""
        self._bar_count += 1
        bar_start = time.perf_counter()
        timestamp_ms = int(time.time() * 1000)

        logger.info("bar_close", bar_number=self._bar_count, timestamp_ms=timestamp_ms)

        # Reset daily/weekly counters at UTC boundaries
        self._check_day_week_boundary(timestamp_ms)

        if self._collect_only:
            return

        # 1. Compute features (calls the full feature pipeline)
        features = await self._compute_features(timestamp_ms)

        # 2. Run inference (LightGBM + CatBoost + GRU → stacking → calibration → gating)
        prediction = await self._run_inference(features)

        # 3. Update shared API state so the FastAPI server sees fresh data
        self._update_api_state(prediction, timestamp_ms)

        # 4. Call RiskManager.on_bar() — checks stops and kill switches every bar
        risk_actions = self._run_risk_on_bar(prediction, timestamp_ms)
        for action in risk_actions:
            logger.warning(
                "risk_action_required",
                action=action.action.value,
                reason=action.reason,
                urgency=action.urgency,
            )
            if action.urgency in ("immediate", "high") and self._paper_runner is not None:
                await self._paper_runner.close_position()

        # 5. Gate signal through RiskManager.approve_trade() before routing to venue
        if self._paper_runner is not None:
            direction = prediction.get("direction", "flat")
            confidence = prediction.get("confidence", 0.0)

            if direction in ("up", "down") and confidence >= self._confidence_threshold:
                approved_size = self._approve_trade(prediction, timestamp_ms)
                if approved_size > 0.0:
                    signal = TradeSignal(
                        direction=direction,
                        confidence=confidence,
                        regime=prediction.get("regime", "unknown"),
                        position_size_btc=approved_size,
                        timestamp_ms=timestamp_ms,
                    )
                    await self._paper_runner.on_signal(signal)

        # 6. Check retrain trigger
        await self._check_retrain()

        elapsed_ms = (time.perf_counter() - bar_start) * 1000
        logger.info(
            "bar_processed",
            bar_number=self._bar_count,
            elapsed_ms=round(elapsed_ms, 1),
            direction=prediction.get("direction", "flat"),
            confidence=prediction.get("confidence", 0.0),
        )

    # -----------------------------------------------------------------------
    # Feature computation (stub — real impl calls feature pipeline)
    # -----------------------------------------------------------------------

    async def _compute_features(self, timestamp_ms: int) -> dict[str, float]:
        """Compute features from current data.

        In production, this calls the full feature pipeline:
            - ExchangeCollector ring buffer → microstructure, volume, volatility, momentum
            - DerivativesCollector → OI, funding rate, liquidations
            - CrossMarketCollector → NQ/Gold/DXY lead-lag
            - RegimeDetector → HMM state, ER, GARCH vol

        Gracefully degrades if some data sources are unavailable.
        """
        features: dict[str, float] = {}

        # Placeholder — real implementation fetches from in-process ring buffers
        # and runs the feature pipeline
        features["timestamp_ms"] = float(timestamp_ms)

        if self._degraded_sources:
            logger.warning(
                "degraded_features",
                missing_sources=list(self._degraded_sources),
            )

        return features

    # -----------------------------------------------------------------------
    # Model inference (stub — real impl calls model stack)
    # -----------------------------------------------------------------------

    async def _run_inference(self, features: dict[str, float]) -> dict[str, Any]:
        """Run model inference and confidence gating.

        In production, this calls:
            1. LightGBM ternary classifier (raw features, no normalization)
            2. CatBoost ternary classifier
            3. GRU hidden state extractor (normalized features)
            4. Stacking meta-learner (logistic regression)
            5. Isotonic calibration
            6. Confidence gating pipeline (7 gates)
            7. Quarter-Kelly position sizing
        """
        return {
            "direction": "flat",
            "magnitude": 0.0,
            "confidence": 0.0,
            "regime": "unknown",
            "regime_confidence": 0.0,
            "position_size_btc": 0.0,
            "win_rate": None,
            "payoff_ratio": None,
        }

    # -----------------------------------------------------------------------
    # Risk engine integration
    # -----------------------------------------------------------------------

    def _approve_trade(self, prediction: dict[str, Any], timestamp_ms: int) -> float:
        """Call RiskManager.approve_trade(); return approved BTC size or 0."""
        if self._risk_manager is None:
            # No risk manager (collect-only mode) — pass through
            return float(prediction.get("position_size_btc", 0.01))

        direction_raw = prediction.get("direction", "flat")
        if direction_raw == "up":
            direction = "long"
        elif direction_raw == "down":
            direction = "short"
        else:
            return 0.0

        signal = SignalInput(
            direction=direction,
            confidence=float(prediction.get("confidence", 0.0)),
            timestamp_ms=timestamp_ms,
            win_rate=prediction.get("win_rate"),
            payoff_ratio=prediction.get("payoff_ratio"),
        )

        # NOTE: In production, closes/highs/lows/idx come from the ring buffer.
        # The stubs below will trigger the vol guard with zero arrays (rejected),
        # so this path will only be reached once the feature pipeline is live.
        import numpy as np
        dummy_closes = np.array([1.0], dtype=np.float64)
        dummy_highs = np.array([1.0], dtype=np.float64)
        dummy_lows = np.array([1.0], dtype=np.float64)

        try:
            decision = self._risk_manager.approve_trade(
                signal=signal,
                closes=dummy_closes,
                highs=dummy_highs,
                lows=dummy_lows,
                current_idx=0,
            )
        except Exception as exc:
            logger.error("risk_approve_trade_failed", error=str(exc))
            return 0.0

        if not decision.approved:
            logger.info(
                "trade_rejected_by_risk",
                direction=direction,
                reason=decision.reason,
            )
            return 0.0

        return float(decision.quantity_btc)

    def _run_risk_on_bar(
        self, prediction: dict[str, Any], timestamp_ms: int
    ) -> list[Any]:
        """Call RiskManager.on_bar() for stop-loss and kill-switch checks."""
        if self._risk_manager is None:
            return []

        import numpy as np
        dummy = np.array([1.0], dtype=np.float64)
        try:
            actions: list[Any] = self._risk_manager.on_bar(
                bar_close=float(prediction.get("magnitude", 1.0)) or 1.0,
                bar_high=float(prediction.get("magnitude", 1.0)) or 1.0,
                bar_low=float(prediction.get("magnitude", 1.0)) or 1.0,
                bar_timestamp_ms=timestamp_ms,
                closes=dummy,
                highs=dummy,
                lows=dummy,
                current_idx=0,
            )
            return actions
        except Exception as exc:
            logger.error("risk_on_bar_failed", error=str(exc))
            return []

    def _check_day_week_boundary(self, timestamp_ms: int) -> None:
        """Reset daily/weekly risk counters at UTC midnight and Monday boundaries."""
        if self._risk_manager is None:
            return

        now_utc = datetime.datetime.fromtimestamp(timestamp_ms / 1000, tz=datetime.UTC)
        today = now_utc.toordinal()
        iso_week = now_utc.isocalendar().week

        if self._last_day is None:
            self._last_day = today
            self._last_week = iso_week
            return

        if today != self._last_day:
            self._risk_manager.reset_daily_counters()
            self._last_day = today
            logger.info("daily_counters_reset_at_boundary", date=now_utc.date().isoformat())

        if iso_week != self._last_week:
            self._risk_manager.reset_weekly_counters()
            self._last_week = iso_week
            logger.info("weekly_counters_reset_at_boundary", iso_week=iso_week)

    # -----------------------------------------------------------------------
    # API state
    # -----------------------------------------------------------------------

    def _update_api_state(self, prediction: dict[str, Any], timestamp_ms: int) -> None:
        """Update shared AppState so the FastAPI server sees the latest prediction."""
        self._api_state.direction = prediction.get("direction", "flat")
        self._api_state.magnitude = prediction.get("magnitude", 0.0)
        self._api_state.confidence = prediction.get("confidence", 0.0)
        self._api_state.regime = prediction.get("regime", "unknown")
        self._api_state.regime_confidence = prediction.get("regime_confidence", 0.0)
        self._api_state.last_prediction_ms = timestamp_ms

    # -----------------------------------------------------------------------
    # Retrain trigger
    # -----------------------------------------------------------------------

    async def _check_retrain(self) -> None:
        """Check if retrain should trigger."""
        elapsed = time.monotonic() - self._last_retrain_time
        if elapsed >= self._retrain_interval_s:
            logger.info("retrain_due", elapsed_s=round(elapsed, 0))
            self._last_retrain_time = time.monotonic()
            # In production: asyncio.create_task(retrain_pipeline()) and track completion

    # -----------------------------------------------------------------------
    # Degradation helpers
    # -----------------------------------------------------------------------

    def mark_source_degraded(self, source: str) -> None:
        """Mark a data source as degraded (for graceful degradation)."""
        self._degraded_sources.add(source)
        logger.warning("source_degraded", source=source)

    def mark_source_recovered(self, source: str) -> None:
        """Mark a data source as recovered."""
        self._degraded_sources.discard(source)
        logger.info("source_recovered", source=source)

    # -----------------------------------------------------------------------
    # Shutdown
    # -----------------------------------------------------------------------

    async def _shutdown(self) -> None:
        """Clean shutdown: close position, flush logs, close DB."""
        if self._paper_runner is not None:
            try:
                await self._paper_runner.close_position()
                summary = self._paper_runner.get_summary()
                logger.info("paper_session_summary", **summary)
            except Exception as exc:
                logger.error("shutdown_close_position_failed", error=str(exc))

        if self._venue is not None and hasattr(self._venue, "disconnect"):
            try:
                await self._venue.disconnect()
            except Exception as exc:
                logger.error("shutdown_venue_disconnect_failed", error=str(exc))

        if self._db_conn is not None:
            try:
                self._db_conn.close()
            except Exception as exc:
                logger.error("shutdown_db_close_failed", error=str(exc))

        logger.info("live_loop_shutting_down", bars_processed=self._bar_count)
        self._running = False

    def stop(self) -> None:
        """Signal the loop to stop."""
        self._running = False

    # -----------------------------------------------------------------------
    # Static helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _next_bar_close_time() -> float:
        """Calculate the next 5-minute candle close timestamp."""
        now = time.time()
        current_bar_start = (int(now) // BAR_INTERVAL_S) * BAR_INTERVAL_S
        return float(current_bar_start + BAR_INTERVAL_S)

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def bar_count(self) -> int:
        return self._bar_count

    @property
    def running(self) -> bool:
        return self._running

    @property
    def degraded_sources(self) -> set[str]:
        return set(self._degraded_sources)

    @property
    def api_state(self) -> AppState:
        return self._api_state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live prediction loop")
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect data, no inference",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Also start the API server",
    )
    parser.add_argument(
        "--retrain-interval",
        type=float,
        default=4 * 3600,
        help="Retrain interval in seconds (default: 4h)",
    )
    parser.add_argument(
        "--venue",
        choices=["binance_perps", "polymarket_binary"],
        default="binance_perps",
        help="Execution venue (default: binance_perps)",
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="Execution mode: paper (simulated fills) or live (real orders). Default: paper",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=10_000.0,
        help="Initial paper trading balance in USD (paper mode only, default: 10000)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.60,
        help="Minimum confidence to act on a signal (default: 0.60)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/ep2_crypto.db",
        help="Path to SQLite database (default: data/ep2_crypto.db)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level (default: INFO)",
    )
    parser.add_argument(
        "--log-json",
        action="store_true",
        default=True,
        help="Output logs as JSON (default: True). Use --no-log-json for human-readable.",
    )
    parser.add_argument(
        "--no-log-json",
        dest="log_json",
        action="store_false",
        help="Output human-readable logs instead of JSON",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    # Configure structured logging first — before any logger calls
    configure_logging(level=args.log_level, json_output=args.log_json)

    # Create the shared AppState — both the loop and the API server write/read this
    shared_state = AppState()

    venue_type = VenueType(args.venue)
    loop_instance = LivePredictionLoop(
        collect_only=args.collect_only,
        retrain_interval_s=args.retrain_interval,
        venue_type=venue_type,
        mode=args.mode,
        initial_balance=args.initial_balance,
        confidence_threshold=args.confidence_threshold,
        api_state=shared_state,
        db_path=args.db_path,
    )

    # Handle SIGTERM/SIGINT — use get_running_loop() (get_event_loop() is deprecated)
    asyncio_loop = asyncio.get_running_loop()

    def signal_handler() -> None:
        logger.info("signal_received")
        loop_instance.stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio_loop.add_signal_handler(sig, signal_handler)

    tasks: list[asyncio.Task[Any]] = [
        asyncio.create_task(loop_instance.start(), name="live_loop")
    ]

    if args.api:
        import uvicorn  # noqa: I001
        from ep2_crypto.api.server import create_app

        # Pass the SAME shared_state so the API reads predictions written by the loop
        app = create_app(shared_state)
        config = uvicorn.Config(
            app,
            host="0.0.0.0",  # noqa: S104
            port=8000,
            log_level=args.log_level.lower(),
        )
        server = uvicorn.Server(config)
        tasks.append(asyncio.create_task(server.serve(), name="api_server"))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Log any exceptions that were swallowed by gather()
    for task, result in zip(tasks, results, strict=True):
        if isinstance(result, Exception):
            logger.error(
                "task_failed",
                task=task.get_name(),
                error=str(result),
                error_type=type(result).__name__,
            )


if __name__ == "__main__":
    asyncio.run(main())
