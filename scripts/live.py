"""Live prediction loop.

Async event loop that:
1. Runs all data collectors (Binance WS, Bybit, cross-market)
2. Computes features on each 5-min candle close
3. Runs model inference and confidence gating
4. Updates API state for the prediction server
5. Logs every bar and trade to the flight recorder
6. Auto-retrains every 2-4h (warm-start) if triggered
7. Graceful degradation if a data source fails

Usage:
    uv run python scripts/live.py
    uv run python scripts/live.py --collect-only   # Data collection only, no inference
    uv run python scripts/live.py --api            # Also start the API server
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import time
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

BAR_INTERVAL_S = 300  # 5 minutes


class LivePredictionLoop:
    """Main async prediction loop.

    Coordinates collectors, feature pipeline, model inference,
    and monitoring in a single asyncio event loop.
    """

    def __init__(
        self,
        collect_only: bool = False,
        retrain_interval_s: float = 4 * 3600,
    ) -> None:
        self._collect_only = collect_only
        self._retrain_interval_s = retrain_interval_s
        self._running = False
        self._bar_count = 0
        self._last_retrain_time = time.monotonic()
        self._degraded_sources: set[str] = set()

        # Placeholder state — in production, these are initialized
        # with real model/pipeline instances
        self._api_state: dict[str, Any] = {}

    async def start(self) -> None:
        """Start the live prediction loop."""
        self._running = True
        logger.info(
            "live_loop_starting",
            collect_only=self._collect_only,
            retrain_interval_s=self._retrain_interval_s,
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

        if self._collect_only:
            return

        # 1. Compute features (would call feature pipeline)
        features = await self._compute_features(timestamp_ms)

        # 2. Run inference (would call model stack)
        prediction = await self._run_inference(features)

        # 3. Update API state
        self._update_api_state(prediction, timestamp_ms)

        # 4. Check retrain trigger
        await self._check_retrain()

        elapsed_ms = (time.perf_counter() - bar_start) * 1000
        logger.info(
            "bar_processed",
            bar_number=self._bar_count,
            elapsed_ms=round(elapsed_ms, 1),
            direction=prediction.get("direction", "flat"),
            confidence=prediction.get("confidence", 0.0),
        )

    async def _compute_features(
        self, timestamp_ms: int
    ) -> dict[str, float]:
        """Compute features from current data.

        In production, this calls the full feature pipeline.
        Gracefully degrades if some data sources are unavailable.
        """
        features: dict[str, float] = {}

        # Placeholder — real implementation fetches from data stores
        # and runs the feature pipeline
        features["timestamp_ms"] = float(timestamp_ms)

        if self._degraded_sources:
            logger.warning(
                "degraded_features",
                missing_sources=list(self._degraded_sources),
            )

        return features

    async def _run_inference(
        self, features: dict[str, float]
    ) -> dict[str, Any]:
        """Run model inference and confidence gating.

        In production, this calls LightGBM/CatBoost/GRU stack,
        stacking ensemble, calibration, and confidence gates.
        """
        return {
            "direction": "flat",
            "magnitude": 0.0,
            "confidence": 0.0,
            "regime": "unknown",
            "regime_confidence": 0.0,
        }

    def _update_api_state(
        self, prediction: dict[str, Any], timestamp_ms: int
    ) -> None:
        """Update shared API state with latest prediction."""
        self._api_state.update({
            "direction": prediction.get("direction", "flat"),
            "magnitude": prediction.get("magnitude", 0.0),
            "confidence": prediction.get("confidence", 0.0),
            "regime": prediction.get("regime", "unknown"),
            "last_prediction_ms": timestamp_ms,
        })

    async def _check_retrain(self) -> None:
        """Check if retrain should trigger."""
        elapsed = time.monotonic() - self._last_retrain_time
        if elapsed >= self._retrain_interval_s:
            logger.info("retrain_due", elapsed_s=round(elapsed, 0))
            self._last_retrain_time = time.monotonic()
            # In production: trigger retrain pipeline

    def mark_source_degraded(self, source: str) -> None:
        """Mark a data source as degraded (for graceful degradation)."""
        self._degraded_sources.add(source)
        logger.warning("source_degraded", source=source)

    def mark_source_recovered(self, source: str) -> None:
        """Mark a data source as recovered."""
        self._degraded_sources.discard(source)
        logger.info("source_recovered", source=source)

    async def _shutdown(self) -> None:
        """Clean shutdown: stop collectors, flush logs."""
        logger.info("live_loop_shutting_down", bars_processed=self._bar_count)
        self._running = False

    def stop(self) -> None:
        """Signal the loop to stop."""
        self._running = False

    @staticmethod
    def _next_bar_close_time() -> float:
        """Calculate the next 5-minute candle close timestamp."""
        now = time.time()
        current_bar_start = (int(now) // BAR_INTERVAL_S) * BAR_INTERVAL_S
        return float(current_bar_start + BAR_INTERVAL_S)

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
    def api_state(self) -> dict[str, Any]:
        return dict(self._api_state)


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
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    loop_instance = LivePredictionLoop(
        collect_only=args.collect_only,
        retrain_interval_s=args.retrain_interval,
    )

    # Handle SIGTERM/SIGINT
    def signal_handler() -> None:
        logger.info("signal_received")
        loop_instance.stop()

    asyncio_loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio_loop.add_signal_handler(sig, signal_handler)

    tasks = [asyncio.create_task(loop_instance.start())]

    if args.api:
        import uvicorn

        from ep2_crypto.api.server import AppState, create_app

        state = AppState()
        app = create_app(state)
        config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")  # noqa: S104
        server = uvicorn.Server(config)
        tasks.append(asyncio.create_task(server.serve()))

    await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
