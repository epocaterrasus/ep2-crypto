"""Orchestrator for managing multiple data collectors.

Handles lifecycle management, signal handling, health aggregation,
and graceful shutdown of all collectors.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from dataclasses import dataclass, field
from typing import Any

import structlog

from ep2_crypto.ingest.base import BaseCollector, HealthStatus  # noqa: TC001

logger = structlog.get_logger(__name__)


@dataclass
class OrchestratorHealth:
    """Aggregated health status for all collectors."""

    healthy: bool
    collectors: list[HealthStatus] = field(default_factory=list)
    total_messages: int = 0
    unhealthy_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "total_messages": self.total_messages,
            "unhealthy_count": self.unhealthy_count,
            "collectors": [
                {
                    "name": c.name,
                    "healthy": c.healthy,
                    "state": c.state.value,
                    "messages_received": c.messages_received,
                    "reconnect_count": c.reconnect_count,
                    "error": c.error,
                }
                for c in self.collectors
            ],
        }


class Orchestrator:
    """Manages lifecycle of multiple data collectors.

    Provides:
    - Add/remove collectors
    - Start all / stop all
    - SIGTERM/SIGINT graceful shutdown
    - Aggregated health checks
    - Async context manager pattern
    """

    def __init__(self) -> None:
        self._collectors: dict[str, BaseCollector] = {}
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._log = logger.bind(component="orchestrator")

    @property
    def running(self) -> bool:
        return self._running

    @property
    def collector_names(self) -> list[str]:
        return list(self._collectors.keys())

    async def __aenter__(self) -> Orchestrator:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.shutdown()

    def add_collector(self, collector: BaseCollector) -> None:
        """Register a collector. Must be called before run()."""
        if self._running:
            msg = "Cannot add collectors while orchestrator is running"
            raise RuntimeError(msg)
        if collector.name in self._collectors:
            msg = f"Collector already registered: {collector.name}"
            raise ValueError(msg)

        self._collectors[collector.name] = collector
        self._log.info("collector_registered", collector=collector.name)

    def remove_collector(self, name: str) -> None:
        """Unregister a collector. Must be called before run()."""
        if self._running:
            msg = "Cannot remove collectors while orchestrator is running"
            raise RuntimeError(msg)
        if name not in self._collectors:
            msg = f"Collector not registered: {name}"
            raise KeyError(msg)

        del self._collectors[name]
        self._log.info("collector_unregistered", collector=name)

    async def run(self) -> None:
        """Start all collectors and wait for shutdown signal.

        Installs SIGTERM/SIGINT handlers. Blocks until shutdown() is called
        or a signal is received.
        """
        if not self._collectors:
            self._log.warning("no_collectors_registered")
            return

        self._running = True
        self._shutdown_event.clear()

        # Install signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._signal_handler, sig)

        self._log.info(
            "orchestrator_starting",
            collectors=list(self._collectors.keys()),
        )

        try:
            # Start all collectors
            await self._start_all()

            # Wait for shutdown signal
            await self._shutdown_event.wait()

        finally:
            await self._stop_all()
            self._running = False

            # Remove signal handlers
            for sig in (signal.SIGTERM, signal.SIGINT):
                with contextlib.suppress(ValueError, RuntimeError):
                    loop.remove_signal_handler(sig)

            self._log.info("orchestrator_stopped")

    async def shutdown(self) -> None:
        """Trigger graceful shutdown."""
        self._log.info("shutdown_requested")
        self._shutdown_event.set()

    def health_check(self) -> OrchestratorHealth:
        """Aggregate health from all collectors."""
        statuses = [c.health_check() for c in self._collectors.values()]
        unhealthy = [s for s in statuses if not s.healthy]
        total_msgs = sum(s.messages_received for s in statuses)

        return OrchestratorHealth(
            healthy=len(unhealthy) == 0 and self._running,
            collectors=statuses,
            total_messages=total_msgs,
            unhealthy_count=len(unhealthy),
        )

    def _signal_handler(self, sig: signal.Signals) -> None:
        self._log.info("signal_received", signal=sig.name)
        self._shutdown_event.set()

    async def _start_all(self) -> None:
        """Start all registered collectors concurrently."""
        tasks = [c.start() for c in self._collectors.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for collector, result in zip(self._collectors.values(), results, strict=True):
            if isinstance(result, Exception):
                self._log.error(
                    "collector_start_failed",
                    collector=collector.name,
                    error=str(result),
                )

    async def _stop_all(self) -> None:
        """Stop all registered collectors concurrently."""
        tasks = [c.stop() for c in self._collectors.values()]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for collector, result in zip(self._collectors.values(), results, strict=True):
            if isinstance(result, Exception):
                self._log.error(
                    "collector_stop_failed",
                    collector=collector.name,
                    error=str(result),
                )
