"""Base collector interface for all data ingestion sources.

All collectors implement this ABC with async context manager pattern,
health checking, and structured lifecycle management.
"""

from __future__ import annotations

import asyncio
import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class CollectorState(Enum):
    """Lifecycle states for a collector."""

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    RECONNECTING = "reconnecting"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class HealthStatus:
    """Health check result for a collector."""

    name: str
    healthy: bool
    state: CollectorState
    last_message_ts: float | None = None
    messages_received: int = 0
    reconnect_count: int = 0
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class BaseCollector(ABC):
    """Abstract base for all data collectors.

    Provides async context manager pattern, exponential backoff reconnection,
    and health status tracking. Subclasses implement _connect, _disconnect,
    and _run_loop.
    """

    def __init__(
        self,
        name: str,
        *,
        reconnect_delay_s: float = 1.0,
        max_reconnect_delay_s: float = 60.0,
        max_reconnect_attempts: int = 0,
    ) -> None:
        self.name = name
        self._state = CollectorState.IDLE
        self._reconnect_delay_s = reconnect_delay_s
        self._max_reconnect_delay_s = max_reconnect_delay_s
        self._max_reconnect_attempts = max_reconnect_attempts  # 0 = unlimited

        self._messages_received: int = 0
        self._last_message_ts: float | None = None
        self._reconnect_count: int = 0
        self._last_error: str | None = None

        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._log = logger.bind(collector=name)

    @property
    def state(self) -> CollectorState:
        return self._state

    async def __aenter__(self) -> BaseCollector:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    async def start(self) -> None:
        """Start the collector's run loop as a background task."""
        if self._state in (CollectorState.RUNNING, CollectorState.STARTING):
            self._log.warning("collector_already_running")
            return

        self._stop_event.clear()
        self._state = CollectorState.STARTING
        self._task = asyncio.create_task(self._run_with_reconnect())
        self._log.info("collector_started")

    async def stop(self) -> None:
        """Signal the collector to stop and wait for cleanup."""
        if self._state in (CollectorState.STOPPED, CollectorState.STOPPING):
            return

        self._state = CollectorState.STOPPING
        self._stop_event.set()
        self._log.info("collector_stopping")

        if self._task is not None and not self._task.done():
            try:
                await asyncio.wait_for(self._task, timeout=10.0)
            except TimeoutError:
                self._log.warning("collector_stop_timeout", timeout_s=10.0)
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task

        try:
            await self._disconnect()
        except Exception:
            self._log.exception("disconnect_error")

        self._state = CollectorState.STOPPED
        self._log.info("collector_stopped")

    def health_check(self) -> HealthStatus:
        """Return current health status."""
        return HealthStatus(
            name=self.name,
            healthy=self._state == CollectorState.RUNNING,
            state=self._state,
            last_message_ts=self._last_message_ts,
            messages_received=self._messages_received,
            reconnect_count=self._reconnect_count,
            error=self._last_error,
        )

    def _record_message(self, timestamp: float) -> None:
        """Record that a message was received (call from subclass)."""
        self._messages_received += 1
        self._last_message_ts = timestamp

    async def _run_with_reconnect(self) -> None:
        """Run loop with exponential backoff reconnection."""
        current_delay = self._reconnect_delay_s
        attempts = 0

        while not self._stop_event.is_set():
            try:
                await self._connect()
                self._state = CollectorState.RUNNING
                self._last_error = None
                current_delay = self._reconnect_delay_s  # reset on success
                attempts = 0
                self._log.info("collector_connected")

                await self._run_loop()

            except asyncio.CancelledError:
                self._log.info("collector_cancelled")
                return

            except Exception as exc:
                self._last_error = str(exc)
                self._log.error(
                    "collector_error",
                    error=str(exc),
                    reconnect_count=self._reconnect_count,
                )

                try:
                    await self._disconnect()
                except Exception:
                    self._log.exception("disconnect_error_during_reconnect")

            if self._stop_event.is_set():
                break

            # Check max attempts
            attempts += 1
            if self._max_reconnect_attempts > 0 and attempts >= self._max_reconnect_attempts:
                self._state = CollectorState.ERROR
                self._log.error(
                    "max_reconnect_attempts_reached",
                    attempts=attempts,
                )
                return

            # Exponential backoff
            self._state = CollectorState.RECONNECTING
            self._reconnect_count += 1
            self._log.info(
                "collector_reconnecting",
                delay_s=current_delay,
                attempt=attempts,
            )

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=current_delay,
                )
                # stop_event was set during wait
                break
            except TimeoutError:
                # Normal: timeout expired, proceed with reconnect
                pass

            current_delay = min(current_delay * 2, self._max_reconnect_delay_s)

    @abstractmethod
    async def _connect(self) -> None:
        """Establish connection to data source."""

    @abstractmethod
    async def _disconnect(self) -> None:
        """Clean up connection resources."""

    @abstractmethod
    async def _run_loop(self) -> None:
        """Main data collection loop. Should check self._stop_event periodically."""
