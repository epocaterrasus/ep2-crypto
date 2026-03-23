"""Tests for the base collector interface."""

from __future__ import annotations

import asyncio
import time

import pytest

from ep2_crypto.ingest.base import BaseCollector, CollectorState, HealthStatus


class FakeCollector(BaseCollector):
    """Minimal concrete collector for testing."""

    def __init__(
        self,
        name: str = "fake",
        *,
        fail_connect: bool = False,
        fail_after_messages: int = 0,
        message_interval_s: float = 0.01,
        **kwargs: object,
    ) -> None:
        super().__init__(name, **kwargs)  # type: ignore[arg-type]
        self.fail_connect = fail_connect
        self.fail_after_messages = fail_after_messages
        self.message_interval_s = message_interval_s
        self.connected = False
        self.disconnect_count = 0

    async def _connect(self) -> None:
        if self.fail_connect:
            msg = "Connection refused"
            raise ConnectionError(msg)
        self.connected = True

    async def _disconnect(self) -> None:
        self.connected = False
        self.disconnect_count += 1

    async def _run_loop(self) -> None:
        count = 0
        while not self._stop_event.is_set():
            self._record_message(time.time())
            count += 1
            if self.fail_after_messages > 0 and count >= self.fail_after_messages:
                msg = "Simulated stream error"
                raise RuntimeError(msg)
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.message_interval_s,
                )
                return  # stop was requested
            except TimeoutError:
                pass


@pytest.mark.asyncio
async def test_collector_lifecycle() -> None:
    """Collector transitions through IDLE -> STARTING -> RUNNING -> STOPPED."""
    collector = FakeCollector()
    assert collector.state == CollectorState.IDLE

    await collector.start()
    # Give the task time to connect
    await asyncio.sleep(0.05)
    assert collector.state == CollectorState.RUNNING
    assert collector.connected

    await collector.stop()
    assert collector.state == CollectorState.STOPPED
    assert not collector.connected


@pytest.mark.asyncio
async def test_collector_context_manager() -> None:
    """Async context manager starts and stops the collector."""
    collector = FakeCollector()
    async with collector:
        await asyncio.sleep(0.05)
        assert collector.state == CollectorState.RUNNING

    assert collector.state == CollectorState.STOPPED


@pytest.mark.asyncio
async def test_collector_records_messages() -> None:
    """Messages are counted and timestamped."""
    collector = FakeCollector(message_interval_s=0.01)
    async with collector:
        await asyncio.sleep(0.1)

    health = collector.health_check()
    assert health.messages_received > 0
    assert health.last_message_ts is not None


@pytest.mark.asyncio
async def test_collector_health_check() -> None:
    """Health check reflects collector state."""
    collector = FakeCollector()
    health = collector.health_check()
    assert not health.healthy
    assert health.state == CollectorState.IDLE
    assert health.name == "fake"

    async with collector:
        await asyncio.sleep(0.05)
        health = collector.health_check()
        assert health.healthy
        assert health.state == CollectorState.RUNNING


@pytest.mark.asyncio
async def test_collector_reconnect_on_error() -> None:
    """Collector reconnects with backoff after run_loop error."""
    collector = FakeCollector(
        fail_after_messages=2,
        reconnect_delay_s=0.01,
        max_reconnect_delay_s=0.1,
    )
    async with collector:
        await asyncio.sleep(0.15)

    assert collector._reconnect_count > 0
    assert collector._messages_received > 2  # received across reconnects


@pytest.mark.asyncio
async def test_collector_connect_failure_reconnect() -> None:
    """Collector retries when _connect raises."""
    collector = FakeCollector(
        fail_connect=True,
        reconnect_delay_s=0.01,
        max_reconnect_attempts=3,
    )
    await collector.start()
    # Wait for attempts to exhaust
    await asyncio.sleep(0.2)

    assert collector.state == CollectorState.ERROR
    assert collector._reconnect_count >= 2
    assert "Connection refused" in (collector._last_error or "")
    await collector.stop()


@pytest.mark.asyncio
async def test_collector_max_reconnect_attempts() -> None:
    """Collector stops after max reconnect attempts."""
    collector = FakeCollector(
        fail_connect=True,
        reconnect_delay_s=0.01,
        max_reconnect_attempts=2,
    )
    await collector.start()
    await asyncio.sleep(0.15)

    assert collector.state == CollectorState.ERROR
    await collector.stop()


@pytest.mark.asyncio
async def test_collector_double_start_noop() -> None:
    """Starting an already running collector is a no-op."""
    collector = FakeCollector()
    async with collector:
        await asyncio.sleep(0.05)
        await collector.start()  # should warn but not crash
        assert collector.state == CollectorState.RUNNING


@pytest.mark.asyncio
async def test_collector_double_stop_noop() -> None:
    """Stopping an already stopped collector is a no-op."""
    collector = FakeCollector()
    async with collector:
        await asyncio.sleep(0.02)

    await collector.stop()  # second stop should be safe
    assert collector.state == CollectorState.STOPPED


@pytest.mark.asyncio
async def test_health_status_dataclass() -> None:
    """HealthStatus fields are correctly populated."""
    status = HealthStatus(
        name="test",
        healthy=True,
        state=CollectorState.RUNNING,
        messages_received=42,
        reconnect_count=1,
        error=None,
    )
    assert status.name == "test"
    assert status.healthy
    assert status.messages_received == 42
    assert status.reconnect_count == 1
