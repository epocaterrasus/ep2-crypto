"""Tests for the orchestrator task manager."""

from __future__ import annotations

import asyncio
import time

import pytest

from ep2_crypto.ingest.base import BaseCollector, CollectorState
from ep2_crypto.ingest.orchestrator import Orchestrator


class FakeCollector(BaseCollector):
    """Test collector that emits messages at a fixed interval."""

    def __init__(
        self,
        name: str = "fake",
        *,
        message_interval_s: float = 0.01,
        **kwargs: object,
    ) -> None:
        super().__init__(name, **kwargs)  # type: ignore[arg-type]
        self.connected = False

    async def _connect(self) -> None:
        self.connected = True

    async def _disconnect(self) -> None:
        self.connected = False

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self._record_message(time.time())
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=0.01,
                )
                return
            except TimeoutError:
                pass


@pytest.mark.asyncio
async def test_orchestrator_add_and_list_collectors() -> None:
    """Collectors can be registered and listed."""
    orch = Orchestrator()
    c1 = FakeCollector("alpha")
    c2 = FakeCollector("beta")

    orch.add_collector(c1)
    orch.add_collector(c2)

    assert set(orch.collector_names) == {"alpha", "beta"}


@pytest.mark.asyncio
async def test_orchestrator_reject_duplicate_name() -> None:
    """Cannot register two collectors with the same name."""
    orch = Orchestrator()
    orch.add_collector(FakeCollector("dup"))

    with pytest.raises(ValueError, match="already registered"):
        orch.add_collector(FakeCollector("dup"))


@pytest.mark.asyncio
async def test_orchestrator_remove_collector() -> None:
    """Collectors can be unregistered before run()."""
    orch = Orchestrator()
    orch.add_collector(FakeCollector("removeme"))
    orch.remove_collector("removeme")
    assert orch.collector_names == []


@pytest.mark.asyncio
async def test_orchestrator_remove_nonexistent() -> None:
    """Removing a non-existent collector raises KeyError."""
    orch = Orchestrator()
    with pytest.raises(KeyError, match="not registered"):
        orch.remove_collector("ghost")


@pytest.mark.asyncio
async def test_orchestrator_run_and_shutdown() -> None:
    """Orchestrator starts all collectors and shuts down cleanly."""
    orch = Orchestrator()
    c1 = FakeCollector("one")
    c2 = FakeCollector("two")
    orch.add_collector(c1)
    orch.add_collector(c2)

    async def delayed_shutdown() -> None:
        await asyncio.sleep(0.1)
        await orch.shutdown()

    _task = asyncio.create_task(delayed_shutdown())
    await orch.run()

    assert not orch.running
    assert c1.state == CollectorState.STOPPED
    assert c2.state == CollectorState.STOPPED


@pytest.mark.asyncio
async def test_orchestrator_context_manager() -> None:
    """Orchestrator works as async context manager."""
    c1 = FakeCollector("ctx")

    async with Orchestrator() as orch:
        orch.add_collector(c1)

        async def run_briefly() -> None:
            asyncio.get_running_loop().call_later(0.1, orch._shutdown_event.set)
            await orch.run()

        await run_briefly()

    assert c1.state == CollectorState.STOPPED


@pytest.mark.asyncio
async def test_orchestrator_health_check() -> None:
    """Health check aggregates status from all collectors."""
    orch = Orchestrator()
    c1 = FakeCollector("health1")
    c2 = FakeCollector("health2")
    orch.add_collector(c1)
    orch.add_collector(c2)

    # Before running: not healthy (not running)
    health = orch.health_check()
    assert not health.healthy

    async def run_and_check() -> None:
        await asyncio.sleep(0.05)
        health = orch.health_check()
        assert health.healthy
        assert len(health.collectors) == 2
        assert health.unhealthy_count == 0
        assert health.total_messages > 0

        # Verify to_dict works
        d = health.to_dict()
        assert "collectors" in d
        assert len(d["collectors"]) == 2

        await orch.shutdown()

    _task = asyncio.create_task(run_and_check())
    await orch.run()


@pytest.mark.asyncio
async def test_orchestrator_no_collectors_returns_immediately() -> None:
    """Run with no collectors should return immediately."""
    orch = Orchestrator()
    await orch.run()
    assert not orch.running


@pytest.mark.asyncio
async def test_orchestrator_cannot_add_while_running() -> None:
    """Cannot add collectors while orchestrator is running."""
    orch = Orchestrator()
    orch.add_collector(FakeCollector("pre"))

    async def try_add() -> None:
        await asyncio.sleep(0.02)
        with pytest.raises(RuntimeError, match="Cannot add"):
            orch.add_collector(FakeCollector("during"))
        await orch.shutdown()

    _task = asyncio.create_task(try_add())
    await orch.run()


@pytest.mark.asyncio
async def test_orchestrator_cannot_remove_while_running() -> None:
    """Cannot remove collectors while orchestrator is running."""
    orch = Orchestrator()
    orch.add_collector(FakeCollector("locked"))

    async def try_remove() -> None:
        await asyncio.sleep(0.02)
        with pytest.raises(RuntimeError, match="Cannot remove"):
            orch.remove_collector("locked")
        await orch.shutdown()

    _task = asyncio.create_task(try_remove())
    await orch.run()
