import asyncio
import time

from hear.runtime.gpu_idle import AsyncIdleResource, SyncIdleResource


class AsyncFake:
    def __init__(self, identity: int):
        self.identity = identity
        self.closed = False


def test_async_idle_resource_single_flight_and_evicts():
    async def run():
        loads = []
        closes = []

        async def factory():
            await asyncio.sleep(0.02)
            value = AsyncFake(len(loads) + 1)
            loads.append(value)
            return value

        async def closer(value):
            value.closed = True
            closes.append(value.identity)

        resource = AsyncIdleResource(
            "test", factory, closer, idle_seconds=0.05, eviction_enabled=True
        )
        first, second = await asyncio.gather(resource.acquire(), resource.acquire())
        assert first is second
        assert len(loads) == 1
        assert resource.snapshot["active"] == 2
        await resource.release()
        await resource.release()
        await asyncio.sleep(0.08)
        assert first.closed
        assert resource.state == "cold"
        assert closes == [1]

        third = await resource.acquire()
        assert third is not first
        assert len(loads) == 2
        await resource.release()
        await resource.close()
        assert resource.state == "closed"

    asyncio.run(run())


def test_async_idle_resource_does_not_evict_while_active():
    async def run():
        async def factory():
            return AsyncFake(1)

        async def closer(value):
            value.closed = True

        resource = AsyncIdleResource(
            "test", factory, closer, idle_seconds=0.03, eviction_enabled=True
        )
        value = await resource.acquire()
        await asyncio.sleep(0.06)
        assert not value.closed
        await resource.release()
        await asyncio.sleep(0.05)
        assert value.closed
        await resource.close()

    asyncio.run(run())


class SyncFake:
    def __init__(self, identity: int):
        self.identity = identity
        self.closed = False


def test_sync_idle_resource_single_flight_and_evicts():
    loads = []
    closes = []

    def factory():
        value = SyncFake(len(loads) + 1)
        loads.append(value)
        return value

    def closer(value):
        value.closed = True
        closes.append(value.identity)

    resource = SyncIdleResource(
        "sync-test", factory, closer, idle_seconds=0.04, eviction_enabled=True
    )
    first = resource.acquire()
    assert resource.acquire() is first
    resource.release()
    time.sleep(0.06)
    assert not first.closed
    resource.release()
    time.sleep(0.07)
    assert first.closed
    assert closes == [1]
    assert resource.state == "cold"
    resource.close()


def test_sync_idle_resource_disabled_keeps_resource_warm():
    resource = SyncIdleResource(
        "sync-test",
        lambda: SyncFake(1),
        lambda value: setattr(value, "closed", True),
        idle_seconds=0.02,
        eviction_enabled=False,
    )
    value = resource.acquire()
    resource.release()
    time.sleep(0.05)
    assert resource.state == "warm"
    assert not value.closed
    resource.close()
    assert value.closed
