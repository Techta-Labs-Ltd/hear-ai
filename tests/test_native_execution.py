import asyncio
import threading

import pytest

from hear.execution.native import NativeExecutor


@pytest.mark.anyio
async def test_native_executor_keeps_cancelled_work_in_its_slot():
    executor = NativeExecutor("test-native")
    started = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    def native():
        started.set()
        release.wait(timeout=5)
        completed.set()

    first = asyncio.create_task(executor.run(native))
    await asyncio.to_thread(started.wait, 2)
    first.cancel()
    second = asyncio.create_task(executor.run(lambda: completed.is_set()))
    await asyncio.sleep(0.01)
    assert not first.done()
    assert not second.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert await second is True
    await executor.close()
    with pytest.raises(RuntimeError, match="native_executor_closed"):
        await executor.run(lambda: None)
