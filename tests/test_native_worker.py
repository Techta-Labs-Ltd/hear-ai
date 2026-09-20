import asyncio
import threading

import pytest

from hear.core.blocking import NativeWorker


@pytest.mark.anyio
async def test_native_worker_waits_for_completion_before_releasing_cancelled_slot():
    worker = NativeWorker("test-native")
    started = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    def native():
        started.set()
        release.wait(timeout=5)
        completed.set()

    first = asyncio.create_task(worker.run(native))
    await asyncio.to_thread(started.wait, 2)
    first.cancel()
    second = asyncio.create_task(worker.run(lambda: completed.is_set()))
    await asyncio.sleep(0.01)
    assert not first.done()
    assert not second.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert await second is True
    await worker.close()
    with pytest.raises(RuntimeError, match="native_worker_closed"):
        await worker.run(lambda: None)
