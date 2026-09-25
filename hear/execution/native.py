from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial


class NativeExecutor:
    def __init__(self, name: str, *, max_workers: int = 1) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, max_workers),
            thread_name_prefix=name,
        )
        self._slot = asyncio.Semaphore(max(1, max_workers))
        self._closed = False

    async def run(self, function, *args, **kwargs):
        async with self._slot:
            if self._closed:
                raise RuntimeError("native_executor_closed")
            future = asyncio.get_running_loop().run_in_executor(
                self._executor,
                partial(function, *args, **kwargs),
            )
            cancellation: asyncio.CancelledError | None = None
            while True:
                try:
                    result = await asyncio.shield(future)
                except asyncio.CancelledError as exc:
                    if future.done():
                        if cancellation is not None:
                            raise cancellation from exc
                        raise
                    if cancellation is None:
                        cancellation = exc
                    current = asyncio.current_task()
                    if current is not None:
                        current.uncancel()
                    continue
                except BaseException as worker_error:
                    if cancellation is not None:
                        raise cancellation from worker_error
                    raise
                if cancellation is not None:
                    raise cancellation
                return result

    async def close(self) -> None:
        self._closed = True
        async with self._slot:
            self._executor.shutdown(wait=True, cancel_futures=True)