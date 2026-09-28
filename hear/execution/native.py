from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from functools import partial
from threading import Event


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
            return await self.run_blocking_to_completion(
                partial(function, *args, **kwargs), executor=self._executor
            )

    async def run_cancellable(self, function, *args, cancelled: Event, **kwargs):
        async with self._slot:
            if self._closed:
                raise RuntimeError("native_executor_closed")
            return await self.run_blocking_to_completion(
                partial(function, *args, cancelled=cancelled, **kwargs),
                on_cancel=cancelled.set,
                executor=self._executor,
            )

    @staticmethod
    async def run_blocking_to_completion[T](
        function: Callable[[], T],
        *,
        on_cancel: Callable[[], object] | None = None,
        executor: Executor | None = None,
    ) -> T:
        future = asyncio.get_running_loop().run_in_executor(executor, function)
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
                    if on_cancel is not None:
                        try:
                            on_cancel()
                        except BaseException:
                            pass
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

    @staticmethod
    async def run_awaitable_to_completion[T](
        awaitable: Awaitable[T], *, on_cancel: Callable[[], object] | None = None
    ) -> T:
        future = asyncio.ensure_future(awaitable)
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                result = await asyncio.shield(future)
            except asyncio.CancelledError as exc:
                if future.cancelled():
                    if cancellation is not None:
                        raise cancellation from exc
                    raise
                if cancellation is None:
                    cancellation = exc
                    if on_cancel is not None:
                        try:
                            on_cancel()
                        except BaseException:
                            pass
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
