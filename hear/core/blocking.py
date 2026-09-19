"""Cancellation-safe bridges for blocking and externally managed work."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable


async def run_blocking_to_completion[T](
    function: Callable[[], T],
    *,
    on_cancel: Callable[[], object] | None = None,
) -> T:
    """Wait for a blocking worker to stop before propagating cancellation.

    Python cannot cancel a thread that is already writing a local or remote
    artifact. Returning control while that writer is still active lets cleanup
    race the write, so cancellation is deferred until the worker has stopped.
    """
    future = asyncio.get_running_loop().run_in_executor(None, function)
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(future)
        except asyncio.CancelledError as exc:
            # A callable can itself raise ``asyncio.CancelledError``. In that
            # case the executor Future is done (with that exception) but is not
            # marked cancelled. Treat every terminal Future as worker-side
            # cancellation; otherwise repeatedly awaiting it spins forever.
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
                        # Cancellation still needs to wait for the worker's
                        # terminal state even if its best-effort signal fails.
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


async def run_awaitable_to_completion[T](
    awaitable: Awaitable[T],
    *,
    on_cancel: Callable[[], object] | None = None,
) -> T:
    """Cancel an in-flight remote worker and await its terminal state safely."""
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
                        # The callback is best effort. Continue observing the
                        # awaitable so cleanup cannot race a live writer.
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
