from __future__ import annotations

import asyncio
import time

from hear.contracts.jobs import AttemptClaim, AttemptEnvelope
from hear.execution.reporter import BackendAttemptClient


class AttemptLeaseLost(RuntimeError):
    pass


class AttemptLease:
    def __init__(
        self,
        backend: BackendAttemptClient,
        envelope: AttemptEnvelope,
        claim: AttemptClaim,
    ) -> None:
        self._backend = backend
        self._envelope = envelope
        self._lease_seconds = claim.lease_seconds
        self._heartbeat_seconds = claim.heartbeat_seconds
        self._last_success = time.monotonic()
        self._sequence = 0
        self._lost = asyncio.Event()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def close(self) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def next_event(self, iterator):
        event_task = asyncio.create_task(iterator.__anext__())
        lost_task = asyncio.create_task(self._lost.wait())
        done, pending = await asyncio.wait(
            {event_task, lost_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if lost_task in done and lost_task.result():
            event_task.cancel()
            try:
                await event_task
            except (asyncio.CancelledError, StopAsyncIteration):
                pass
            raise AttemptLeaseLost("attempt_lease_lost")
        return await event_task

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_seconds)
            self._sequence += 1
            try:
                await self._backend.heartbeat(self._envelope, self._sequence)
                self._last_success = time.monotonic()
            except Exception:
                if time.monotonic() - self._last_success >= self._lease_seconds:
                    self._lost.set()
                    return
