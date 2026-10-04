"""Spawned worker processes for CPU/GPU-heavy audio stages with deadline polling.

One job spreads its work across these processes (chunk cleaning, MP3 pieces,
loudness ranges). The parent polls its ResourceGuard while waiting so a cancel
or deadline kills every worker at once instead of leaving orphans, and a worker
that dies fails the job instead of hanging it. Workers are spawned, never
forked: the parent holds CUDA and asyncio state.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


class WorkerPool:
    POLL_SECONDS = 0.5

    def __init__(
        self,
        workers: int,
        *,
        initializer: Callable[..., None] | None = None,
        initargs: tuple = (),
    ) -> None:
        if workers < 1:
            raise ValueError("worker_pool_needs_a_worker")
        self._executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initializer,
            initargs=initargs,
        )

    @staticmethod
    def guard(
        workspace: Path, deadline_epoch: float, budget: ResourceBudget | None = None
    ) -> ResourceGuard:
        """A guard inside a worker: wall deadline from the parent, no shared ledger."""
        workspace.mkdir(parents=True, exist_ok=True)
        remaining = deadline_epoch - datetime.now(UTC).timestamp()
        guard = ResourceGuard(
            budget or ResourceBudget(2**62, 2**62, 2**62),
            workspace,
            time.monotonic() + remaining,
            threading.Event(),
        )
        guard.bind_deadline(datetime.fromtimestamp(deadline_epoch, UTC))
        return guard

    def map(self, function: Callable[[Any], Any], tasks: Iterable[Any], guard: ResourceGuard) -> list:
        """Run `function` over `tasks`, results in submission order; fail fast."""
        futures = [self._executor.submit(function, item) for item in tasks]
        try:
            pending = set(futures)
            while pending:
                guard.check()
                done, pending = wait(
                    pending, timeout=self.POLL_SECONDS, return_when="FIRST_EXCEPTION"
                )
                for future in done:
                    future.result()  # raises the worker's error here
            return [future.result() for future in futures]
        except BrokenProcessPool as exc:
            raise CleanExecutionError(
                ErrorCode.PROCESS_FAILED, "audio worker process died"
            ) from exc
        except BaseException:
            self.close(terminate=True)
            raise

    def close(self, *, terminate: bool = False) -> None:
        executor, self._executor = self._executor, None
        if executor is None:
            return
        if terminate:
            executor.shutdown(wait=False, cancel_futures=True)
            for process in list(getattr(executor, "_processes", {}).values()):
                try:
                    os.kill(process.pid, signal.SIGKILL)
                except (ProcessLookupError, AttributeError):
                    pass
        executor.shutdown(wait=True)

    def __enter__(self) -> WorkerPool:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(terminate=exc_type is not None)

    @classmethod
    def run(
        cls,
        function: Callable[[Any], Any],
        tasks: list,
        guard: ResourceGuard,
        *,
        workers: int,
        initializer: Callable[..., None] | None = None,
        initargs: tuple = (),
    ) -> list:
        """Map over tasks, inline for a single task so short jobs pay no spawn cost."""
        if not tasks:
            return []
        if len(tasks) == 1 or workers <= 1:
            return [cls._inline(function, task, guard, initializer, initargs) for task in tasks]
        with cls(min(workers, len(tasks)), initializer=initializer, initargs=initargs) as pool:
            return pool.map(function, tasks, guard)

    @staticmethod
    def _inline(function, task, guard, initializer, initargs):
        guard.check()
        if initializer is not None:
            initializer(*initargs)
        try:
            return function(task)
        except CleanExecutionError:
            raise
        except MemoryError as exc:
            raise CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "worker out of memory") from exc
