"""Bounded attempt-local wall durations, not backend queue or billing metrics."""

import time
from contextlib import contextmanager


class StageTimings:
    STAGES = frozenset(
        {
            "download",
            "inspection",
            "loading",
            "inference",
            "cleanup",
            "validation",
            "mastering",
            "upload",
        }
    )

    def __init__(self):
        self._nanoseconds: dict[str, int] = {}
        self._active: str | None = None

    def reset(self) -> None:
        if self._active is not None:
            raise RuntimeError("cannot reset active stage timing")
        self._nanoseconds.clear()

    @contextmanager
    def measure(self, stage: str):
        if stage not in self.STAGES or self._active is not None:
            raise ValueError("invalid or overlapping stage timing")
        self._active = stage
        started = time.monotonic_ns()
        try:
            yield
        finally:
            elapsed = time.monotonic_ns() - started
            self._active = None
            self._nanoseconds[stage] = self._nanoseconds.get(stage, 0) + max(0, elapsed)

    def snapshot(self) -> dict[str, float]:
        """Completed spans only; absence means unmeasured, never zero work.

        Repeated spans of the same stage are summed (registry construction plus
        session loading). Failure/cancellation durations are retained as observed.
        No identifiers, credentials, audio paths or diagnostics are collected.
        """
        return {stage: elapsed / 1_000_000_000 for stage, elapsed in self._nanoseconds.items()}
