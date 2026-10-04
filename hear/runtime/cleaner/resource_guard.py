import math
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from hear.runtime.cleaner.scratch_ledger import HostScratchLedger
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


@dataclass(frozen=True)
class ResourceBudget:
    scratch_bytes: int
    max_input_bytes: int
    max_frames: int
    # Shared across worker processes so concurrent jobs cannot overcommit the disk.
    ledger: HostScratchLedger | None = field(default=None, compare=False)

    def __post_init__(self):
        if min(self.scratch_bytes, self.max_input_bytes, self.max_frames) <= 0:
            raise ValueError("host budgets must be positive")


class ResourceGuard:
    def __init__(
        self,
        budget: ResourceBudget,
        workspace: Path,
        deadline_monotonic: float,
        cancelled: threading.Event,
    ):
        if not math.isfinite(deadline_monotonic):
            raise ValueError("worker deadline must be finite")
        self.budget = budget
        self.workspace = workspace
        self.deadline = deadline_monotonic
        self.cancelled = cancelled
        self.wall_deadline: datetime | None = None
        self._scratch_peak_bytes = 0

    @property
    def scratch_peak_bytes(self) -> int:
        return self._scratch_peak_bytes

    def _scratch_bytes(self) -> int:
        size = 0
        for path in self.workspace.rglob("*"):
            if path.is_symlink():
                raise CleanExecutionError(
                    ErrorCode.RESOURCE_EXHAUSTED, "workspace contains a symlink"
                )
            if path.is_file():
                size += path.stat().st_size
        self._scratch_peak_bytes = max(self._scratch_peak_bytes, size)
        return size

    def bind_deadline(self, deadline: datetime) -> None:
        """Only tighten the worker timeout to an authenticated attempt deadline.

        Keep both clocks: wall-clock jumps forward must expire the attempt,
        while jumps backward must never extend its monotonic execution budget.
        """
        if deadline.utcoffset() is None:
            raise ValueError("attempt deadline must include a timezone")
        started = time.monotonic()
        remaining = (deadline - datetime.now(UTC)).total_seconds()
        self.deadline = min(self.deadline, started + remaining)
        self.wall_deadline = min(self.wall_deadline, deadline) if self.wall_deadline else deadline

    def check(self) -> None:
        if self.cancelled.is_set():
            raise CleanExecutionError(ErrorCode.CANCELLED, "attempt cancelled")
        if time.monotonic() >= self.deadline or (
            self.wall_deadline is not None and datetime.now(UTC) >= self.wall_deadline
        ):
            raise CleanExecutionError(ErrorCode.DEADLINE_EXCEEDED, "attempt deadline exceeded")

    def check_scratch(self) -> int:
        self.check()
        size = self._scratch_bytes()
        if size > self.budget.scratch_bytes:
            raise CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "scratch budget exceeded")
        return size

    def reserve_scratch(self, additional_bytes: int) -> int:
        self.check()
        if type(additional_bytes) is not int or additional_bytes < 0:
            raise ValueError("scratch reservation must be a non-negative integer")
        required = self._scratch_bytes() + additional_bytes
        if required > self.budget.scratch_bytes:
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED, "insufficient scratch reservation"
            )
        if self.budget.ledger is not None:
            # Queue behind other jobs' disk use; give up only on cancel or deadline.
            self.budget.ledger.reserve(self.workspace, required, wait=self.check)
        self._scratch_peak_bytes = max(self._scratch_peak_bytes, required)
        return required

    def preflight_pcm(self, frames: int, channels: int, copies: int, output_bytes: int) -> int:
        self.check_scratch()
        if (
            not 0 < frames <= self.budget.max_frames
            or channels not in (1, 2)
            or copies < 1
            or output_bytes < 0
        ):
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED, "unsupported decode reservation"
            )
        required = frames * channels * 4 * copies + output_bytes
        return self.reserve_scratch(required)
