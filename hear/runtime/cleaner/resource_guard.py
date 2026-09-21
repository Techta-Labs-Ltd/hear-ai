import math
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


@dataclass(frozen=True)
class ResourceBudget:
    scratch_bytes: int
    max_input_bytes: int
    max_frames: int
    gpu_limit_bytes: int = 12_000_000_000
    gpu_target_bytes: int = 10_000_000_000
    allocator_cap_bytes: int = 9_000_000_000

    def __post_init__(self):
        if min(self.scratch_bytes, self.max_input_bytes, self.max_frames) <= 0:
            raise ValueError("host budgets must be positive")
        if (
            not 0
            < self.allocator_cap_bytes
            <= self.gpu_target_bytes
            < self.gpu_limit_bytes
            <= 12_000_000_000
        ):
            raise ValueError("invalid cleaner GPU budgets")


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
        size = 0
        for path in self.workspace.rglob("*"):
            if path.is_symlink():
                raise CleanExecutionError(
                    ErrorCode.RESOURCE_EXHAUSTED, "workspace contains a symlink"
                )
            if path.is_file():
                size += path.stat().st_size
        if size > self.budget.scratch_bytes:
            raise CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "scratch budget exceeded")

    def preflight_pcm(self, frames: int, channels: int, copies: int, output_bytes: int) -> int:
        self.check()
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
        if required > self.budget.scratch_bytes:
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED, "insufficient scratch reservation"
            )
        return required
