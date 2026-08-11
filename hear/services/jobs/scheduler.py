"""Per-user round-robin admission for durable orchestrator jobs."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class PendingJob:
    job_id: str
    run_id: str
    user_id: str
    job_type: str

    @property
    def key(self) -> tuple[str, str]:
        return self.job_id, self.run_id


class FairJobScheduler:
    """Round-robin users while enforcing per-user and per-type limits."""

    def __init__(
        self,
        *,
        max_active: int,
        max_active_per_user: int,
        type_limits: dict[str, int] | None = None,
    ) -> None:
        self.max_active = max(1, max_active)
        self.max_active_per_user = max(1, max_active_per_user)
        self.type_limits = {
            key: max(1, value) for key, value in (type_limits or {}).items()
        }
        self._queues: dict[str, deque[PendingJob]] = defaultdict(deque)
        self._users: deque[str] = deque()
        self._queued: set[tuple[str, str]] = set()
        self._active: dict[tuple[str, str], PendingJob] = {}
        self._active_by_user: dict[str, int] = defaultdict(int)
        self._active_by_type: dict[str, int] = defaultdict(int)

    @property
    def queued_count(self) -> int:
        return len(self._queued)

    @property
    def active_count(self) -> int:
        return len(self._active)

    def enqueue(self, job: PendingJob) -> bool:
        if job.key in self._queued or job.key in self._active:
            return False
        queue = self._queues[job.user_id]
        if not queue:
            self._users.append(job.user_id)
        queue.append(job)
        self._queued.add(job.key)
        return True

    def remove(self, job_id: str) -> bool:
        removed = False
        for user_id in list(self._users):
            queue = self._queues[user_id]
            kept = deque(job for job in queue if job.job_id != job_id)
            if len(kept) != len(queue):
                removed = True
                for job in queue:
                    if job.job_id == job_id:
                        self._queued.discard(job.key)
                self._queues[user_id] = kept
            if not kept:
                self._drop_user(user_id)
        return removed

    def pop_next(self) -> PendingJob | None:
        if self.active_count >= self.max_active or not self._users:
            return None
        checks = len(self._users)
        for _ in range(checks):
            user_id = self._users.popleft()
            queue = self._queues[user_id]
            if not queue:
                self._queues.pop(user_id, None)
                continue
            job = queue[0]
            type_limit = self.type_limits.get(job.job_type, self.max_active)
            eligible = (
                self._active_by_user[user_id] < self.max_active_per_user
                and self._active_by_type[job.job_type] < type_limit
            )
            self._users.append(user_id)
            if not eligible:
                continue
            queue.popleft()
            self._queued.discard(job.key)
            self._active[job.key] = job
            self._active_by_user[user_id] += 1
            self._active_by_type[job.job_type] += 1
            if not queue:
                self._drop_user(user_id)
            return job
        return None

    def complete(self, job: PendingJob) -> None:
        active = self._active.pop(job.key, None)
        if active is None:
            return
        self._active_by_user[active.user_id] -= 1
        self._active_by_type[active.job_type] -= 1

    def stats(self) -> dict:
        return {
            "queued": self.queued_count,
            "active": self.active_count,
            "active_users": sum(1 for value in self._active_by_user.values() if value),
            "queued_users": len(self._users),
            "active_by_type": {
                key: value for key, value in self._active_by_type.items() if value
            },
        }

    def _drop_user(self, user_id: str) -> None:
        self._queues.pop(user_id, None)
        try:
            self._users.remove(user_id)
        except ValueError:
            pass
