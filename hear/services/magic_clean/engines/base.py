from pathlib import Path
from typing import Protocol

from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.services.magic_clean.contracts import CleanPlan, RuntimeIdentity


class EngineSession(Protocol):
    """State belongs to one attempt; inputs/outputs remain bounded files."""

    def process(
        self, source: Path, destination: Path, plan: CleanPlan, guard: ResourceGuard
    ) -> None: ...

    def close(self) -> None: ...


class CleanEngine(Protocol):
    @property
    def identity(self) -> RuntimeIdentity: ...

    def open_session(self, plan: CleanPlan, guard: ResourceGuard) -> EngineSession: ...
