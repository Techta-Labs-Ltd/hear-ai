from collections.abc import Callable
from dataclasses import dataclass

from hear.runtime.roles import RoleCapabilities


@dataclass(slots=True)
class HealthSnapshot:
    ready: bool
    role: str
    active: int
    capacity: int
    checks: dict[str, bool]


class HealthService:
    def __init__(
        self,
        capabilities: RoleCapabilities,
        checks: dict[str, Callable[[], bool]] | None = None,
        capacity: int = 1,
    ) -> None:
        self._capabilities = capabilities
        self._checks = dict(checks or {})
        self._capacity = max(1, capacity)
        self._active = 0
        self._draining = False

    def set_active(self, active: int) -> None:
        self._active = max(0, active)

    def set_draining(self, draining: bool) -> None:
        self._draining = draining

    def snapshot(self) -> HealthSnapshot:
        results = {name: bool(check()) for name, check in self._checks.items()}
        ready = not self._draining and all(results.values()) and self._active < self._capacity
        return HealthSnapshot(
            ready=ready,
            role=self._capabilities.role.value,
            active=self._active,
            capacity=self._capacity,
            checks=results,
        )

    def capabilities(self) -> dict:
        return {
            "role": self._capabilities.role.value,
            "job_types": sorted(item.value for item in self._capabilities.job_types),
            "magic_clean_profiles": sorted(
                item.value for item in self._capabilities.magic_clean_profiles
            ),
        }
