"""Native Ray coordination for resolver taxonomy versions."""

from __future__ import annotations

import ray


_ACTOR_NAME = "resolver_version_coordinator"
_ACTOR_NAMESPACE = "resolver"


@ray.remote(num_cpus=0)
class _ResolverVersionCoordinator:
    def __init__(self) -> None:
        self._desired_version = 0

    def get_desired_version(self) -> int:
        return self._desired_version

    def publish_version(self, version: int) -> int:
        if version > 0:
            self._desired_version = max(self._desired_version, version)
        return self._desired_version


def get_version_coordinator():
    return _ResolverVersionCoordinator.options(
        name=_ACTOR_NAME,
        namespace=_ACTOR_NAMESPACE,
        get_if_exists=True,
        lifetime="detached",
    ).remote()
