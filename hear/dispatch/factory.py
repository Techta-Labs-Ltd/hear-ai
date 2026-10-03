"""Build the configured dispatcher from backend environment variables."""

from __future__ import annotations

import json
from collections.abc import Mapping

from hear.dispatch.client import (
    AttemptDispatcher,
    DispatchTransport,
    PodDispatcher,
    ServerlessDispatcher,
)
from hear.runtime.roles import WorkerRole


class DispatcherFactory:
    TRANSPORT = "HEAR_AI_TRANSPORT"
    POD_BASE_URL = "HEAR_POD_BASE_URL"
    POD_API_KEY = "HEAR_POD_API_KEY"
    RUNPOD_API_KEY = "HEAR_RUNPOD_API_KEY"
    RUNPOD_ENDPOINTS = "HEAR_RUNPOD_ENDPOINTS_JSON"
    RUNPOD_BASE_URL = "HEAR_RUNPOD_BASE_URL"

    def __init__(self, environment: Mapping[str, str]) -> None:
        self._environment = environment

    def transport(self) -> DispatchTransport:
        return DispatchTransport(self._environment.get(self.TRANSPORT, "pod").strip().lower())

    def build(self) -> AttemptDispatcher:
        transport = self.transport()
        if transport == DispatchTransport.POD:
            return PodDispatcher(
                self._required(self.POD_BASE_URL), self._required(self.POD_API_KEY)
            )
        raw = json.loads(self._required(self.RUNPOD_ENDPOINTS))
        if not isinstance(raw, dict) or not raw:
            raise ValueError("invalid_runpod_endpoints")
        endpoints = {WorkerRole(role): str(endpoint) for role, endpoint in raw.items()}
        base_url = self._environment.get(self.RUNPOD_BASE_URL, "").strip()
        return ServerlessDispatcher(
            endpoints,
            self._required(self.RUNPOD_API_KEY),
            base_url=base_url or ServerlessDispatcher.DEFAULT_BASE_URL,
        )

    def _required(self, name: str) -> str:
        value = self._environment.get(name, "").strip()
        if not value:
            raise ValueError(f"missing_dispatch_setting:{name}")
        return value
