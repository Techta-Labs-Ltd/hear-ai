"""Explicit local integration-test boundary; never declares commercial deployment approval."""

import os
from urllib.parse import urlsplit

from hear.runtime.ownership import BackendRegistry, DeploymentOwnership


class SimulationBoundary:
    @staticmethod
    def enabled() -> bool:
        mode = os.environ.get("HEAR_RUNTIME_MODE", "production")
        if mode not in ("production", "simulation"):
            raise ValueError("invalid_runtime_mode")
        if mode == "production":
            return False
        registry = DeploymentOwnership.load()
        if not isinstance(registry, BackendRegistry):
            raise ValueError("simulation_requires_explicit_backend_registry")
        for name, (policy, callback, _token) in registry.entries.items():
            if not name.startswith("simulation-"):
                raise ValueError("simulation_cannot_use_real_backend_identity")
            for value in (
                *policy.backend_base_urls,
                callback,
                policy.storage_endpoint,
                policy.public_base_url,
            ):
                parsed = urlsplit(value)
                if parsed.scheme != "https" or parsed.hostname not in (
                    "localhost",
                    "127.0.0.1",
                    "::1",
                ):
                    raise ValueError("simulation_requires_loopback_https_services")
            if set(policy.source_hosts) - {"localhost", "127.0.0.1", "::1"}:
                raise ValueError("simulation_source_must_be_local")
            if not policy.bucket_name.startswith("hear-simulation-"):
                raise ValueError("simulation_storage_bucket_must_be_explicit")
        return True
