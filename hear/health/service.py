from __future__ import annotations

from pathlib import Path
from typing import Any

from hear.inference.manifest import ModelManifest
from hear.runtime.roles import WorkerCapabilityRegistry, WorkerRole


class RuntimeReadiness:
    def __init__(
        self,
        role: WorkerRole,
        manifest: ModelManifest,
        model_root: Path,
        patch_manager: Any,
        *,
        enabled_features: frozenset[str] = frozenset(),
        require_manifest_models: bool = True,
        simulation: bool = False,
    ) -> None:
        self._simulation = simulation
        self._role = role
        self._manifest = manifest
        self._model_root = model_root
        self._patch_manager = patch_manager
        self._enabled_features = enabled_features
        self._require_manifest_models = require_manifest_models
        self._patch_required = role in {WorkerRole.PIPELINE, WorkerRole.TRANSCRIPTION}
        self._patch_verified = not self._patch_required
        self._patch_error: str | None = None
        self._initialized = False
        self._draining = False
        self._checks: dict[str, Any] = {}

    def add_check(self, name: str, check: Any) -> None:
        self._checks[name] = check

    def set_draining(self, draining: bool) -> None:
        self._draining = draining

    def initialize(self) -> None:
        if self._patch_required:
            try:
                self._patch_manager.run(check=True)
                self._patch_verified = True
                self._patch_error = None
            except Exception as exc:
                self._patch_verified = False
                self._patch_error = str(exc)
        self._initialized = True

    def snapshot(self) -> dict:
        missing_models = (
            self._manifest.validate_local(
                self._model_root,
                self._role,
                enabled_features=self._enabled_features,
            )
            if self._require_manifest_models
            else ()
        )
        license_blockers = (
            self._manifest.license_blockers(
                self._role,
                enabled_features=self._enabled_features,
            )
            if self._require_manifest_models
            else ()
        )
        capability = WorkerCapabilityRegistry().get(self._role)
        check_results: dict[str, bool] = {}
        check_errors: dict[str, str] = {}
        for name, check in self._checks.items():
            try:
                check_results[name] = bool(check())
            except Exception as exc:
                check_results[name] = False
                check_errors[name] = str(exc)
        ready = (
            self._initialized
            and not self._draining
            and self._patch_verified
            and not missing_models
            and (not license_blockers or self._simulation)
            and all(check_results.values())
        )
        status = "draining" if self._draining else ("ready" if ready else "loading")
        return {
            "status": status,
            "runtime_mode": "simulation" if self._simulation else "production",
            "deployment_approved": not bool(license_blockers),
            "role": self._role.value,
            "job_types": [item.value for item in capability.job_types],
            "magic_clean_profile": (
                capability.magic_clean_profile.value
                if capability.magic_clean_profile is not None
                else None
            ),
            "patch_required": self._patch_required,
            "patch_verified": self._patch_verified,
            "patch_error": self._patch_error,
            "missing_models": list(missing_models),
            "license_blockers": list(license_blockers),
            "features": sorted(self._enabled_features),
            "checks": check_results,
            "check_errors": check_errors,
        }

    def is_ready(self) -> bool:
        return self.snapshot()["status"] == "ready"
