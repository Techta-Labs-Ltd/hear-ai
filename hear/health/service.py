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
    ) -> None:
        self._role = role
        self._manifest = manifest
        self._model_root = model_root
        self._patch_manager = patch_manager
        self._enabled_features = enabled_features
        self._patch_required = role in {WorkerRole.PIPELINE, WorkerRole.TRANSCRIPTION}
        self._patch_verified = not self._patch_required
        self._patch_error: str | None = None
        self._initialized = False

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
        missing_models = self._manifest.validate_local(
            self._model_root,
            self._role,
            enabled_features=self._enabled_features,
        )
        capability = WorkerCapabilityRegistry().get(self._role)
        ready = self._initialized and self._patch_verified and not missing_models
        return {
            "status": "ready" if ready else "loading",
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
            "features": sorted(self._enabled_features),
        }

    def is_ready(self) -> bool:
        return self.snapshot()["status"] == "ready"