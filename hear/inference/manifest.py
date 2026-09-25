from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import snapshot_download
from pydantic import BaseModel, ConfigDict, Field

from hear.runtime.roles import WorkerRole


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=128)
    repo_id: str = Field(min_length=3, max_length=255)
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    relative_path: str = Field(min_length=1, max_length=255)
    roles: tuple[WorkerRole, ...]
    required_files: tuple[str, ...]
    feature: str | None = Field(default=None, max_length=128)


class ModelManifest:
    def __init__(self, path: Path) -> None:
        self._path = path
        payload = json.loads(path.read_text())
        self._models = tuple(ModelSpec.model_validate(item) for item in payload["models"])

    @property
    def models(self) -> tuple[ModelSpec, ...]:
        return self._models

    def models_for(
        self,
        role: WorkerRole,
        *,
        enabled_features: frozenset[str] = frozenset(),
    ) -> tuple[ModelSpec, ...]:
        return tuple(
            model
            for model in self._models
            if role in model.roles and (model.feature is None or model.feature in enabled_features)
        )

    def validate_local(
        self,
        model_root: Path,
        role: WorkerRole,
        *,
        enabled_features: frozenset[str] = frozenset(),
    ) -> tuple[str, ...]:
        missing: list[str] = []
        for model in self.models_for(role, enabled_features=enabled_features):
            root = model_root / model.relative_path
            if not root.is_dir():
                missing.append(f"{model.name}:directory")
                continue
            for required_file in model.required_files:
                if not (root / required_file).is_file():
                    missing.append(f"{model.name}:{required_file}")
        return tuple(missing)

    def provision(
        self,
        model_root: Path,
        role: WorkerRole,
        *,
        enabled_features: frozenset[str] = frozenset(),
        cache_dir: Path | None = None,
    ) -> dict[str, str]:
        model_root.mkdir(parents=True, exist_ok=True)
        resolved_cache = cache_dir or model_root / ".hub-cache"
        resolved_cache.mkdir(parents=True, exist_ok=True)
        result: dict[str, str] = {}
        for model in self.models_for(role, enabled_features=enabled_features):
            local_dir = model_root / model.relative_path
            result[model.name] = snapshot_download(
                repo_id=model.repo_id,
                revision=model.revision,
                local_dir=local_dir,
                cache_dir=resolved_cache,
                ignore_patterns=(
                    "*.msgpack",
                    "flax_model*",
                    "tf_model*",
                    "*.h5",
                ),
            )
        missing = self.validate_local(
            model_root,
            role,
            enabled_features=enabled_features,
        )
        if missing:
            raise RuntimeError("model manifest validation failed: " + ", ".join(missing))
        return result