from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

import httpx
from huggingface_hub import snapshot_download
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hear.runtime.roles import WorkerRole


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, max_length=128)
    source_type: Literal["huggingface", "url"] = "huggingface"
    repo_id: str | None = Field(default=None, min_length=3, max_length=255)
    revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")
    download_url: str | None = Field(default=None, min_length=8, max_length=2048)
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    filename: str | None = Field(default=None, min_length=1, max_length=255)
    relative_path: str = Field(min_length=1, max_length=255)
    roles: tuple[WorkerRole, ...]
    required_files: tuple[str, ...]
    feature: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def validate_source(self):
        if self.source_type == "huggingface":
            if not self.repo_id or not self.revision:
                raise ValueError("huggingface model requires repo_id and revision")
            if self.download_url or self.sha256 or self.filename:
                raise ValueError("huggingface model cannot define url file fields")
        else:
            if not self.download_url or not self.sha256 or not self.filename:
                raise ValueError("url model requires download_url, sha256, and filename")
            if self.repo_id or self.revision:
                raise ValueError("url model cannot define repo_id or revision")
        return self


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
                path = root / required_file
                if not path.is_file():
                    missing.append(f"{model.name}:{required_file}")
                    continue
                if model.source_type == "url" and model.filename == required_file:
                    if self._sha256(path) != model.sha256:
                        missing.append(f"{model.name}:{required_file}:sha256")
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
            if model.source_type == "huggingface":
                result[model.name] = snapshot_download(
                    repo_id=str(model.repo_id),
                    revision=str(model.revision),
                    local_dir=local_dir,
                    cache_dir=resolved_cache,
                    ignore_patterns=[
                        "*.msgpack",
                        "flax_model*",
                        "tf_model*",
                        "*.h5",
                    ],
                )
            else:
                result[model.name] = str(self._provision_url(model, local_dir))
        missing = self.validate_local(
            model_root,
            role,
            enabled_features=enabled_features,
        )
        if missing:
            raise RuntimeError("model manifest validation failed: " + ", ".join(missing))
        return result

    @classmethod
    def _provision_url(cls, model: ModelSpec, local_dir: Path) -> Path:
        local_dir.mkdir(parents=True, exist_ok=True)
        destination = local_dir / str(model.filename)
        if destination.is_file() and cls._sha256(destination) == model.sha256:
            return destination
        with httpx.Client(timeout=httpx.Timeout(300.0), follow_redirects=True) as client:
            with client.stream("GET", str(model.download_url)) as response:
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(
                    dir=local_dir,
                    prefix=f".{model.filename}-",
                    suffix=".part",
                    delete=False,
                ) as stream:
                    temporary = Path(stream.name)
                    digest = hashlib.sha256()
                    try:
                        for chunk in response.iter_bytes(1024 * 1024):
                            if not chunk:
                                continue
                            stream.write(chunk)
                            digest.update(chunk)
                        stream.flush()
                        os.fsync(stream.fileno())
                        if digest.hexdigest() != model.sha256:
                            raise RuntimeError(f"model checksum mismatch: {model.name}")
                        temporary.replace(destination)
                    finally:
                        temporary.unlink(missing_ok=True)
        return destination

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()
