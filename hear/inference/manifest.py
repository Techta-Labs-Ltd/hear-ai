from __future__ import annotations

import hashlib
import importlib
import os
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hear.runtime.roles import WorkerRole


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    logical_name: str = Field(min_length=1, max_length=128)
    source_type: Literal["huggingface", "url"] = "huggingface"
    repo_id: str | None = Field(default=None, min_length=3, max_length=255)
    revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{40}$")
    download_url: str | None = Field(default=None, min_length=8, max_length=2048)
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    filename: str | None = Field(default=None, min_length=1, max_length=255)
    relative_path: str = Field(min_length=1, max_length=255)
    roles: tuple[WorkerRole, ...] = Field(min_length=1)
    required_files: tuple[str, ...]
    engine_adapter: str = Field(min_length=1, max_length=128)
    provenance_url: str = Field(min_length=8, max_length=2048)
    license_name: str = Field(min_length=1, max_length=128)
    license_status: Literal["verified", "review_required", "permission_required"]
    license_url: str = Field(min_length=8, max_length=2048)
    file_hashes: dict[str, str] = Field(default_factory=dict)
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
            if self.filename not in self.required_files:
                raise ValueError("url model filename must be required")
        paths = (self.relative_path, *self.required_files, *self.file_hashes)
        if len(set(self.required_files)) != len(self.required_files):
            raise ValueError("duplicate_required_model_file")
        if any(not self._safe_relative_path(path) for path in paths):
            raise ValueError("model_manifest_path_must_be_relative")
        if any("\\" in path for path in paths):
            raise ValueError("model_manifest_path_must_be_posix")
        if not set(self.file_hashes).issubset(self.required_files):
            raise ValueError("model_manifest_file_hash_must_target_required_file")
        if any(
            len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)
            for digest in self.file_hashes.values()
        ):
            raise ValueError("invalid_model_file_hash")
        return self

    @staticmethod
    def _safe_relative_path(value: str) -> bool:
        path = PurePosixPath(value)
        return (
            not path.is_absolute()
            and bool(value)
            and all(part not in {"", ".", ".."} for part in value.split("/"))
        )


class ModelManifestDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: Literal[1]
    models: tuple[ModelSpec, ...]

    @model_validator(mode="after")
    def validate_unique_models(self):
        names = [model.logical_name for model in self.models]
        paths = [model.relative_path for model in self.models]
        if len(names) != len(set(names)):
            raise ValueError("duplicate_model_logical_name")
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate_model_relative_path")
        return self


class ModelManifest:
    def __init__(self, path: Path) -> None:
        document = ModelManifestDocument.model_validate_json(path.read_text())
        self._version = document.manifest_version
        self._models = document.models
        self._hash_cache: dict[str, tuple[tuple[int, int, int, int], str]] = {}
        self._hash_lock = threading.Lock()

    @property
    def version(self) -> int:
        return self._version

    @property
    def models(self) -> tuple[ModelSpec, ...]:
        return self._models

    def spec(self, logical_name: str) -> ModelSpec:
        for model in self._models:
            if model.logical_name == logical_name:
                return model
        raise RuntimeError(f"unknown_model:{logical_name}")

    def validate_overrides(self, overrides: Mapping[str, Path]) -> None:
        names = {model.logical_name for model in self._models}
        unknown = sorted(set(overrides) - names)
        if unknown:
            raise ValueError("unknown_model_override:" + ",".join(unknown))

    def local_path(
        self,
        model_root: Path,
        logical_name: str,
        overrides: Mapping[str, Path] | None = None,
    ) -> Path:
        override = (overrides or {}).get(logical_name)
        if override is not None:
            return override
        return model_root / self.spec(logical_name).relative_path

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

    def license_blockers(
        self,
        role: WorkerRole,
        *,
        enabled_features: frozenset[str] = frozenset(),
    ) -> tuple[str, ...]:
        return tuple(
            f"{model.logical_name}:{model.license_status}"
            for model in self.models_for(role, enabled_features=enabled_features)
            if model.license_status != "verified"
        )

    def validate_local(
        self,
        model_root: Path,
        role: WorkerRole,
        *,
        enabled_features: frozenset[str] = frozenset(),
        overrides: Mapping[str, Path] | None = None,
    ) -> tuple[str, ...]:
        missing: list[str] = []
        for model in self.models_for(role, enabled_features=enabled_features):
            root = self.local_path(model_root, model.logical_name, overrides)
            # An override is trusted as given; manifest-relative paths must stay inside the root.
            boundary = root.resolve() if root != model_root / model.relative_path else model_root.resolve()
            if not root.resolve().is_relative_to(boundary):
                missing.append(f"{model.logical_name}:path_outside_model_root")
                continue
            if not root.is_dir():
                missing.append(f"{model.logical_name}:directory")
                continue
            for required_file in model.required_files:
                path = root / required_file
                if not path.resolve().is_relative_to(boundary):
                    missing.append(f"{model.logical_name}:{required_file}:path_outside_model_root")
                    continue
                if not path.is_file():
                    missing.append(f"{model.logical_name}:{required_file}")
                    continue
                if model.source_type == "url" and model.filename == required_file:
                    if self._verified_hash(path) != model.sha256:
                        missing.append(f"{model.logical_name}:{required_file}:sha256")
                expected_hash = model.file_hashes.get(required_file)
                if expected_hash and self._verified_hash(path) != expected_hash:
                    missing.append(f"{model.logical_name}:{required_file}:sha256")
        return tuple(missing)

    def provision(
        self,
        model_root: Path,
        role: WorkerRole,
        *,
        enabled_features: frozenset[str] = frozenset(),
        cache_dir: Path | None = None,
    ) -> dict[str, str]:
        blockers = self.license_blockers(role, enabled_features=enabled_features)
        if blockers:
            raise RuntimeError("model license approval required: " + ", ".join(blockers))
        model_root.mkdir(parents=True, exist_ok=True)
        resolved_cache = cache_dir or model_root / ".hub-cache"
        resolved_cache.mkdir(parents=True, exist_ok=True)
        result: dict[str, str] = {}
        for model in self.models_for(role, enabled_features=enabled_features):
            local_dir = model_root / model.relative_path
            if model.source_type == "huggingface":
                snapshot_download = importlib.import_module("huggingface_hub").snapshot_download
                result[model.logical_name] = snapshot_download(
                    repo_id=str(model.repo_id),
                    revision=str(model.revision),
                    local_dir=local_dir,
                    cache_dir=resolved_cache,
                    # Package the declared weight format and small tokenizer/
                    # configuration assets, avoiding duplicate binary exports.
                    allow_patterns=[
                        *model.required_files,
                        "*.json",
                        "*.txt",
                        "*.jinja",
                        "LICENSE*",
                        "README.md",
                    ],
                    ignore_patterns=[
                        "*.msgpack",
                        "flax_model*",
                        "tf_model*",
                        "*.h5",
                    ],
                )
            else:
                result[model.logical_name] = str(self._provision_url(model, local_dir))
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
                            raise RuntimeError(f"model checksum mismatch: {model.logical_name}")
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

    def _verified_hash(self, path: Path) -> str | None:
        resolved = path.resolve()
        key = str(resolved)
        try:
            before = path.stat()
        except OSError:
            return None
        signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        with self._hash_lock:
            cached = self._hash_cache.get(key)
            if cached is not None and cached[0] == signature:
                return cached[1]
            try:
                digest = self._sha256(path)
                after = path.stat()
            except OSError:
                return None
            final_signature = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            if final_signature != signature:
                self._hash_cache.pop(key, None)
                return None
            self._hash_cache[key] = (signature, digest)
            return digest
