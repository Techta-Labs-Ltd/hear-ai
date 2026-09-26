from __future__ import annotations

import hashlib
from pathlib import Path

from hear.contracts.outcomes import ArtifactManifest
from hear.storage.b2 import B2Storage


class ReconstructionStorageAdapter:
    def __init__(self, storage: B2Storage) -> None:
        self._storage = storage
        self._artifacts: list[ArtifactManifest] = []

    @property
    def bucket_name(self) -> str:
        return self._storage.bucket_name

    @property
    def artifacts(self) -> tuple[ArtifactManifest, ...]:
        return tuple(self._artifacts)

    def key(self, *parts: str) -> str:
        return self._storage.key(*parts)

    def upload_file(
        self,
        local_path: str,
        object_key: str,
        content_type: str | None = None,
    ) -> str:
        path = Path(local_path)
        digest = self._sha256(path)
        artifact = self._storage.upload_file(
            path,
            object_key,
            sha256=digest,
            content_type=content_type,
        )
        self._artifacts.append(artifact)
        if artifact.audio_url is None:
            raise RuntimeError("reconstruction_artifact_url_missing")
        return artifact.audio_url

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while block := source.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()
