from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from hear.contracts.jobs import WorkerRole


class ModelAsset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1, max_length=128)
    source: str = Field(min_length=1, max_length=512)
    revision: str = Field(min_length=1, max_length=128)
    local_path: str = Field(min_length=1, max_length=1024)
    roles: tuple[WorkerRole, ...]
    required_files: tuple[str, ...] = ()


class ModelManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: int = Field(ge=1)
    assets: tuple[ModelAsset, ...]

    def for_role(self, role: WorkerRole) -> tuple[ModelAsset, ...]:
        return tuple(asset for asset in self.assets if role in asset.roles)


class ModelManifestLoader:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> ModelManifest:
        return ModelManifest.model_validate(yaml.safe_load(self._path.read_text()))
