from pathlib import Path

from huggingface_hub import snapshot_download

from hear.contracts.jobs import WorkerRole

from .manifest import ModelAsset, ModelManifest


class ModelProvisioner:
    def __init__(self, manifest: ModelManifest, cache_dir: Path) -> None:
        self._manifest = manifest
        self._cache_dir = cache_dir

    def assets_for_role(self, role: WorkerRole) -> tuple[ModelAsset, ...]:
        return self._manifest.for_role(role)

    def provision(self, role: WorkerRole) -> tuple[Path, ...]:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        resolved: list[Path] = []
        for asset in self.assets_for_role(role):
            destination = Path(asset.local_path)
            destination.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=asset.source,
                revision=asset.revision,
                local_dir=destination,
                cache_dir=self._cache_dir,
            )
            self._verify_asset(asset)
            resolved.append(destination)
        return tuple(resolved)

    def verify(self, role: WorkerRole) -> bool:
        for asset in self.assets_for_role(role):
            self._verify_asset(asset)
        return True

    @staticmethod
    def _verify_asset(asset: ModelAsset) -> None:
        destination = Path(asset.local_path)
        if not destination.is_dir():
            raise RuntimeError(f"model asset directory missing: {asset.name}")
        for required in asset.required_files:
            if not (destination / required).exists():
                raise RuntimeError(f"model asset file missing: {asset.name}:{required}")
