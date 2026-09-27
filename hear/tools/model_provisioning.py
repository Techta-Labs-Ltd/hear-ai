from __future__ import annotations

import argparse
from pathlib import Path

from hear.inference.manifest import ModelManifest
from hear.runtime.roles import WorkerRole


class ModelProvisioner:
    def __init__(
        self,
        manifest: ModelManifest,
        model_root: Path,
        *,
        cache_dir: Path | None = None,
    ) -> None:
        self._manifest = manifest
        self._model_root = model_root
        self._cache_dir = cache_dir or model_root / ".hub-cache"

    def provision(
        self,
        role: WorkerRole,
        *,
        enabled_features: frozenset[str] = frozenset(),
    ) -> dict[str, str]:
        return self._manifest.provision(
            self._model_root,
            role,
            enabled_features=enabled_features,
            cache_dir=self._cache_dir,
        )

    def verify(
        self,
        role: WorkerRole,
        *,
        enabled_features: frozenset[str] = frozenset(),
    ) -> tuple[str, ...]:
        return (
            *self._manifest.license_blockers(role, enabled_features=enabled_features),
            *self._manifest.validate_local(
                self._model_root,
                role,
                enabled_features=enabled_features,
            ),
        )

    @classmethod
    def main(cls) -> int:
        parser = argparse.ArgumentParser()
        parser.add_argument("--role", choices=[item.value for item in WorkerRole], required=True)
        parser.add_argument("--model-root", type=Path, default=Path("/models"))
        parser.add_argument("--manifest", type=Path, default=Path("hear/model_manifest.json"))
        parser.add_argument("--cache-dir", type=Path)
        parser.add_argument("--feature", action="append", default=[])
        parser.add_argument("--verify-only", action="store_true")
        args = parser.parse_args()
        instance = cls(
            ModelManifest(args.manifest),
            args.model_root,
            cache_dir=args.cache_dir,
        )
        features = frozenset(args.feature)
        if args.verify_only:
            missing = instance.verify(
                WorkerRole(args.role),
                enabled_features=features,
            )
            if missing:
                raise RuntimeError("model verification failed: " + ", ".join(missing))
            return 0
        instance.provision(
            WorkerRole(args.role),
            enabled_features=features,
        )
        missing = instance.verify(
            WorkerRole(args.role),
            enabled_features=features,
        )
        if missing:
            raise RuntimeError("model verification failed: " + ", ".join(missing))
        return 0


if __name__ == "__main__":
    raise SystemExit(ModelProvisioner.main())
