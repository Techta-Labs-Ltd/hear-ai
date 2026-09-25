from __future__ import annotations

import argparse
from pathlib import Path

from hear.inference.manifest import ModelManifest
from hear.runtime.roles import WorkerRole
from hear.tools.dependency_patches import DependencyPatchManager


class ImageValidator:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._manifest = ModelManifest(root / "hear" / "model_manifest.json")
        self._patch_manager = DependencyPatchManager(root)

    def validate(
        self,
        role: WorkerRole,
        model_root: Path,
        *,
        enabled_features: frozenset[str],
        require_models: bool,
    ) -> None:
        self._patch_manager.run(check=True)
        if not require_models:
            return
        missing = self._manifest.validate_local(
            model_root,
            role,
            enabled_features=enabled_features,
        )
        if missing:
            raise RuntimeError("missing required model files: " + ", ".join(missing))

    @classmethod
    def main(cls) -> int:
        parser = argparse.ArgumentParser()
        parser.add_argument("--role", choices=[item.value for item in WorkerRole], required=True)
        parser.add_argument("--model-root", type=Path, default=Path("/models"))
        parser.add_argument("--feature", action="append", default=[])
        parser.add_argument("--require-models", action="store_true")
        args = parser.parse_args()
        validator = cls(Path(__file__).resolve().parents[1])
        validator.validate(
            WorkerRole(args.role),
            args.model_root,
            enabled_features=frozenset(args.feature),
            require_models=args.require_models,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(ImageValidator.main())