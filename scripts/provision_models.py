from __future__ import annotations

import argparse
from pathlib import Path

from hear.inference.manifest import ModelManifest
from hear.runtime.roles import WorkerRole


class ModelProvisioningCommand:
    def __init__(self, root: Path) -> None:
        self._root = root
        self._manifest = ModelManifest(root / "hear" / "model_manifest.json")

    def run(self, role: WorkerRole, model_root: Path, features: frozenset[str]) -> None:
        result = self._manifest.provision(
            model_root,
            role,
            enabled_features=features,
        )
        for name, location in sorted(result.items()):
            print(f"{name}={location}")

    @classmethod
    def main(cls) -> int:
        parser = argparse.ArgumentParser()
        parser.add_argument("--role", choices=[item.value for item in WorkerRole], required=True)
        parser.add_argument("--model-root", type=Path, default=Path("/models"))
        parser.add_argument("--feature", action="append", default=[])
        args = parser.parse_args()
        command = cls(Path(__file__).resolve().parents[1])
        command.run(
            WorkerRole(args.role),
            args.model_root,
            frozenset(args.feature),
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(ModelProvisioningCommand.main())