from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib import request


class MagicCleanProvisioner:
    def __init__(self, model_root: Path) -> None:
        self.root = model_root / "magic-clean"
        self.root.mkdir(parents=True, exist_ok=True)

    def deepfilter(self) -> None:
        destination = self.root / "DeepFilterNet3"
        if self._deepfilter_valid(destination):
            return
        with tempfile.TemporaryDirectory(prefix="deepfilter-", dir=self.root) as raw:
            staging = Path(raw)
            archive_path = staging / "DeepFilterNet3.zip"
            with (
                request.urlopen(
                    "https://github.com/Rikorose/DeepFilterNet/raw/main/models/DeepFilterNet3.zip",
                    timeout=120,
                ) as response,
                archive_path.open("wb") as target,
            ):
                shutil.copyfileobj(response, target)
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    resolved = (staging / member.filename).resolve()
                    if not resolved.is_relative_to(staging.resolve()):
                        raise RuntimeError("invalid_deepfilter_archive")
                archive.extractall(staging)
            candidate = staging / "DeepFilterNet3"
            if not self._deepfilter_valid(candidate):
                raise RuntimeError("deepfilter_asset_verification_failed")
            self._replace(candidate, destination)

    @staticmethod
    def _replace(candidate: Path, destination: Path) -> None:
        previous = destination.with_name(destination.name + ".previous")
        if previous.exists():
            shutil.rmtree(previous)
        if destination.exists():
            destination.rename(previous)
        try:
            candidate.rename(destination)
        except BaseException:
            if previous.exists() and not destination.exists():
                previous.rename(destination)
            raise
        if previous.exists():
            shutil.rmtree(previous)

    @classmethod
    def _deepfilter_valid(cls, directory: Path) -> bool:
        return cls._files_valid(
            directory,
            {
                "config.ini": "415eb925d44990d938fb739f514aa3662c1ec0ea836cff044fa1291b82cb4290",
                "checkpoints/model_120.ckpt.best": (
                    "23b92884f63ccf54bb026014604625ab231657b6480df65db4095c4c171e6003"
                ),
            },
        )

    @classmethod
    def _files_valid(cls, directory: Path, files: dict[str, str]) -> bool:
        try:
            return all(
                cls._sha256(directory / filename) == digest for filename, digest in files.items()
            )
        except OSError:
            return False

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()


class MagicCleanProvisioningCli:
    @staticmethod
    def main() -> int:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model-root", type=Path, default=Path("/models"))
        parser.add_argument("--engine", choices=("all", "deepfilter"), default="deepfilter")
        args = parser.parse_args()
        provisioner = MagicCleanProvisioner(args.model_root)
        if args.engine in {"all", "deepfilter"}:
            provisioner.deepfilter()
        return 0


if __name__ == "__main__":
    raise SystemExit(MagicCleanProvisioningCli.main())
