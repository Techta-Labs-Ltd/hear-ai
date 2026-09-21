"""Build the cleaner-only source allowlist, never the root legacy application."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


class CleanerWheelBuilder:
    @staticmethod
    def files(root: Path) -> list[str]:
        files = json.loads((root / "deploy/cleaner/package-files.json").read_text())
        if not isinstance(files, list) or not files or len(files) != len(set(files)):
            raise ValueError("invalid cleaner file allowlist")
        for name in files:
            relative = PurePosixPath(name)
            path = root / name
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in name
                or relative.parts[0] != "hear"
                or relative.suffix not in (".py", ".pyi", ".proto")
                or path.is_symlink()
                or not path.is_file()
                or not path.resolve().is_relative_to(root.resolve())
            ):
                raise ValueError("unsafe or missing cleaner source")
        return sorted(files)

    @staticmethod
    def audit(wheel: Path, expected: list[str], root: Path) -> dict:
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            sources = sorted(name for name in names if name.startswith("hear/"))
            if sources != sorted(expected) or len(names) != len(set(names)):
                raise ValueError("wheel differs from cleaner source allowlist")
            extras = set(names) - set(sources)
            allowed_metadata = {
                "hear_cleaner_runtime-0.1.0.dist-info/" + name
                for name in ("METADATA", "WHEEL", "RECORD", "top_level.txt")
            }
            if extras != allowed_metadata:
                raise ValueError("unexpected wheel payload")
            hashes = {}
            for name in sources:
                content = archive.read(name)
                if content != (root / name).read_bytes():
                    raise ValueError("wheel source content mismatch")
                hashes[name] = hashlib.sha256(content).hexdigest()
        return {"wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(), "sources": hashes}

    @classmethod
    def build(cls, root: Path, destination: Path) -> dict:
        if importlib.metadata.version("setuptools") != "83.0.0":
            raise RuntimeError("use the pinned setuptools 83.0.0 build environment")
        files = cls.files(root)
        if destination.exists() and any(destination.iterdir()):
            raise ValueError(
                "build output directory must be empty; existing artifacts are preserved"
            )
        destination.mkdir(parents=True, exist_ok=True)
        destination = destination.resolve()
        uv = shutil.which("uv")
        if uv is None:
            raise RuntimeError("uv build is required")
        with tempfile.TemporaryDirectory(prefix="hear-cleaner-build-") as directory:
            staging = Path(directory)
            for name in files:
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / name, target)
            shutil.copyfile(root / "deploy/cleaner/pyproject.toml", staging / "pyproject.toml")
            environment = os.environ.copy()
            environment["SOURCE_DATE_EPOCH"] = "315532800"
            subprocess.run(
                [
                    uv,
                    "build",
                    "--wheel",
                    "--no-build-logs",
                    "--no-create-gitignore",
                    "--offline",
                    "--no-build-isolation",
                    "--no-python-downloads",
                    "--python",
                    sys.executable,
                    "--out-dir",
                    str(destination),
                    str(staging),
                ],
                check=True,
                env=environment,
            )
        wheels = list(destination.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("expected one cleaner wheel")
        result = cls.audit(wheels[0], files, root)
        result["wheel"] = str(wheels[0])
        return result

    @classmethod
    def main(cls) -> None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--output", required=True, type=Path)
        args = parser.parse_args()
        print(
            json.dumps(cls.build(Path(__file__).resolve().parents[1], args.output), sort_keys=True)
        )


if __name__ == "__main__":
    CleanerWheelBuilder.main()
