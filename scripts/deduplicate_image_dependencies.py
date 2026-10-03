"""Deduplicate byte-identical native packages only inside the image assembly stage."""

import hashlib
import os
import shutil
from pathlib import Path


class ImageDependencyDeduplication:
    @staticmethod
    def digest(root: Path):
        value = hashlib.sha256()
        for path in sorted(root.rglob("*")):
            if "__pycache__" in path.parts:
                continue
            value.update(str(path.relative_to(root)).encode())
            if path.is_symlink():
                value.update(b"link:" + os.readlink(path).encode())
            elif path.is_file():
                with path.open("rb") as stream:
                    value.update(hashlib.file_digest(stream, "sha256").digest())
        return value.hexdigest()

    @classmethod
    def main(cls):
        if os.environ.get("HEAR_IMAGE_ASSEMBLY") != "1":
            raise RuntimeError("This command is for the disposable image build stage only")
        moved = cls.deduplicate(Path("/opt/hear-image-assembly"))
        print("Byte-identical shared packages:", ", ".join(moved))

    @classmethod
    def deduplicate(cls, root: Path) -> list[str]:
        shared = root / "shared"
        shared.mkdir(exist_ok=True)
        roles = ("pipeline", "reconstruction", "magic_clean_natural")
        moved = []
        for package in ("torch", "torchgen", "functorch", "nvidia", "triton"):
            paths = [
                root / "venvs" / role / "lib/python3.12/site-packages" / package for role in roles
            ]
            existing = [path for path in paths if path.is_dir() and not path.is_symlink()]
            if len(existing) < 2:
                continue
            expected = cls.digest(existing[0])
            identical = [path for path in existing if cls.digest(path) == expected]
            if len(identical) < 2:
                continue
            first = identical[0]
            # Overlay layers may live on different devices. Copy into the
            # disposable assembly tree before removing the original packages.
            shutil.copytree(first, shared / package, symlinks=True)
            for path in identical:
                if path.exists():
                    shutil.rmtree(path)
                path.symlink_to("/opt/hear-ai-v11/shared/" + package, target_is_directory=True)
            moved.append(package)
        return moved


if __name__ == "__main__":
    ImageDependencyDeduplication.main()
