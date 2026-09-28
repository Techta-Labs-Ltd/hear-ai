"""Produce a deterministic source-only Docker context; never include models or secrets."""

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path


class ImageContext:
    @staticmethod
    def main():
        output, *names = sys.argv[1:]
        manifest = {}
        with tarfile.open(output, "w", format=tarfile.PAX_FORMAT) as archive:
            for name in sorted(set(names)):
                path = Path(name)
                if path.is_absolute() or ".." in path.parts or path.name.startswith(".env"):
                    raise ValueError("unsafe image context path: " + name)
                if path.suffix in (
                    ".mp3",
                    ".wav",
                    ".flac",
                    ".pth",
                    ".ckpt",
                    ".safetensors",
                    ".pem",
                ):
                    raise ValueError("audio, weights and keys do not belong in the source context")
                data = path.read_bytes()
                manifest[name] = hashlib.sha256(data).hexdigest()
                entry = tarfile.TarInfo(name)
                entry.size = len(data)
                entry.mode = 0o755 if name.endswith(".sh") else 0o644
                archive.addfile(entry, io.BytesIO(data))
            data = json.dumps(manifest, sort_keys=True, indent=2).encode()
            entry = tarfile.TarInfo("BUILD_CONTEXT_MANIFEST.json")
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))


if __name__ == "__main__":
    ImageContext.main()
