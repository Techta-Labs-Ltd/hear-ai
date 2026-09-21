"""Regenerate Cleaner v2 bindings with the compiler version pinned in uv.lock."""

import importlib.metadata
import subprocess
import sys
from pathlib import Path


class CleanerProtoGenerator:
    @staticmethod
    def run():
        if importlib.metadata.version("grpcio-tools") != "1.75.1":
            raise SystemExit("Cleaner codegen requires grpcio-tools==1.75.1; use isolated tooling")
        root = Path(__file__).resolve().parents[1]
        subprocess.run(
            [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                "-I.",
                "--python_out=.",
                "--pyi_out=.",
                "--grpc_python_out=.",
                "hear/proto/cleaner_v2.proto",
            ],
            cwd=root,
            check=True,
        )


if __name__ == "__main__":
    CleanerProtoGenerator.run()
