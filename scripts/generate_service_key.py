#!/usr/bin/env python3
"""Generate a backend service key and optionally install only its hash in .env.

The plaintext key is printed once for the backend/API secret store. The AI
server keeps only the SHA-256 digest in BACKEND_REGISTRY_JSON, so a database or
`.env` leak cannot be used directly as an API credential.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from pathlib import Path


class ServiceKeyGenerator:
    @staticmethod
    def replace_registry(env_path: Path, backend_id: str, digest: str) -> None:
        lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)
        for index, line in enumerate(lines):
            if not line.startswith("BACKEND_REGISTRY_JSON="):
                continue
            raw = line.partition("=")[2].strip()
            registry = json.loads(raw)
            if not isinstance(registry, dict) or backend_id not in registry:
                raise SystemExit(
                    f"backend {backend_id!r} is not present in BACKEND_REGISTRY_JSON; "
                    "add its allowed storage URLs/bucket first"
                )
            registration = registry[backend_id]
            if not isinstance(registration, dict):
                raise SystemExit(f"backend {backend_id!r} registration must be an object")
            registration["service_key_sha256"] = digest
            lines[index] = "BACKEND_REGISTRY_JSON=" + json.dumps(
                registry, separators=(",", ":"), sort_keys=True
            ) + "\n"
            env_path.write_text("".join(lines), encoding="utf-8")
            env_path.chmod(0o600)
            return
        raise SystemExit(f"BACKEND_REGISTRY_JSON was not found in {env_path}")

    @staticmethod
    def run() -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--backend-id", required=True)
        parser.add_argument("--env-file", type=Path, default=Path(".env"))
        parser.add_argument(
            "--write",
            action="store_true",
            help="replace that backend's service_key_sha256 in --env-file",
        )
        args = parser.parse_args()

        key = secrets.token_urlsafe(48)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        if args.write:
            if not args.env_file.is_file():
                raise SystemExit(f"env file does not exist: {args.env_file}")
            ServiceKeyGenerator.replace_registry(args.env_file, args.backend_id, digest)

        print(f"HEAR_SERVICE_KEY={key}")
        print(f"BACKEND_ID={args.backend_id}")
        print(f"SERVICE_KEY_SHA256={digest}")
        if args.write:
            print(
                f"Installed digest in {args.env_file}; store HEAR_SERVICE_KEY in the "
                "backend secret store."
            )
        else:
            print("The AI .env must contain this digest under the matching backend registration.")
        return 0


if __name__ == "__main__":
    raise SystemExit(ServiceKeyGenerator.run())
