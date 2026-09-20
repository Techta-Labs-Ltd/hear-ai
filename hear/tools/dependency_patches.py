import argparse
import ast
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


@dataclass(frozen=True)
class DependencyPatch:
    package: str
    target: str
    patch: str
    revision: str
    before_sha256: str
    after_sha256: str
    patch_sha256: str


class DependencyPatchManager:
    def __init__(self, root: Path | None = None):
        self.root = root or Path(__file__).resolve().parents[2]

    @staticmethod
    def digest(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def patched_content(source: bytes, patch: bytes) -> bytes:
        lines = source.decode().splitlines(keepends=True)
        result: list[str] = []
        position = 0
        active = False
        for line in patch.decode().splitlines(keepends=True):
            if line.startswith("@@"):
                match = re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", line)
                if match is None:
                    raise RuntimeError("invalid_patch_hunk")
                start = int(match.group(1)) - 1
                if start < position or start > len(lines):
                    raise RuntimeError("invalid_patch_position")
                result.extend(lines[position:start])
                position = start
                active = True
            elif active and line.startswith("+"):
                result.append(line[1:])
            elif active and (line.startswith(("-", " ")) or line == "\n"):
                expected = line if line == "\n" else line[1:]
                if position >= len(lines) or lines[position] != expected:
                    raise RuntimeError("patch_context_mismatch")
                if not line.startswith("-"):
                    result.append(lines[position])
                position += 1
            elif active:
                raise RuntimeError("unsupported_patch_content")
        if not active:
            raise RuntimeError("empty_patch")
        result.extend(lines[position:])
        return "".join(result).encode()

    @classmethod
    def apply_file(cls, target: Path, patch_file: Path, spec: DependencyPatch, check: bool) -> str:
        patch = patch_file.read_bytes()
        if cls.digest(patch) != spec.patch_sha256:
            raise RuntimeError(f"patch_digest_mismatch: {spec.patch}")
        original = target.read_bytes()
        current = cls.digest(original)
        if current == spec.after_sha256:
            return "verified"
        if current != spec.before_sha256:
            raise RuntimeError(f"unsupported_dependency_source: {spec.target}")
        if check:
            raise RuntimeError(
                f"dependency_patch_required: {spec.patch}; "
                "run uv run --no-sync python -m hear.tools.dependency_patches"
            )
        updated = cls.patched_content(original, patch)
        if cls.digest(updated) != spec.after_sha256:
            raise RuntimeError(f"patched_digest_mismatch: {spec.target}")
        ast.parse(updated)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=".hear-patch-", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(updated)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(target.stat().st_mode)
            if target.read_bytes() != original:
                raise RuntimeError("dependency_changed_during_patch")
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return "applied"

    def run(self, check: bool = False) -> dict[str, str]:
        manifest = json.loads((self.root / "patches/manifest.json").read_text())
        result = {}
        for entry in manifest:
            spec = DependencyPatch(**entry)
            try:
                package = distribution(spec.package)
            except PackageNotFoundError as exc:
                raise RuntimeError(f"dependency_not_installed: {spec.package}") from exc
            source = json.loads(package.read_text("direct_url.json") or "{}")
            if source.get("vcs_info", {}).get("commit_id") != spec.revision:
                raise RuntimeError(f"unsupported_dependency_revision: {spec.package}")
            target = Path(package.locate_file(spec.target))
            result[spec.package] = self.apply_file(
                target, self.root / "patches" / spec.patch, spec, check
            )
        return result

    @classmethod
    def main(cls) -> int:
        parser = argparse.ArgumentParser(description="Apply or verify pinned dependency patches")
        parser.add_argument("--check", action="store_true")
        args = parser.parse_args()
        try:
            print(json.dumps(cls().run(check=args.check), sort_keys=True))
        except (OSError, RuntimeError, ValueError) as exc:
            parser.exit(1, f"{exc}\n")
        return 0


if __name__ == "__main__":
    raise SystemExit(DependencyPatchManager.main())
