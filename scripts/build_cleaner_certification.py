import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from hear.inference.magic_clean import CleanerCertifications
from hear.runtime.cleaner.asset_probe import PinnedAssetProbe


class CleanerCertificationBuilder:
    MAX_DRAFT_BYTES = 64 * 1024
    MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
    PROFILES = ("natural",)

    @classmethod
    def build(cls, *, draft_path: Path, output_path: Path) -> dict:
        draft = PinnedAssetProbe.read_regular(
            draft_path,
            maximum_bytes=cls.MAX_DRAFT_BYTES,
        )
        try:
            payload = json.loads(draft)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ValueError("cleaner certification draft is invalid") from None
        if not isinstance(payload, dict):
            raise ValueError("cleaner certification draft must be an object")
        evidence_hashes = {}
        for profile in cls.PROFILES:
            entry = payload.get(profile)
            if entry is None:
                continue
            if not isinstance(entry, dict) or not isinstance(entry.get("limits"), dict):
                raise ValueError("cleaner profile limits are missing")
            path_value = entry["limits"].get("evidence_path")
            if not isinstance(path_value, str) or not path_value:
                raise ValueError("cleaner profile evidence path is missing")
            evidence = PinnedAssetProbe.read_regular(
                Path(path_value),
                maximum_bytes=cls.MAX_EVIDENCE_BYTES,
            )
            digest = hashlib.sha256(evidence).hexdigest()
            entry["limits"]["evidence_sha256"] = digest
            evidence_hashes[profile] = digest
        if not evidence_hashes:
            raise ValueError("cleaner certification draft has no profiles")
        certification = CleanerCertifications.model_validate(payload)
        output = (
            json.dumps(
                certification.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        cls._write_create_only(output_path, output)
        return {
            "certification": str(output_path),
            "certification_sha256": hashlib.sha256(output).hexdigest(),
            "profiles": sorted(evidence_hashes),
            "evidence_sha256": evidence_hashes,
        }

    @staticmethod
    def _write_create_only(path: Path, payload: bytes) -> None:
        if not path.is_absolute() or path.parent.resolve(strict=True) != path.parent:
            raise ValueError("cleaner certification output path must be absolute and canonical")
        descriptor, temporary_name = tempfile.mkstemp(prefix=".cleaner-cert-", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            offset = 0
            while offset < len(payload):
                count = os.write(descriptor, payload[offset:])
                if count <= 0:
                    raise OSError("cleaner certification write made no progress")
                offset += count
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            os.link(temporary, path, follow_symlinks=False)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    @classmethod
    def main(cls) -> None:
        parser = argparse.ArgumentParser(description="Build a pinned V11 Magic Clean certificate")
        parser.add_argument("--draft", required=True, type=Path)
        parser.add_argument("--output", required=True, type=Path)
        args = parser.parse_args()
        print(json.dumps(cls.build(draft_path=args.draft, output_path=args.output), sort_keys=True))


if __name__ == "__main__":
    CleanerCertificationBuilder.main()
