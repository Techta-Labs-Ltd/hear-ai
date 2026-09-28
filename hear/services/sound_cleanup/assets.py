"""Trusted deployment bundles only; no runtime downloads or user-selected paths."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from hear.runtime.cleaner.asset_probe import PinnedAssetProbe
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


@dataclass(frozen=True)
class SoundCleanupAssets:
    root: Path
    manifest_sha256: str
    manifest: dict

    @classmethod
    def load(cls, root: Path, expected: str):
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("sound_cleanup_manifest_digest_required")
        payload = PinnedAssetProbe.read_regular(root / "manifest.json", maximum_bytes=65536)
        if hashlib.sha256(payload).hexdigest() != expected:
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "sound_cleanup_manifest_mismatch"
            )
        manifest = json.loads(payload)
        if manifest.get("schema_version") != 1 or manifest.get("vad_frame_samples") != 512:
            raise CleanExecutionError(
                ErrorCode.ENGINE_UNAVAILABLE, "sound_cleanup_bundle_schema_mismatch"
            )
        assets = cls(root.resolve(), expected, manifest)
        assets.verify()
        return assets

    def path(self, name: str) -> Path:
        if name not in ("silero_vad.jit", "panns_sed.jit", "labels.csv"):
            raise ValueError("unknown_sound_cleanup_asset")
        return self.root / name

    def verify(self) -> None:
        for name in ("silero_vad.jit", "panns_sed.jit", "labels.csv"):
            record = self.manifest.get("files", {}).get(name, {})
            digest = record.get("sha256", "")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise CleanExecutionError(
                    ErrorCode.ENGINE_UNAVAILABLE, "sound_cleanup_asset_digest_missing"
                )
            PinnedAssetProbe.sha256(self.path(name), digest, maximum_bytes=400_000_000)
