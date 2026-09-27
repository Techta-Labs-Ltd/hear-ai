import json
from pathlib import Path

from hear.health.service import RuntimeReadiness
from hear.inference.manifest import ModelManifest
from hear.runtime.roles import WorkerRole


class PatchVerifier:
    def __init__(self, failure: Exception | None = None) -> None:
        self._failure = failure

    def run(self, check: bool = False):
        if self._failure is not None:
            raise self._failure
        return {"whisperx": "verified"}


class TestRuntimeReadiness:
    def test_ready_when_patch_and_models_are_valid(self, tmp_path: Path):
        model_root = tmp_path / "models"
        asr = model_root / "asr"
        asr.mkdir(parents=True)
        (asr / "config.json").write_text("{}")
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "manifest_version": 1,
                    "models": [
                        {
                            "logical_name": "asr",
                            "repo_id": "owner/asr",
                            "revision": "a" * 40,
                            "relative_path": "asr",
                            "roles": ["transcription"],
                            "required_files": ["config.json"],
                            "engine_adapter": "test_asr",
                            "provenance_url": "https://example.com/asr",
                            "license_name": "Apache-2.0",
                            "license_status": "verified",
                            "license_url": "https://example.com/asr/license",
                        }
                    ]
                }
            )
        )
        readiness = RuntimeReadiness(
            WorkerRole.TRANSCRIPTION,
            ModelManifest(manifest_path),
            model_root,
            PatchVerifier(),
        )
        readiness.initialize()
        assert readiness.is_ready() is True

    def test_not_ready_when_patch_fails(self, tmp_path: Path):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({"manifest_version": 1, "models": []}))
        readiness = RuntimeReadiness(
            WorkerRole.TRANSCRIPTION,
            ModelManifest(manifest_path),
            tmp_path / "models",
            PatchVerifier(RuntimeError("patch failed")),
        )
        readiness.initialize()
        assert readiness.is_ready() is False
        assert readiness.snapshot()["patch_verified"] is False

    def test_non_asr_role_does_not_require_patch(self, tmp_path: Path):
        model_root = tmp_path / "models"
        fish = model_root / "fish"
        fish.mkdir(parents=True)
        (fish / "config.json").write_text("{}")
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "manifest_version": 1,
                    "models": [
                        {
                            "logical_name": "fish",
                            "repo_id": "owner/fish",
                            "revision": "b" * 40,
                            "relative_path": "fish",
                            "roles": ["reconstruction"],
                            "required_files": ["config.json"],
                            "engine_adapter": "test_fish",
                            "provenance_url": "https://example.com/fish",
                            "license_name": "Apache-2.0",
                            "license_status": "verified",
                            "license_url": "https://example.com/fish/license",
                        }
                    ]
                }
            )
        )
        readiness = RuntimeReadiness(
            WorkerRole.RECONSTRUCTION,
            ModelManifest(manifest_path),
            model_root,
            PatchVerifier(RuntimeError("unused")),
        )
        readiness.initialize()
        assert readiness.is_ready() is True
        assert readiness.snapshot()["patch_required"] is False

    def test_unreviewed_model_license_keeps_worker_unready(self, tmp_path: Path):
        model_root = tmp_path / "models"
        fish = model_root / "fish"
        fish.mkdir(parents=True)
        (fish / "config.json").write_text("{}")
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "manifest_version": 1,
                    "models": [
                        {
                            "logical_name": "fish",
                            "repo_id": "owner/fish",
                            "revision": "b" * 40,
                            "relative_path": "fish",
                            "roles": ["reconstruction"],
                            "required_files": ["config.json"],
                            "engine_adapter": "test_fish",
                            "provenance_url": "https://example.com/fish",
                            "license_name": "Research License",
                            "license_status": "permission_required",
                            "license_url": "https://example.com/fish/license",
                        }
                    ],
                }
            )
        )
        readiness = RuntimeReadiness(
            WorkerRole.RECONSTRUCTION,
            ModelManifest(manifest_path),
            model_root,
            PatchVerifier(),
        )
        readiness.initialize()

        assert readiness.is_ready() is False
        assert readiness.snapshot()["license_blockers"] == [
            "fish:permission_required",
        ]
