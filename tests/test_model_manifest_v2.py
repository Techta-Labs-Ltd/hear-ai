import json
from pathlib import Path

import pytest

from hear.inference.manifest import ModelManifest
from hear.runtime.roles import WorkerRole


class TestModelManifest:
    def test_manifest_revisions_are_pinned(self):
        manifest = ModelManifest(Path("hear/model_manifest.json"))
        assert manifest.models
        for model in manifest.models:
            if model.source_type == "huggingface":
                assert model.revision is not None
                assert len(model.revision) == 40
            else:
                assert model.sha256 is not None
                assert len(model.sha256) == 64

    def test_transcription_loads_only_asr_models(self):
        manifest = ModelManifest(Path("hear/model_manifest.json"))
        names = {model.logical_name for model in manifest.models_for(WorkerRole.TRANSCRIPTION)}
        assert names == {"qwen3-asr-1.7b", "qwen3-forced-aligner"}

    def test_reconstruction_has_only_required_model_families(self):
        manifest = ModelManifest(Path("hear/model_manifest.json"))
        names = {model.logical_name for model in manifest.models_for(WorkerRole.RECONSTRUCTION)}
        assert names == {
            "fish-speech-s2-pro",
        }

    def test_pipeline_llm_is_opt_in(self):
        manifest = ModelManifest(Path("hear/model_manifest.json"))
        without_llm = {
            model.logical_name for model in manifest.models_for(WorkerRole.PIPELINE)
        }
        with_llm = {
            model.logical_name
            for model in manifest.models_for(
                WorkerRole.PIPELINE,
                enabled_features=frozenset({"qwen_llm"}),
            )
        }
        assert "qwen2.5-7b-instruct-awq" not in without_llm
        assert "qwen2.5-7b-instruct-awq" in with_llm

    def test_unreviewed_model_licenses_block_runtime_roles(self):
        manifest = ModelManifest(Path("hear/model_manifest.json"))

        assert manifest.license_blockers(WorkerRole.PIPELINE) == ()
        assert manifest.license_blockers(WorkerRole.TRANSCRIPTION) == ()
        assert manifest.license_blockers(WorkerRole.RECONSTRUCTION) == (
            "fish-speech-s2-pro:permission_required",
        )
        assert all("sam-audio" not in model.logical_name for model in manifest.models)

    def test_manifest_json_has_unique_names_and_paths(self):
        payload = json.loads(Path("hear/model_manifest.json").read_text())
        names = [item["logical_name"] for item in payload["models"]]
        paths = [item["relative_path"] for item in payload["models"]]
        assert len(names) == len(set(names))
        assert len(paths) == len(set(paths))

    def test_model_paths_can_be_overridden_per_logical_name(self, tmp_path):
        manifest = ModelManifest(Path("hear/model_manifest.json"))
        root = tmp_path / "models"
        custom = tmp_path / "custom-asr"
        for model in manifest.models_for(WorkerRole.TRANSCRIPTION):
            target = custom if model.logical_name == "qwen3-asr-1.7b" else root / model.relative_path
            target.mkdir(parents=True)
            for name in model.required_files:
                (target / name).write_bytes(b"x")
        overrides = {"qwen3-asr-1.7b": custom}
        assert manifest.local_path(root, "qwen3-asr-1.7b", overrides) == custom
        assert manifest.local_path(root, "qwen3-forced-aligner", overrides) == root / "qwen3-forced-aligner"
        assert manifest.validate_local(root, WorkerRole.TRANSCRIPTION, overrides=overrides) == ()
        assert "qwen3-asr-1.7b:directory" in manifest.validate_local(root, WorkerRole.TRANSCRIPTION)
        with pytest.raises(ValueError, match="unknown_model_override:nope"):
            manifest.validate_overrides({"nope": custom})
        with pytest.raises(RuntimeError, match="unknown_model:nope"):
            manifest.spec("nope")
