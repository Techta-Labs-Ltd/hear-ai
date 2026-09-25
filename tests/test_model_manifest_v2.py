import json
from pathlib import Path

from hear.inference.manifest import ModelManifest
from hear.runtime.roles import WorkerRole


class TestModelManifest:
    def test_manifest_revisions_are_pinned(self):
        manifest = ModelManifest(Path("hear/model_manifest.json"))
        assert manifest.models
        assert all(len(model.revision) == 40 for model in manifest.models)

    def test_transcription_loads_only_asr_models(self):
        manifest = ModelManifest(Path("hear/model_manifest.json"))
        names = {model.name for model in manifest.models_for(WorkerRole.TRANSCRIPTION)}
        assert names == {"qwen3-asr-1.7b", "qwen3-forced-aligner"}

    def test_pipeline_llm_is_opt_in(self):
        manifest = ModelManifest(Path("hear/model_manifest.json"))
        without_llm = {model.name for model in manifest.models_for(WorkerRole.PIPELINE)}
        with_llm = {
            model.name
            for model in manifest.models_for(
                WorkerRole.PIPELINE,
                enabled_features=frozenset({"qwen_llm"}),
            )
        }
        assert "qwen2.5-7b-instruct" not in without_llm
        assert "qwen2.5-7b-instruct" in with_llm

    def test_manifest_json_has_unique_names_and_paths(self):
        payload = json.loads(Path("hear/model_manifest.json").read_text())
        names = [item["name"] for item in payload["models"]]
        paths = [item["relative_path"] for item in payload["models"]]
        assert len(names) == len(set(names))
        assert len(paths) == len(set(paths))