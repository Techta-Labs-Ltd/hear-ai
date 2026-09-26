from pathlib import Path

from hear.inference.manifest import ModelManifest
from hear.runtime.roles import WorkerRole
from hear.tools.model_provisioning import ModelProvisioner


def test_role_provisioner_requests_only_selected_models(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """
{
  "models": [
    {
      "name": "asr",
      "repo_id": "example/asr",
      "revision": "1111111111111111111111111111111111111111",
      "relative_path": "asr",
      "roles": ["transcription", "pipeline"],
      "required_files": ["config.json"]
    },
    {
      "name": "tts",
      "repo_id": "example/tts",
      "revision": "2222222222222222222222222222222222222222",
      "relative_path": "tts",
      "roles": ["reconstruction"],
      "required_files": ["config.json"]
    }
  ]
}
"""
    )
    calls = []

    def fake_provision(model_root, role, *, enabled_features, cache_dir):
        calls.append((model_root, role, enabled_features, cache_dir))
        return {"asr": str(model_root / "asr")}

    manifest = ModelManifest(manifest_path)
    monkeypatch.setattr(manifest, "provision", fake_provision)
    provisioner = ModelProvisioner(
        manifest,
        tmp_path / "models",
        cache_dir=tmp_path / "cache",
    )
    result = provisioner.provision(WorkerRole.TRANSCRIPTION)

    assert set(result) == {"asr"}
    assert calls == [
        (
            tmp_path / "models",
            WorkerRole.TRANSCRIPTION,
            frozenset(),
            tmp_path / "cache",
        )
    ]


def test_real_manifest_role_sets_are_isolated():
    manifest = ModelManifest(Path("hear/model_manifest.json"))

    transcription = {
        item.name for item in manifest.models_for(WorkerRole.TRANSCRIPTION)
    }
    reconstruction = {
        item.name for item in manifest.models_for(WorkerRole.RECONSTRUCTION)
    }
    pipeline = {
        item.name for item in manifest.models_for(WorkerRole.PIPELINE)
    }

    assert transcription == {"qwen3-asr-1.7b", "qwen3-forced-aligner"}
    assert "fish-speech-s2-pro" not in transcription
    assert reconstruction == {"fish-speech-s2-pro", "dnsmos"}
    assert "fish-speech-s2-pro" not in pipeline
    assert "dnsmos" not in pipeline


def test_optional_pipeline_llm_is_not_default():
    manifest = ModelManifest(Path("hear/model_manifest.json"))

    default = {item.name for item in manifest.models_for(WorkerRole.PIPELINE)}
    enabled = {
        item.name
        for item in manifest.models_for(
            WorkerRole.PIPELINE,
            enabled_features=frozenset({"qwen_llm"}),
        )
    }

    assert "qwen2.5-7b-instruct" not in default
    assert "qwen2.5-7b-instruct" in enabled
