import importlib
from fnmatch import fnmatch
from pathlib import Path

import pytest

from hear.inference.manifest import ModelManifest
from hear.runtime.roles import WorkerRole
from hear.tools.model_provisioning import ModelProvisioner


def test_role_provisioner_requests_only_selected_models(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """
{
  "manifest_version": 1,
  "models": [
    {
      "logical_name": "asr",
      "repo_id": "example/asr",
      "revision": "1111111111111111111111111111111111111111",
      "relative_path": "asr",
      "roles": ["transcription", "pipeline"],
      "required_files": ["config.json"],
      "engine_adapter": "test_asr",
      "provenance_url": "https://example.com/asr",
      "license_name": "Apache-2.0",
      "license_status": "verified",
      "license_url": "https://example.com/asr/license"
    },
    {
      "logical_name": "tts",
      "repo_id": "example/tts",
      "revision": "2222222222222222222222222222222222222222",
      "relative_path": "tts",
      "roles": ["reconstruction"],
      "required_files": ["config.json"],
      "engine_adapter": "test_tts",
      "provenance_url": "https://example.com/tts",
      "license_name": "Apache-2.0",
      "license_status": "verified",
      "license_url": "https://example.com/tts/license"
    }
  ]
}
"""
    )
    calls = []

    def fake_provision(model_root, role, *, enabled_features, cache_dir, acknowledge_license_review=False):
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

    transcription = {item.logical_name for item in manifest.models_for(WorkerRole.TRANSCRIPTION)}
    reconstruction = {item.logical_name for item in manifest.models_for(WorkerRole.RECONSTRUCTION)}
    pipeline = {item.logical_name for item in manifest.models_for(WorkerRole.PIPELINE)}

    assert transcription == {"qwen3-asr-1.7b", "qwen3-forced-aligner"}
    assert "fish-speech-s2-pro" not in transcription
    assert reconstruction == {"fish-speech-s2-pro"}
    assert "fish-speech-s2-pro" not in pipeline
    assert "dnsmos" not in pipeline


def test_optional_pipeline_llm_is_not_default():
    manifest = ModelManifest(Path("hear/model_manifest.json"))

    default = {item.logical_name for item in manifest.models_for(WorkerRole.PIPELINE)}
    enabled = {
        item.logical_name
        for item in manifest.models_for(
            WorkerRole.PIPELINE,
            enabled_features=frozenset({"qwen_llm"}),
        )
    }

    assert "qwen2.5-7b-instruct-awq" not in default
    assert "qwen2.5-7b-instruct-awq" in enabled


def test_unapproved_model_license_blocks_provisioning_before_network_access(tmp_path):
    manifest = ModelManifest(Path("hear/model_manifest.json"))
    model_root = tmp_path / "models"
    provisioner = ModelProvisioner(manifest, model_root)

    with pytest.raises(RuntimeError, match="model license approval required"):
        provisioner.provision(WorkerRole.RECONSTRUCTION)

    assert not model_root.exists()


def test_snapshot_download_excludes_duplicate_weights_and_preserves_required_files(
    monkeypatch, tmp_path
):
    requests = []

    def download(**kwargs):
        requests.append(kwargs)
        root = kwargs["local_dir"]
        root.mkdir(parents=True)
        spec = next(model for model in manifest.models if model.repo_id == kwargs["repo_id"])
        for name in spec.required_files:
            (root / name).write_bytes(b"mock-pinned-asset")
        return str(root)

    manifest = ModelManifest(Path("hear/model_manifest.json"))
    monkeypatch.setattr(importlib.import_module("huggingface_hub"), "snapshot_download", download)
    manifest.provision(tmp_path / "models", WorkerRole.PIPELINE)
    for request, spec in zip(requests, manifest.models_for(WorkerRole.PIPELINE), strict=True):
        patterns = request["allow_patterns"]
        assert all(
            any(fnmatch(name, pattern) for pattern in patterns) for name in spec.required_files
        )
        assert any(fnmatch("generation_config.json", pattern) for pattern in patterns)
        assert not any(fnmatch("alternate-export.onnx", pattern) for pattern in patterns)
        if "pytorch_model.bin" not in spec.required_files:
            assert not any(fnmatch("pytorch_model.bin", pattern) for pattern in patterns)
        assert request["revision"] == spec.revision
