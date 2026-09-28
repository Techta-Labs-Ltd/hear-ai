"""Low-VRAM loader configuration and immutable runtime-view tests."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from hear.bootstrap import RuntimeBootstrap
from hear.config import RuntimeSettings
from hear.inference.fish_nf4_assets import FishNF4Assets
from hear.inference.manifest import ModelManifest
from hear.runtime.roles import WorkerRole


def test_nf4_is_explicit_reconstruction_default():
    settings = RuntimeSettings.from_environment({})
    assert settings.fish_speech_bnb_mode == "nf4"
    with pytest.raises(ValueError):
        RuntimeSettings.from_environment({"FISH_SPEECH_BNB_MODE": "nf8"})


def test_nf4_manifest_is_pinned_and_does_not_claim_commercial_permission():
    model = ModelManifest(Path("hear/model_manifest.json")).models_for(WorkerRole.RECONSTRUCTION)[0]
    assert model.repo_id == FishNF4Assets.MODEL_REPO
    assert model.revision == FishNF4Assets.MODEL_REVISION
    assert model.relative_path == FishNF4Assets.RELATIVE_PATH
    assert "model.pth" in model.required_files
    assert "model.safetensors.index.json" not in model.required_files
    assert set(model.required_files) == set(model.file_hashes)
    assert model.license_status == "permission_required"


def test_reconstruction_can_use_persistent_model_root_without_changing_cleaner(tmp_path):
    fish_root = tmp_path / "persistent"
    common_root = tmp_path / "common"
    bootstrap = RuntimeBootstrap(
        {"HEAR_MODEL_ROOT": str(common_root), "FISH_SPEECH_MODEL_ROOT": str(fish_root)}
    )
    assert bootstrap.readiness(WorkerRole.RECONSTRUCTION)._model_root == fish_root
    assert bootstrap.readiness(WorkerRole.MAGIC_CLEAN_NATURAL)._model_root == common_root


def test_runtime_requires_identical_hardlinked_weight_bytes(tmp_path, monkeypatch):
    raw = tmp_path / FishNF4Assets.RELATIVE_PATH
    raw.mkdir(parents=True)
    runtime = raw.with_name(raw.name + "-runtime")
    runtime.mkdir()
    for name in FishNF4Assets.LINKED_FILES:
        (raw / name).write_bytes(b"immutable-test-asset")
        os.link(raw / name, runtime / name)
    data = json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"}).encode()
    (runtime / "tokenizer_config.json").write_bytes(data)
    monkeypatch.setattr(
        FishNF4Assets, "TOKENIZER_METADATA_SHA256", hashlib.sha256(data).hexdigest()
    )
    assert FishNF4Assets.runtime_path(tmp_path) == runtime
    (runtime / "model.pth").unlink()
    (runtime / "model.pth").write_bytes(b"different-unverified-model")
    with pytest.raises(RuntimeError, match="view_mismatch"):
        FishNF4Assets.runtime_path(tmp_path)


def test_both_provider_images_use_pinned_nf4_loader():
    docker = Path("Dockerfile").read_text()
    assert FishNF4Assets.SOURCE_REPO in docker
    assert FishNF4Assets.SOURCE_REVISION in docker
    dependencies = Path("deploy/runtime/pyproject.toml").read_text()
    assert '"bitsandbytes==0.49.2"' in dependencies
    assert '"inflect==7.5.0"' in dependencies
