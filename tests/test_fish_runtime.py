"""Fish S2 Pro runtime configuration and model storage policy tests."""

from pathlib import Path

import pytest

from hear.bootstrap import RuntimeBootstrap
from hear.config import PROJECT_ROOT, RuntimeSettings
from hear.inference.manifest import ModelManifest
from hear.runtime.roles import WorkerRole

UPSTREAM_REPO = "https://github.com/fishaudio/fish-speech.git"
UPSTREAM_REVISION = "214da3cd841bda85da2496b96cd3c4d7edb1337e"


def test_fish_source_defaults_to_pinned_upstream_checkout():
    settings = RuntimeSettings.from_environment({})
    assert settings.fish_speech_home == Path("/opt/fish-speech")
    assert not hasattr(settings, "fish_speech_bnb_mode")


def test_official_bf16_manifest_is_fully_hashed_and_still_requires_permission():
    model = ModelManifest(Path("hear/model_manifest.json")).models_for(WorkerRole.RECONSTRUCTION)[0]
    assert model.repo_id == "fishaudio/s2-pro"
    assert model.relative_path == "fish-speech/s2-pro"
    assert "model.safetensors.index.json" in model.required_files
    assert "codec.pth" in model.required_files
    assert "model.pth" not in model.required_files
    assert set(model.required_files) == set(model.file_hashes)
    assert model.license_status == "permission_required"


def test_reconstruction_checkpoint_resolves_through_manifest_and_overrides(tmp_path):
    fish_root = tmp_path / "persistent"
    common_root = tmp_path / "common"
    bootstrap = RuntimeBootstrap(
        {"HEAR_MODEL_ROOT": str(common_root), "FISH_SPEECH_MODEL_ROOT": str(fish_root)}
    )
    assert bootstrap.readiness(WorkerRole.RECONSTRUCTION)._model_root == fish_root
    assert bootstrap.readiness(WorkerRole.MAGIC_CLEAN_NATURAL)._model_root == common_root
    custom = tmp_path / "custom-fish"
    override = RuntimeBootstrap(
        {
            "HEAR_MODEL_ROOT": str(common_root),
            "HEAR_MODEL_PATHS_JSON": f'{{"fish-speech-s2-pro": "{custom}"}}',
        }
    )
    assert override._model_path("fish-speech-s2-pro") == custom


def test_provisioning_requires_explicit_license_acknowledgement(tmp_path):
    manifest = ModelManifest(Path("hear/model_manifest.json"))
    with pytest.raises(RuntimeError, match="license approval required"):
        manifest.provision(tmp_path, WorkerRole.RECONSTRUCTION)


def test_images_pin_upstream_fish_source_without_quantisation_packages():
    docker = Path("Dockerfile").read_text()
    assert UPSTREAM_REPO in docker and UPSTREAM_REVISION in docker
    assert "int4" not in docker and "nf4" not in docker.lower()
    assert "--acknowledge-license-review" in docker
    dependencies = Path("deploy/runtime/pyproject.toml").read_text()
    assert "bitsandbytes" not in dependencies
    assert '"inflect==7.5.0"' in dependencies


@pytest.mark.parametrize("key", ["HEAR_MODEL_ROOT", "FISH_SPEECH_MODEL_ROOT", "FISH_SPEECH_HOME"])
def test_model_paths_reject_source_checkout(key):
    with pytest.raises(ValueError, match="model_storage_must_not_use_source_checkout"):
        RuntimeSettings.from_environment({key: str(PROJECT_ROOT / "models")})


@pytest.mark.parametrize("root", ["/opt/hear-ai-models", "/root/hear-ai-models-test"])
def test_root_storage_model_roots_allowed(root):
    result = RuntimeSettings.from_environment({"FISH_SPEECH_MODEL_ROOT": root})
    assert result.fish_speech_model_root == Path(root).resolve()


@pytest.mark.parametrize(
    "variable,value",
    [
        ("HEAR_MODEL_ROOT", "/workspace/hear-ai-models"),
        ("FISH_SPEECH_MODEL_ROOT", "/runpod-volume/hear-ai/models"),
        ("HEAR_MAGIC_CLEAN_MODEL_DIR", "/workspace/df3"),
        ("HEAR_SOUND_CLEANUP_BUNDLE", "/workspace/bundle"),
        ("HEAR_MODEL_PATHS_JSON", '{"toxic-bert": "/workspace/toxic"}'),
    ],
)
def test_network_volume_model_storage_rejected(variable, value):
    with pytest.raises(ValueError, match="model_storage_must_not_use_network_volume"):
        RuntimeSettings.from_environment({variable: value})


def test_model_symlink_cannot_hide_storage_inside_checkout(tmp_path):
    link = tmp_path / "models"
    link.symlink_to(PROJECT_ROOT / "models")
    with pytest.raises(ValueError, match="model_storage_must_not_use_source_checkout"):
        RuntimeSettings.from_environment({"HEAR_MODEL_ROOT": str(link)})
