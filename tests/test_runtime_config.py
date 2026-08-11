import hashlib
import json
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from hear.config import Settings
from main import validate_runtime


def configured_settings(model_dir: Path, **overrides) -> Settings:
    service_key_hash = hashlib.sha256(b"test-secret").hexdigest()
    values = {
        "BACKEND_REGISTRY_JSON": json.dumps({
            "backend-a": {
                "service_key_sha256": service_key_hash,
                "allowed_endpoint_urls": ["https://s3.example.test"],
                "allowed_buckets": ["bucket-a"],
                "allowed_public_base_urls": ["https://cdn.example.test"],
            }
        }),
        "STORAGE_CONTEXT_ENCRYPTION_KEY": Fernet.generate_key().decode(),
        "DATABASE_URL": "postgresql+psycopg2://test:test@db:5432/test",
        "MODEL_CACHE_DIR": str(model_dir),
        "QWEN_ASR_MODEL_PATH": str(model_dir),
        "ALIGNER_MODEL_PATH": str(model_dir),
        "LLM_MODEL_PATH": str(model_dir),
        "TOXIC_MODEL_PATH": str(model_dir),
        "SENTIMENT_MODEL_PATH": str(model_dir),
        "NLI_MODEL_PATH": str(model_dir),
        "MOSSFORMER_MODEL_PATH": str(model_dir),
        "FISH_SPEECH_HOME": str(model_dir),
        "FISH_SPEECH_CHECKPOINT_PATH": str(model_dir),
        "RESOLVER_SEMANTIC_MODEL": str(model_dir),
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_runtime_validation_accepts_preprovisioned_artifacts(tmp_path):
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "codec.pth").touch()

    validate_runtime(configured_settings(tmp_path))


def test_runtime_validation_rejects_missing_model_path(tmp_path):
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "codec.pth").touch()
    settings = configured_settings(tmp_path, QWEN_ASR_MODEL_PATH="")

    with pytest.raises(RuntimeError, match="QWEN_ASR_MODEL_PATH must be configured"):
        validate_runtime(settings)


def test_runtime_validation_rejects_missing_backend_registry(tmp_path):
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "codec.pth").touch()
    settings = configured_settings(tmp_path, BACKEND_REGISTRY_JSON="")

    with pytest.raises(RuntimeError, match="BACKEND_REGISTRY_JSON"):
        validate_runtime(settings)


def test_runtime_validation_rejects_invalid_storage_encryption_key(tmp_path):
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "codec.pth").touch()
    settings = configured_settings(tmp_path, STORAGE_CONTEXT_ENCRYPTION_KEY="invalid")

    with pytest.raises(RuntimeError, match="STORAGE_CONTEXT_ENCRYPTION_KEY"):
        validate_runtime(settings)


def test_magic_clean_has_a_demucs_model_default(tmp_path):
    assert configured_settings(tmp_path).DEMUCS_MODEL == "htdemucs"


def test_default_single_gpu_deployment_budget_allows_one_heavy_actor(tmp_path):
    runtime = configured_settings(tmp_path)
    resident = (
        0.20  # transcription
        + 0.10  # small models
        + 0.25  # LLM
        + (runtime.RESOLVER_NUM_GPUS * runtime.RESOLVER_REPLICA_COUNT)
        + 0.05  # orchestrator
    )
    on_demand = 0.35

    assert runtime.MAGIC_CLEAN_REPLICA_COUNT == 1
    assert runtime.FISH_SPEECH_REPLICA_COUNT == 1
    assert runtime.RESOLVER_REPLICA_COUNT == 3
    assert runtime.RESOLVER_NUM_GPUS == 0.01
    assert resident + on_demand <= 1.0
    assert resident + (2 * on_demand) > 1.0


def test_on_demand_gpu_models_have_a_short_idle_timeout(tmp_path):
    assert configured_settings(tmp_path).GPU_ON_DEMAND_IDLE_SECONDS == 15.0


def test_fish_speech_deployment_imports_startup_dependencies():
    from hear.deployments import fish_speech

    assert fish_speech.os.path is not None
    assert callable(fish_speech.time.time)


def test_transcription_deployment_uses_qwen_backend(monkeypatch):
    from hear.deployments import transcription

    captured = {}

    def fake_load(model_path, **kwargs):
        captured["model_path"] = model_path
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(transcription, "load_qwen_asr_model", fake_load)
    deployment = transcription.TranscriptionDeployment.func_or_class()

    assert captured["model_path"] == transcription.settings.QWEN_ASR_MODEL_PATH
    assert captured["qwen_forced_aligner"] == transcription.settings.ALIGNER_MODEL_PATH
    assert captured["local_files_only"] is True
    assert captured["vad_options"]["vad_onset"] == 0.65
    del deployment


def test_transcription_deployment_imports_cleanup_dependency():
    from hear.deployments import transcription

    assert callable(transcription.os.unlink)


def test_ray_graph_uses_audio_cleanup_and_not_resolver():
    from hear.deployments import app

    source = Path(app.__file__).read_text()

    assert "AudioCleanupDeployment.bind()" in source
    assert "ResolverDeployment" not in source


def test_main_registers_only_pipeline_grpc_service():
    source = Path(__import__("main").__file__).read_text()

    assert "add_PipelineServicer_to_server" in source
    assert "add_ResolverServicer_to_server" not in source
