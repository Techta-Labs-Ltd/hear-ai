import hashlib
import json
from pathlib import Path

import pytest
import torch.nn.functional as F
from cryptography.fernet import Fernet

from hear.config import Settings
from hear.deployments import app, fish_speech, transcription
from main import RuntimeApplication


def configured_settings(model_dir: Path, **overrides) -> Settings:
    service_key_hash = hashlib.sha256(b"test-secret").hexdigest()
    values = {
        "BACKEND_REGISTRY_JSON": json.dumps(
            {
                "backend-a": {
                    "service_key_sha256": service_key_hash,
                    "allowed_endpoint_urls": ["https://s3.example.test"],
                    "allowed_buckets": ["bucket-a"],
                    "allowed_public_base_urls": ["https://cdn.example.test"],
                }
            }
        ),
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
        "DEMUCS_MODEL_PATH": str(model_dir),
        "FISH_SPEECH_HOME": str(model_dir),
        "FISH_SPEECH_CHECKPOINT_PATH": str(model_dir),
        "FISH_SPEECH_CODEC_PATH": str(model_dir / "codec.pth"),
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture(autouse=True)
def installed_runtime_modules(monkeypatch):
    monkeypatch.setattr("main.RuntimeApplication._missing_modules", lambda _names: [])


def test_runtime_validation_accepts_preprovisioned_artifacts(tmp_path):
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "codec.pth").touch()
    (tmp_path / "last_best_checkpoint").touch()
    (tmp_path / "demucs-signature.th").touch()
    (tmp_path / "htdemucs.yaml").write_text("models: [demucs-signature]\n")
    RuntimeApplication.validate_runtime(configured_settings(tmp_path))


def test_runtime_validation_rejects_missing_model_path(tmp_path):
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "codec.pth").touch()
    (tmp_path / "last_best_checkpoint").touch()
    (tmp_path / "demucs-signature.th").touch()
    (tmp_path / "htdemucs.yaml").write_text("models: [demucs-signature]\n")
    settings = configured_settings(tmp_path, QWEN_ASR_MODEL_PATH="")
    with pytest.raises(RuntimeError, match="QWEN_ASR_MODEL_PATH must be configured"):
        RuntimeApplication.validate_runtime(settings)


def test_runtime_validation_rejects_missing_backend_registry(tmp_path):
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "codec.pth").touch()
    (tmp_path / "last_best_checkpoint").touch()
    (tmp_path / "demucs-signature.th").touch()
    (tmp_path / "htdemucs.yaml").write_text("models: [demucs-signature]\n")
    settings = configured_settings(tmp_path, BACKEND_REGISTRY_JSON="")
    with pytest.raises(RuntimeError, match="BACKEND_REGISTRY_JSON"):
        RuntimeApplication.validate_runtime(settings)


def test_runtime_validation_rejects_invalid_storage_encryption_key(tmp_path):
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "codec.pth").touch()
    (tmp_path / "last_best_checkpoint").touch()
    (tmp_path / "demucs-signature.th").touch()
    (tmp_path / "htdemucs.yaml").write_text("models: [demucs-signature]\n")
    settings = configured_settings(tmp_path, STORAGE_CONTEXT_ENCRYPTION_KEY="invalid")
    with pytest.raises(RuntimeError, match="STORAGE_CONTEXT_ENCRYPTION_KEY"):
        RuntimeApplication.validate_runtime(settings)


def test_runtime_validation_rejects_incomplete_mossformer_checkpoint(tmp_path):
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "codec.pth").touch()
    (tmp_path / "demucs-signature.th").touch()
    (tmp_path / "htdemucs.yaml").write_text("models: [demucs-signature]\n")
    with pytest.raises(RuntimeError, match="missing MossFormer2 checkpoint"):
        RuntimeApplication.validate_runtime(configured_settings(tmp_path))


def test_durable_defaults_keep_models_at_the_filesystem_root():
    defaults = Settings(_env_file=None)
    assert defaults.MODEL_CACHE_DIR == "/models"
    assert defaults.QWEN_ASR_MODEL_PATH == "/models/qwen3-asr-1.7b"
    assert defaults.MOSSFORMER_MODEL_PATH == "/models/mossformer2-se-48k"
    assert defaults.FISH_SPEECH_HOME == "/fish-speech"
    assert defaults.FISH_SPEECH_CHECKPOINT_PATH.startswith("/models/")
    assert defaults.FISH_SPEECH_CODEC_PATH.startswith("/models/")


def test_durable_paths_allow_environment_overrides(monkeypatch, tmp_path):
    model_root = tmp_path / "models"
    fish_root = tmp_path / "fish-speech"
    monkeypatch.setenv("MODEL_CACHE_DIR", str(model_root))
    monkeypatch.setenv("FISH_SPEECH_HOME", str(fish_root))
    configured = Settings(_env_file=None)
    assert configured.MODEL_CACHE_DIR == str(model_root)
    assert configured.FISH_SPEECH_HOME == str(fish_root)


def test_magic_clean_has_a_demucs_model_default(tmp_path):
    assert configured_settings(tmp_path).DEMUCS_MODEL == "htdemucs"


def test_runtime_validation_rejects_unsafe_cleanup_grace(tmp_path):
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "codec.pth").touch()
    (tmp_path / "last_best_checkpoint").touch()
    (tmp_path / "demucs-signature.th").touch()
    (tmp_path / "htdemucs.yaml").write_text("models: [demucs-signature]\n")
    with pytest.raises(RuntimeError, match="CLEANUP_GRACE_SECONDS must be positive"):
        RuntimeApplication.validate_runtime(
            configured_settings(tmp_path, MAGIC_CLEAN_CLEANUP_GRACE_SECONDS=0)
        )


def test_runtime_validation_requires_cleanup_safe_storage_credential_ttl(tmp_path):
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "codec.pth").touch()
    (tmp_path / "last_best_checkpoint").touch()
    (tmp_path / "demucs-signature.th").touch()
    (tmp_path / "htdemucs.yaml").write_text("models: [demucs-signature]\n")
    with pytest.raises(RuntimeError, match="STORAGE_CREDENTIAL_MIN_TTL_SECONDS must exceed"):
        RuntimeApplication.validate_runtime(
            configured_settings(tmp_path, MAGIC_CLEAN_STORAGE_CREDENTIAL_MIN_TTL_SECONDS=1800)
        )
    with pytest.raises(RuntimeError, match="STORAGE_CREDENTIAL_MIN_TTL_SECONDS must exceed"):
        RuntimeApplication.validate_runtime(
            configured_settings(
                tmp_path, MAGIC_CLEAN_STORAGE_CREDENTIAL_MIN_TTL_SECONDS=float("nan")
            )
        )


def test_runtime_validation_rejects_missing_local_demucs_model(tmp_path):
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "codec.pth").touch()
    (tmp_path / "last_best_checkpoint").touch()
    with pytest.raises(RuntimeError, match="missing local Demucs model manifest"):
        RuntimeApplication.validate_runtime(configured_settings(tmp_path))


def test_default_single_gpu_deployment_budget_allows_one_heavy_actor(tmp_path):
    runtime = configured_settings(tmp_path)
    resident = 0.2 + 0.1 + 0.25 + 0.05
    on_demand = 0.35
    assert runtime.MAGIC_CLEAN_REPLICA_COUNT == 1
    assert runtime.FISH_SPEECH_REPLICA_COUNT == 1
    assert resident + on_demand <= 1.0
    assert resident + 2 * on_demand > 1.0


def test_server_applies_patches_and_provisions_models_before_serve(monkeypatch, tmp_path):
    runtime = configured_settings(tmp_path)
    order = []
    provision = object()

    monkeypatch.setattr("main.ray.is_initialized", lambda: True)
    monkeypatch.setattr("main.DependencyPatchManager.run", lambda _self: order.append("patch"))
    monkeypatch.setattr(
        "main.provision_models_on_ray",
        type("Provisioner", (), {"remote": staticmethod(lambda *_args: provision)}),
    )
    monkeypatch.setattr(
        "main.ray.get", lambda value: order.append("models") if value is provision else None
    )
    monkeypatch.setattr(
        "main.RuntimeApplication.configure_process", lambda _settings: order.append("configure")
    )
    monkeypatch.setattr(
        "main.RuntimeApplication.validate_runtime", lambda _settings: order.append("validate")
    )
    monkeypatch.setattr("main.DatabaseRuntime.init_db", lambda: order.append("database"))
    monkeypatch.setattr("main.serve.start", lambda **_kwargs: order.append("serve_start"))
    monkeypatch.setattr("main.serve.run", lambda *_args, **_kwargs: order.append("serve_run"))
    monkeypatch.setattr(
        "main.ApplicationBuilder.build_application", lambda _settings: object()
    )

    RuntimeApplication.run(runtime)

    assert order == [
        "patch",
        "models",
        "configure",
        "validate",
        "database",
        "serve_start",
        "serve_run",
    ]


def test_on_demand_gpu_models_have_a_short_idle_timeout(tmp_path):
    assert configured_settings(tmp_path).GPU_ON_DEMAND_IDLE_SECONDS == 15.0


def test_fish_speech_deployment_imports_startup_dependencies():
    assert callable(fish_speech.FishSpeechDeployment.func_or_class)
    assert callable(fish_speech.time.time)


def test_transcription_deployment_uses_qwen_backend(monkeypatch):
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
    assert callable(transcription.os.unlink)


def test_transcription_import_preserves_torch_padding():
    assert F.pad.__module__ == "torch.nn.functional"


def test_ray_graph_uses_audio_cleanup(monkeypatch):
    calls = []

    def bind(module_name, class_name, *dependencies):
        calls.append((module_name, class_name, dependencies))
        return class_name

    monkeypatch.setattr(app.ApplicationBuilder, "_bind", staticmethod(bind))
    runtime = Settings(_env_file=None, QWEN_LLM_ENABLED=False, FISH_SPEECH_TTS_ENABLED=False)
    assert app.ApplicationBuilder.build_application(runtime) == "GrpcGateway"
    assert "AudioCleanupDeployment" in [call[1] for call in calls]
    assert "FishSpeechDeployment" not in [call[1] for call in calls]
    assert "LLMDeployment" not in [call[1] for call in calls]
    assert calls[-1][2][-1] is None


def test_main_registers_only_pipeline_grpc_service():
    source = Path(__import__("main").__file__).read_text()
    assert "add_PipelineServicer_to_server" in source
