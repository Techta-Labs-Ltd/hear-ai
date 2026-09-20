from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import os
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

import ray
import yaml
from pydantic import ValidationError
from ray import serve

from hear.config import Settings
from hear.core.backend_registry import BackendRegistry
from hear.core.storage import StorageContexts
from hear.deployments.app import ApplicationBuilder
from hear.models.database import DatabaseRuntime
from hear.tools.dependency_patches import DependencyPatchManager
from hear.tools.model_provisioning import provision_models_on_ray

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
REQUIRED_MODULES = (
    "boto3",
    "clearvoice",
    "fish_speech",
    "psycopg2",
    "qwen_asr",
    "ray",
    "sqlalchemy",
    "torch",
    "torchaudio",
    "transformers",
    "whisperx",
)


class RuntimeApplication:
    @staticmethod
    def _missing_modules(names: Iterable[str]) -> list[str]:
        return [name for name in names if importlib.util.find_spec(name) is None]

    @staticmethod
    def validate_runtime(settings: Settings) -> None:
        """Fail before Ray starts when the immutable runtime is incomplete."""
        errors: list[str] = []
        try:
            DependencyPatchManager().run(check=True)
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(str(exc))
        required_modules = tuple(
            name
            for name in REQUIRED_MODULES
            if name != "fish_speech" or settings.FISH_SPEECH_TTS_ENABLED
        )
        missing_modules = RuntimeApplication._missing_modules(required_modules)
        if missing_modules:
            errors.append("missing Python modules: " + ", ".join(missing_modules))
        try:
            BackendRegistry.parse_backend_registry(settings.BACKEND_REGISTRY_JSON)
        except RuntimeError as exc:
            errors.append(str(exc))
        try:
            StorageContexts.validate_storage_encryption_key(settings.STORAGE_CONTEXT_ENCRYPTION_KEY)
        except RuntimeError as exc:
            errors.append(str(exc))
        if not settings.DATABASE_URL:
            errors.append("DATABASE_URL must be configured")
        ffmpeg_path = shutil.which("ffmpeg")
        ffprobe_path = shutil.which("ffprobe")
        if ffmpeg_path is None:
            errors.append("ffmpeg executable is unavailable")
        if ffprobe_path is None:
            errors.append("ffprobe executable is unavailable")
        if ffmpeg_path is not None:
            try:
                encoders = subprocess.run(
                    [ffmpeg_path, "-hide_banner", "-encoders"],
                    capture_output=True,
                    check=True,
                    text=True,
                    timeout=30,
                ).stdout
                filters = subprocess.run(
                    [ffmpeg_path, "-hide_banner", "-filters"],
                    capture_output=True,
                    check=True,
                    text=True,
                    timeout=30,
                ).stdout
                if "libmp3lame" not in encoders:
                    errors.append("ffmpeg is missing the libmp3lame encoder")
                for required_filter in ("loudnorm", "alimiter"):
                    if required_filter not in filters:
                        errors.append(f"ffmpeg is missing the {required_filter} filter")
            except (OSError, subprocess.SubprocessError) as exc:
                errors.append(f"could not inspect ffmpeg capabilities: {type(exc).__name__}")
        if not settings.MAGIC_CLEAN_ENGINE_REVISION.strip():
            errors.append("MAGIC_CLEAN_ENGINE_REVISION must be configured")
        if not 8 <= settings.MAGIC_CLEAN_MP3_BITRATE_KBPS <= 320:
            errors.append("MAGIC_CLEAN_MP3_BITRATE_KBPS must be between 8 and 320")
        if (
            not math.isfinite(settings.MAGIC_CLEAN_CLEANUP_GRACE_SECONDS)
            or settings.MAGIC_CLEAN_CLEANUP_GRACE_SECONDS <= 0
        ):
            errors.append("MAGIC_CLEAN_CLEANUP_GRACE_SECONDS must be positive")
        if (
            not math.isfinite(settings.MAGIC_CLEAN_STORAGE_CREDENTIAL_MIN_TTL_SECONDS)
            or settings.MAGIC_CLEAN_STORAGE_CREDENTIAL_MIN_TTL_SECONDS
            <= settings.MAGIC_CLEAN_CLEANUP_GRACE_SECONDS
        ):
            errors.append(
                "MAGIC_CLEAN_STORAGE_CREDENTIAL_MIN_TTL_SECONDS must exceed MAGIC_CLEAN_CLEANUP_GRACE_SECONDS"
            )
        if settings.MAGIC_CLEAN_CHUNK_SECONDS <= 0:
            errors.append("MAGIC_CLEAN_CHUNK_SECONDS must be positive")
        if settings.MAGIC_CLEAN_CHUNK_OVERLAP_SECONDS < 0:
            errors.append("MAGIC_CLEAN_CHUNK_OVERLAP_SECONDS cannot be negative")
        if settings.MAGIC_CLEAN_CHUNK_OVERLAP_SECONDS * 2 >= settings.MAGIC_CLEAN_CHUNK_SECONDS:
            errors.append("Magic Clean overlap margins must be shorter than the chunk")
        configured_directories = {
            "QWEN_ASR_MODEL_PATH": settings.QWEN_ASR_MODEL_PATH,
            "ALIGNER_MODEL_PATH": settings.ALIGNER_MODEL_PATH,
            "TOXIC_MODEL_PATH": settings.TOXIC_MODEL_PATH,
            "SENTIMENT_MODEL_PATH": settings.SENTIMENT_MODEL_PATH,
            "NLI_MODEL_PATH": settings.NLI_MODEL_PATH,
            "MOSSFORMER_MODEL_PATH": settings.MOSSFORMER_MODEL_PATH,
            "DEMUCS_MODEL_PATH": settings.DEMUCS_MODEL_PATH,
            "MODEL_CACHE_DIR": settings.MODEL_CACHE_DIR,
        }
        if settings.QWEN_LLM_ENABLED:
            configured_directories["LLM_MODEL_PATH"] = settings.LLM_MODEL_PATH
        if settings.FISH_SPEECH_TTS_ENABLED:
            configured_directories.update(
                {
                    "FISH_SPEECH_HOME": settings.FISH_SPEECH_HOME,
                    "FISH_SPEECH_CHECKPOINT_PATH": settings.FISH_SPEECH_CHECKPOINT_PATH,
                }
            )
        for variable, raw_path in configured_directories.items():
            if not raw_path:
                errors.append(f"{variable} must be configured")
                continue
            path = Path(raw_path)
            if not path.is_dir():
                errors.append(f"{variable} is not a directory: {path}")
        mossformer_checkpoint = Path(settings.MOSSFORMER_MODEL_PATH) / "last_best_checkpoint"
        if not mossformer_checkpoint.is_file():
            errors.append(f"missing MossFormer2 checkpoint: {mossformer_checkpoint}")
        demucs_root = Path(settings.DEMUCS_MODEL_PATH)
        demucs_candidates = (
            demucs_root / f"{settings.DEMUCS_MODEL}.yaml",
            demucs_root / f"{settings.DEMUCS_MODEL}.th",
        )
        demucs_manifest = next(
            (candidate for candidate in demucs_candidates if candidate.is_file()), None
        )
        if demucs_manifest is None:
            errors.append(
                "missing local Demucs model manifest: expected "
                + " or ".join(str(candidate) for candidate in demucs_candidates)
            )
        elif demucs_manifest.suffix == ".yaml":
            try:
                payload = yaml.safe_load(demucs_manifest.read_text())
                signatures = payload["models"]
                if not isinstance(signatures, list) or not signatures:
                    raise ValueError("models must be a non-empty list")
                missing_signatures = [
                    str(signature)
                    for signature in signatures
                    if not any(demucs_root.glob(f"{signature}*.th"))
                ]
                if missing_signatures:
                    errors.append(
                        "Demucs manifest references missing local models: "
                        + ", ".join(missing_signatures)
                    )
            except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
                errors.append(
                    f"invalid local Demucs model manifest: {demucs_manifest} ({type(exc).__name__})"
                )
        fish_checkpoint = Path(settings.FISH_SPEECH_CHECKPOINT_PATH)
        codec = Path(settings.FISH_SPEECH_CODEC_PATH)
        if settings.FISH_SPEECH_TTS_ENABLED and (not codec.is_file()):
            errors.append(f"missing Fish Speech checkpoint: {codec}")
        model_files = (
            fish_checkpoint / "model.safetensors",
            fish_checkpoint / "model.safetensors.index.json",
        )
        if settings.FISH_SPEECH_TTS_ENABLED and (not any(path.is_file() for path in model_files)):
            errors.append(
                "missing Fish Speech model: expected a single or sharded safetensors checkpoint"
            )
        if errors:
            raise RuntimeError("runtime validation failed:\n - " + "\n - ".join(errors))

    @staticmethod
    def configure_process(settings: Settings) -> None:
        logging.basicConfig(
            level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO), format=LOG_FORMAT
        )
        os.environ.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "GRPC_APPLICATION_NAME": settings.GRPC_APPLICATION_NAME,
            }
        )

    @staticmethod
    def run(settings: Settings) -> None:
        DependencyPatchManager().run()
        owns_ray = not ray.is_initialized()
        if owns_ray:
            ray.init(
                address=settings.RAY_ADDRESS,
                dashboard_host=settings.RAY_DASHBOARD_HOST,
                dashboard_port=settings.RAY_DASHBOARD_PORT,
                ignore_reinit_error=False,
            )
        try:
            ray.get(provision_models_on_ray.remote(settings.MODEL_CACHE_DIR))
            RuntimeApplication.configure_process(settings)
            RuntimeApplication.validate_runtime(settings)
            DatabaseRuntime.init_db()
            serve.start(
                proxy_location="EveryNode",
                http_options={"host": settings.HTTP_HOST, "port": settings.HTTP_PORT},
                grpc_options={
                    "port": settings.GRPC_PORT,
                    "grpc_servicer_functions": [
                        "hear.proto.pipeline_pb2_grpc.add_PipelineServicer_to_server"
                    ],
                },
            )
            serve.run(
                ApplicationBuilder.build_application(settings),
                blocking=True,
                name=settings.GRPC_APPLICATION_NAME,
                route_prefix="/",
            )
        finally:
            if owns_ray:
                if settings.RAY_ADDRESS == "local":
                    serve.shutdown()
                ray.shutdown()

    @staticmethod
    def main() -> int:
        parser = argparse.ArgumentParser(
            description="Run the Hear Ray Serve FastAPI and gRPC service"
        )
        parser.add_argument("--validate-only", action="store_true")
        args = parser.parse_args()
        try:
            settings = Settings()
            if args.validate_only:
                RuntimeApplication.configure_process(settings)
                RuntimeApplication.validate_runtime(settings)
            else:
                RuntimeApplication.run(settings)
        except (RuntimeError, ValidationError) as exc:
            logging.basicConfig(level=logging.ERROR, format=LOG_FORMAT)
            logging.getLogger(__name__).error("%s", exc)
            return 2
        return 0


if __name__ == "__main__":
    raise SystemExit(RuntimeApplication.main())
