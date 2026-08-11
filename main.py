from __future__ import annotations

import argparse
import importlib.util
import logging
import os
from collections.abc import Iterable
from pathlib import Path

import ray
from pydantic import ValidationError
from ray import serve

from hear.config import Settings
from hear.core.backend_registry import parse_backend_registry
from hear.core.storage import validate_storage_encryption_key
from hear.deployments.app import build_application

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


def _missing_modules(names: Iterable[str]) -> list[str]:
    return [name for name in names if importlib.util.find_spec(name) is None]


def validate_runtime(settings: Settings) -> None:
    """Fail before Ray starts when the immutable runtime is incomplete."""

    errors: list[str] = []
    missing_modules = _missing_modules(REQUIRED_MODULES)
    if missing_modules:
        errors.append("missing Python modules: " + ", ".join(missing_modules))

    try:
        parse_backend_registry(settings.BACKEND_REGISTRY_JSON)
    except RuntimeError as exc:
        errors.append(str(exc))
    try:
        validate_storage_encryption_key(settings.STORAGE_CONTEXT_ENCRYPTION_KEY)
    except RuntimeError as exc:
        errors.append(str(exc))
    if not settings.DATABASE_URL:
        errors.append("DATABASE_URL must be configured")

    configured_directories = {
        "QWEN_ASR_MODEL_PATH": settings.QWEN_ASR_MODEL_PATH,
        "ALIGNER_MODEL_PATH": settings.ALIGNER_MODEL_PATH,
        "LLM_MODEL_PATH": settings.LLM_MODEL_PATH,
        "TOXIC_MODEL_PATH": settings.TOXIC_MODEL_PATH,
        "SENTIMENT_MODEL_PATH": settings.SENTIMENT_MODEL_PATH,
        "NLI_MODEL_PATH": settings.NLI_MODEL_PATH,
        "FISH_SPEECH_HOME": settings.FISH_SPEECH_HOME,
        "FISH_SPEECH_CHECKPOINT_PATH": settings.FISH_SPEECH_CHECKPOINT_PATH,
        "MOSSFORMER_MODEL_PATH": settings.MOSSFORMER_MODEL_PATH,
        "MODEL_CACHE_DIR": settings.MODEL_CACHE_DIR,
    }
    for variable, raw_path in configured_directories.items():
        if not raw_path:
            errors.append(f"{variable} must be configured")
            continue
        path = Path(raw_path)
        if not path.is_dir():
            errors.append(f"{variable} is not a directory: {path}")

    fish_checkpoint = Path(settings.FISH_SPEECH_CHECKPOINT_PATH)
    codec = fish_checkpoint / "codec.pth"
    if not codec.is_file():
        errors.append(f"missing Fish Speech checkpoint: {codec}")
    model_files = (
        fish_checkpoint / "model.safetensors",
        fish_checkpoint / "model.safetensors.index.json",
    )
    if not any(path.is_file() for path in model_files):
        errors.append(
            "missing Fish Speech model: expected a single or sharded safetensors checkpoint"
        )

    if errors:
        raise RuntimeError("runtime validation failed:\n - " + "\n - ".join(errors))


def configure_process(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format=LOG_FORMAT,
    )
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "GRPC_APPLICATION_NAME": settings.GRPC_APPLICATION_NAME,
        }
    )


def run(settings: Settings) -> None:
    configure_process(settings)
    validate_runtime(settings)

    owns_ray = not ray.is_initialized()
    if owns_ray:
        ray.init(
            address=settings.RAY_ADDRESS,
            dashboard_host=settings.RAY_DASHBOARD_HOST,
            dashboard_port=settings.RAY_DASHBOARD_PORT,
            ignore_reinit_error=False,
        )

    try:
        serve.start(
            proxy_location="EveryNode",
            http_options={
                "host": settings.HTTP_HOST,
                "port": settings.HTTP_PORT,
            },
            grpc_options={
                "port": settings.GRPC_PORT,
                "grpc_servicer_functions": [
                    "hear.proto.pipeline_pb2_grpc.add_PipelineServicer_to_server",
                ],
            },
        )
        serve.run(
            build_application(),
            blocking=True,
            name=settings.GRPC_APPLICATION_NAME,
            route_prefix="/",
        )
    finally:
        if owns_ray:
            serve.shutdown()
            ray.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Hear Ray Serve FastAPI and gRPC service"
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        settings = Settings()
        if args.validate_only:
            configure_process(settings)
            validate_runtime(settings)
        else:
            run(settings)
    except (RuntimeError, ValidationError) as exc:
        logging.basicConfig(level=logging.ERROR, format=LOG_FORMAT)
        logging.getLogger(__name__).error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
