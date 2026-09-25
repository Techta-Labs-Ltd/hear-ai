from __future__ import annotations

import importlib
import os
import uuid
from pathlib import Path

import httpx

from hear.audio.io import AudioIO
from hear.contracts.jobs import JobType, WorkerIdentity
from hear.execution.executor import JobExecutor
from hear.execution.native import NativeExecutor
from hear.execution.reporter import BackendAttemptClient
from hear.health.service import RuntimeReadiness
from hear.inference.manifest import ModelManifest
from hear.runtime.roles import WorkerRole
from hear.services.transcription.service import TranscriptionService
from hear.storage.b2 import B2StorageFactory
from hear.tools.dependency_patches import DependencyPatchManager
from hear.workflows.transcription import TranscriptionWorkflow


class RuntimeBootstrap:
    def __init__(
        self,
        environment: dict[str, str] | None = None,
        *,
        root: Path | None = None,
    ) -> None:
        self._environment = environment or dict(os.environ)
        self._root = root or Path(__file__).resolve().parents[1]
        self._model_root = Path(self._environment.get("HEAR_MODEL_ROOT", "/models"))
        self._manifest = ModelManifest(self._root / "hear" / "model_manifest.json")
        self._patch_manager = DependencyPatchManager(self._root)
        self._readiness: dict[WorkerRole, RuntimeReadiness] = {}

    def worker_identity(self) -> WorkerIdentity:
        return WorkerIdentity(
            worker_id=self._required("HEAR_WORKER_ID"),
            generation=self._environment.get("HEAR_WORKER_GENERATION") or str(uuid.uuid4()),
            image_revision=self._required("HEAR_IMAGE_REVISION"),
            engine_revision=self._required("HEAR_ENGINE_REVISION"),
        )

    def readiness(self, role: WorkerRole) -> RuntimeReadiness:
        current = self._readiness.get(role)
        if current is not None:
            return current
        current = RuntimeReadiness(
            role,
            self._manifest,
            self._model_root,
            self._patch_manager,
            enabled_features=self._enabled_features(),
        )
        self._readiness[role] = current
        return current

    def ensure_ready(self, role: WorkerRole) -> RuntimeReadiness:
        readiness = self.readiness(role)
        readiness.initialize()
        if not readiness.is_ready():
            raise RuntimeError("runtime_not_ready")
        return readiness

    def transcription_executor(self) -> tuple[JobExecutor, BackendAttemptClient, list[object]]:
        self.ensure_ready(WorkerRole.TRANSCRIPTION)
        qwen_module = importlib.import_module("hear.inference.qwen_asr")
        qwen_engine = qwen_module.QwenAsrEngine
        scratch_root = Path(self._environment.get("HEAR_TEMP_DIR", "/audio"))
        client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(
                connect=15.0,
                read=float(self._environment.get("AUDIO_DOWNLOAD_READ_TIMEOUT_SECONDS", "60")),
                write=30.0,
                pool=30.0,
            ),
        )
        native = NativeExecutor("transcription-runtime")
        engine = qwen_engine(
            model_path=self._model_root / "qwen3-asr-1.7b",
            aligner_path=self._model_root / "qwen3-forced-aligner",
            cache_dir=self._model_root,
            temp_dir=scratch_root,
            dtype=self._environment.get("QWEN_ASR_DTYPE", "bfloat16"),
            device_map=self._environment.get("QWEN_ASR_DEVICE_MAP", "cuda:0"),
            vad_onset=float(self._environment.get("WHISPER_VAD_ONSET", "0.65")),
            vad_offset=float(self._environment.get("WHISPER_VAD_OFFSET", "0.50")),
            max_batch_size=int(self._environment.get("WHISPER_BATCH_SIZE", "36")),
            long_audio_batch_size=int(
                self._environment.get("WHISPER_LONG_AUDIO_BATCH_SIZE", "4")
            ),
            chunk_seconds=int(self._environment.get("WHISPER_CHUNK_SECONDS", "600")),
        )
        service = TranscriptionService(
            engine,
            chunk_seconds=int(self._environment.get("WHISPER_CHUNK_SECONDS", "600")),
            batch_size=int(self._environment.get("WHISPER_BATCH_SIZE", "36")),
            long_audio_batch_size=int(
                self._environment.get("WHISPER_LONG_AUDIO_BATCH_SIZE", "4")
            ),
        )
        audio = AudioIO(
            client,
            native,
            max_download_bytes=int(
                self._environment.get("AUDIO_DOWNLOAD_MAX_BYTES", str(4 * 1024**3))
            ),
            decode_timeout_seconds=float(
                self._environment.get("AUDIO_DECODE_TIMEOUT_SECONDS", "1200")
            ),
        )
        workflow = TranscriptionWorkflow(
            service,
            audio,
            B2StorageFactory(),
            native,
            workspace_root=scratch_root,
        )
        backend = BackendAttemptClient(self.worker_identity(), client)
        return JobExecutor({JobType.TRANSCRIPTION: workflow}), backend, [
            engine,
            native,
            client,
        ]

    def _enabled_features(self) -> frozenset[str]:
        raw = self._environment.get("HEAR_MODEL_FEATURES", "")
        return frozenset(item.strip() for item in raw.split(",") if item.strip())

    def _required(self, name: str) -> str:
        value = self._environment.get(name, "").strip()
        if not value:
            raise RuntimeError(f"missing_runtime_setting:{name}")
        return value