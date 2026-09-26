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
from hear.storage.b2 import B2StorageFactory
from hear.tools.dependency_patches import DependencyPatchManager


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

    def executor_for(
        self,
        role: WorkerRole,
    ) -> tuple[JobExecutor, BackendAttemptClient, list[object]]:
        if role == WorkerRole.PIPELINE:
            return self.pipeline_executor()
        if role == WorkerRole.TRANSCRIPTION:
            return self.transcription_executor()
        if role == WorkerRole.RECONSTRUCTION:
            return self.reconstruction_executor()
        if role in {
            WorkerRole.MAGIC_CLEAN_NATURAL,
            WorkerRole.MAGIC_CLEAN_VOICE_FOCUS,
            WorkerRole.MAGIC_CLEAN_MUSIC_ATMOSPHERE,
        }:
            return self.magic_clean_executor(role)
        raise RuntimeError(f"unsupported_runtime_role:{role.value}")

    def transcription_executor(self) -> tuple[JobExecutor, BackendAttemptClient, list[object]]:
        from hear.services.transcription.service import TranscriptionService
        from hear.workflows.transcription import TranscriptionWorkflow

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
            long_audio_batch_size=int(self._environment.get("WHISPER_LONG_AUDIO_BATCH_SIZE", "4")),
            chunk_seconds=int(self._environment.get("WHISPER_CHUNK_SECONDS", "600")),
        )
        readiness = self.readiness(WorkerRole.TRANSCRIPTION)
        readiness.add_check("asr", lambda: self._engine_healthy(engine))
        service = TranscriptionService(
            engine,
            chunk_seconds=int(self._environment.get("WHISPER_CHUNK_SECONDS", "600")),
            batch_size=int(self._environment.get("WHISPER_BATCH_SIZE", "36")),
            long_audio_batch_size=int(self._environment.get("WHISPER_LONG_AUDIO_BATCH_SIZE", "4")),
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
        return (
            JobExecutor({JobType.TRANSCRIPTION: workflow}),
            backend,
            [
                engine,
                native,
                client,
            ],
        )

    def pipeline_executor(self) -> tuple[JobExecutor, BackendAttemptClient, list[object]]:
        from hear.inference.client import LocalInferenceClient
        from hear.services.categorization.discovery import DiscoveryService
        from hear.services.categorization.service import CategorizationService
        from hear.services.llm import LLMService
        from hear.services.moderation.service import ModerationService
        from hear.services.pipeline.catalog import PipelineCatalogClient
        from hear.services.transcription.service import TranscriptionService
        from hear.workflows.pipeline import PipelineWorkflow
        from hear.workflows.transcription import TranscriptionWorkflow

        self.ensure_ready(WorkerRole.PIPELINE)
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
        model_native = NativeExecutor("pipeline-models")
        audio_native = NativeExecutor("pipeline-audio")
        qwen_module = importlib.import_module("hear.inference.qwen_asr")
        small_module = importlib.import_module("hear.inference.small_models")
        text_module = importlib.import_module("hear.inference.text_generation")
        asr = qwen_module.QwenAsrEngine(
            model_path=self._model_root / "qwen3-asr-1.7b",
            aligner_path=self._model_root / "qwen3-forced-aligner",
            cache_dir=self._model_root,
            temp_dir=scratch_root,
            dtype=self._environment.get("QWEN_ASR_DTYPE", "bfloat16"),
            device_map=self._environment.get("QWEN_ASR_DEVICE_MAP", "cuda:0"),
            vad_onset=float(self._environment.get("WHISPER_VAD_ONSET", "0.65")),
            vad_offset=float(self._environment.get("WHISPER_VAD_OFFSET", "0.50")),
            max_batch_size=int(self._environment.get("WHISPER_BATCH_SIZE", "36")),
            long_audio_batch_size=int(self._environment.get("WHISPER_LONG_AUDIO_BATCH_SIZE", "4")),
            chunk_seconds=int(self._environment.get("WHISPER_CHUNK_SECONDS", "600")),
        )
        small_models = small_module.SmallModelsEngine(
            self._model_root / "toxic-bert",
            self._model_root / "twitter-roberta-sentiment",
            self._model_root / "nli-distilroberta",
            model_native,
        )
        features = self._enabled_features()
        if "qwen_llm" in features:
            text_generation = text_module.VllmTextGenerationEngine(
                self._model_root / "qwen2.5-7b-instruct",
                gpu_memory_utilization=float(
                    self._environment.get("QWEN_LLM_GPU_MEMORY_UTILIZATION", "0.75")
                ),
            )
        else:
            text_generation = text_module.DisabledTextGenerationEngine()
        readiness = self.readiness(WorkerRole.PIPELINE)
        readiness.add_check("asr", lambda: self._engine_healthy(asr))
        readiness.add_check("small_models", lambda: self._engine_healthy(small_models))
        model_client = LocalInferenceClient(
            small_models=small_models,
            text_generation=text_generation,
        )
        llm = LLMService(
            model_client,
            enabled=text_generation.is_available,
            discovery_max_new_tokens=int(self._environment.get("DISCOVERY_MAX_NEW_TOKENS", "1100")),
        )
        catalog = PipelineCatalogClient(
            self._required("HEAR_BACKEND_INTERNAL_URL"),
            self._required("HEAR_BACKEND_SERVICE_KEY"),
        ).fetch()
        if not catalog.categories:
            raise RuntimeError("pipeline_catalog_has_no_categories")
        transcriber = TranscriptionService(
            asr,
            chunk_seconds=int(self._environment.get("WHISPER_CHUNK_SECONDS", "600")),
            batch_size=int(self._environment.get("WHISPER_BATCH_SIZE", "36")),
            long_audio_batch_size=int(self._environment.get("WHISPER_LONG_AUDIO_BATCH_SIZE", "4")),
        )
        audio = AudioIO(
            client,
            audio_native,
            max_download_bytes=int(
                self._environment.get("AUDIO_DOWNLOAD_MAX_BYTES", str(4 * 1024**3))
            ),
            decode_timeout_seconds=float(
                self._environment.get("AUDIO_DECODE_TIMEOUT_SECONDS", "1200")
            ),
        )
        storage_factory = B2StorageFactory()
        transcription = TranscriptionWorkflow(
            transcriber,
            audio,
            storage_factory,
            audio_native,
            workspace_root=scratch_root,
        )
        pipeline = PipelineWorkflow(
            transcriber,
            ModerationService(model_client, llm),
            CategorizationService(model_client, llm),
            DiscoveryService(llm),
            audio,
            storage_factory,
            audio_native,
            workspace_root=scratch_root,
            bitrate_kbps=int(self._environment.get("PIPELINE_MP3_BITRATE_KBPS", "96")),
        )
        backend = BackendAttemptClient(self.worker_identity(), client)
        return (
            JobExecutor(
                {
                    JobType.PIPELINE: pipeline,
                    JobType.TRANSCRIPTION: transcription,
                }
            ),
            backend,
            [
                asr,
                small_models,
                model_native,
                audio_native,
                client,
            ],
        )

    def reconstruction_executor(self) -> tuple[JobExecutor, BackendAttemptClient, list[object]]:
        from hear.inference.client import LocalInferenceClient
        from hear.services.reconstruction.synthesizer import SpeechSynthesizer
        from hear.workflows.reconstruction import ReconstructionWorkflow

        self.ensure_ready(WorkerRole.RECONSTRUCTION)
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
        fish_native = NativeExecutor("reconstruction-fish")
        audio_native = NativeExecutor("reconstruction-audio")
        fish_module = importlib.import_module("hear.inference.fish_speech")
        fish = fish_module.FishSpeechEngine(
            Path(self._environment.get("FISH_SPEECH_HOME", "/fish-speech")),
            self._model_root / "fish-speech" / "s2-pro",
            self._model_root / "fish-speech" / "s2-pro" / "codec.pth",
            fish_native,
            bnb_mode=self._environment.get("FISH_SPEECH_BNB_MODE", "nf4"),
        )
        readiness = self.readiness(WorkerRole.RECONSTRUCTION)
        readiness.add_check("fish_speech", lambda: self._engine_healthy(fish))
        model_client = LocalInferenceClient(speech_generation=fish)
        synthesizer = SpeechSynthesizer(model_client)
        synthesizer.load()
        audio = AudioIO(
            client,
            audio_native,
            max_download_bytes=int(
                self._environment.get("AUDIO_DOWNLOAD_MAX_BYTES", str(4 * 1024**3))
            ),
            decode_timeout_seconds=float(
                self._environment.get("AUDIO_DECODE_TIMEOUT_SECONDS", "1200")
            ),
        )
        workflow = ReconstructionWorkflow(
            synthesizer,
            audio,
            B2StorageFactory(),
            workspace_root=scratch_root,
        )
        backend = BackendAttemptClient(self.worker_identity(), client)
        return (
            JobExecutor({JobType.RECONSTRUCTION: workflow}),
            backend,
            [
                fish,
                fish_native,
                audio_native,
                client,
            ],
        )

    def magic_clean_executor(
        self, role: WorkerRole
    ) -> tuple[JobExecutor, BackendAttemptClient, list[object]]:
        from hear.inference.magic_clean import MagicCleanRuntimeFactory
        from hear.runtime.roles import WorkerCapabilityRegistry
        from hear.workflows.magic_clean import MagicCleanWorkflow

        if role not in {
            WorkerRole.MAGIC_CLEAN_NATURAL,
            WorkerRole.MAGIC_CLEAN_VOICE_FOCUS,
            WorkerRole.MAGIC_CLEAN_MUSIC_ATMOSPHERE,
        }:
            raise RuntimeError("unsupported_magic_clean_role")
        scratch_root = Path(self._environment.get("HEAR_TEMP_DIR", "/audio"))
        client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(connect=15.0, read=60.0, write=30.0, pool=30.0),
        )
        native = NativeExecutor(f"{role.value}-runtime")
        factory = MagicCleanRuntimeFactory(
            Path(self._required("HEAR_CLEANER_CERTIFICATION_PATH")),
            Path(self._environment.get("HEAR_CLEANER_LOCK_DIR", "/tmp/hear-cleaner-locks")),
        )
        worker = factory.build(role)
        profile = WorkerCapabilityRegistry().get(role).magic_clean_profile
        readiness = self.readiness(role)

        def cleaner_ready() -> bool:
            if profile is None:
                return False
            snapshot = worker.capabilities()
            return any(
                item.get("profile") == profile.value and item.get("ready") is True
                for item in snapshot.get("profiles", [])
            )

        readiness.add_check("cleaner", cleaner_ready)
        readiness.initialize()
        if not readiness.is_ready():
            worker.close()
            factory.close()
            raise RuntimeError("runtime_not_ready")
        workflow = MagicCleanWorkflow(
            worker,
            native,
            workspace_root=scratch_root,
            resource_budget=self._magic_clean_budget(),
        )
        backend = BackendAttemptClient(self.worker_identity(), client)
        return (
            JobExecutor({JobType.MAGIC_CLEAN: workflow}),
            backend,
            [
                factory,
                worker,
                native,
                client,
            ],
        )

    def _magic_clean_budget(self):
        from hear.runtime.cleaner.resource_guard import ResourceBudget

        return ResourceBudget(
            int(self._environment.get("MAGIC_CLEAN_SCRATCH_BYTES", str(8 * 1024**3))),
            int(self._environment.get("MAGIC_CLEAN_MAX_INPUT_BYTES", str(4 * 1024**3))),
            int(self._environment.get("MAGIC_CLEAN_MAX_FRAMES", str(96000 * 7200))),
            gpu_limit_bytes=int(
                self._environment.get("MAGIC_CLEAN_GPU_LIMIT_BYTES", "12000000000")
            ),
            gpu_target_bytes=int(
                self._environment.get("MAGIC_CLEAN_GPU_TARGET_BYTES", "10000000000")
            ),
            allocator_cap_bytes=int(
                self._environment.get("MAGIC_CLEAN_GPU_ALLOCATOR_CAP_BYTES", "9000000000")
            ),
        )

    @staticmethod
    def _engine_healthy(engine) -> bool:
        check = getattr(engine, "check_health", None)
        if check is None:
            return True
        check()
        return True

    def _enabled_features(self) -> frozenset[str]:
        raw = self._environment.get("HEAR_MODEL_FEATURES", "")
        return frozenset(item.strip() for item in raw.split(",") if item.strip())

    def _required(self, name: str) -> str:
        value = self._environment.get(name, "").strip()
        if not value:
            raise RuntimeError(f"missing_runtime_setting:{name}")
        return value
