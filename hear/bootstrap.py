from __future__ import annotations

import importlib
import os
import shutil
import uuid
from pathlib import Path

import httpx

from hear.audio.io import AudioIO
from hear.config import RuntimeSettings
from hear.contracts.jobs import JobType, WorkerIdentity
from hear.execution.executor import JobExecutor
from hear.execution.native import NativeExecutor
from hear.execution.reporter import BackendAttemptClient
from hear.health.service import RuntimeReadiness
from hear.inference.manifest import ModelManifest
from hear.runtime.roles import WorkerRole
from hear.runtime.simulation import SimulationBoundary
from hear.storage.b2 import B2StorageFactory
from hear.tools.dependency_patches import DependencyPatchManager


class RuntimeBootstrap:
    def __init__(
        self,
        environment: dict[str, str] | None = None,
        *,
        root: Path | None = None,
    ) -> None:
        source = dict(os.environ) if environment is None else environment
        self._settings = RuntimeSettings.from_environment(source)
        self._generation = self._settings.worker_generation or str(uuid.uuid4())
        self._root = root or Path(__file__).resolve().parents[1]
        self._model_root = self._settings.model_root
        self._manifest = ModelManifest(self._root / "hear" / "model_manifest.json")
        self._manifest.validate_overrides(self._settings.model_paths)
        self._patch_manager = DependencyPatchManager(self._root)
        self._readiness: dict[WorkerRole, RuntimeReadiness] = {}

    def worker_identity(self) -> WorkerIdentity:
        return WorkerIdentity(
            worker_id=self._settings.required("worker_id"),
            generation=self._generation,
            image_revision=self._settings.required("image_revision"),
            engine_revision=self._settings.required("engine_revision"),
        )

    def _attempt_reporter(self, client: httpx.AsyncClient) -> BackendAttemptClient:
        return BackendAttemptClient(
            self.worker_identity(),
            self._settings.required("backend_internal_url"),
            client,
        )

    def readiness(self, role: WorkerRole) -> RuntimeReadiness:
        current = self._readiness.get(role)
        if current is not None:
            return current
        current = RuntimeReadiness(
            role,
            self._manifest,
            self._settings.fish_speech_model_root or self._model_root
            if role == WorkerRole.RECONSTRUCTION
            else self._model_root,
            self._patch_manager,
            enabled_features=self._settings.model_features,
            model_paths=self._settings.model_paths,
            require_manifest_models=role != WorkerRole.MAGIC_CLEAN_NATURAL,
            simulation=SimulationBoundary.enabled(),
        )
        scratch_root = self._settings.temp_dir
        required_scratch_bytes = self._settings.min_free_scratch_bytes
        if role in {
            WorkerRole.MAGIC_CLEAN_NATURAL,
        }:
            required_scratch_bytes = max(
                required_scratch_bytes,
                self._settings.magic_clean_scratch_bytes,
            )
        current.add_check(
            "scratch",
            lambda: self._scratch_has_capacity(scratch_root, required_scratch_bytes),
        )
        self._readiness[role] = current
        return current

    @staticmethod
    def _scratch_has_capacity(path: Path, required_bytes: int) -> bool:
        try:
            path.mkdir(parents=True, exist_ok=True)
            return shutil.disk_usage(path).free >= required_bytes
        except OSError:
            return False

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
        self.worker_identity()
        self._settings.required("backend_internal_url")
        if role == WorkerRole.PIPELINE:
            return self.pipeline_executor()
        if role == WorkerRole.TRANSCRIPTION:
            return self.transcription_executor()
        if role == WorkerRole.RECONSTRUCTION:
            return self.reconstruction_executor()
        if role == WorkerRole.MAGIC_CLEAN_NATURAL:
            return self.magic_clean_executor(role)
        raise RuntimeError(f"unsupported_runtime_role:{role.value}")

    def transcription_executor(self) -> tuple[JobExecutor, BackendAttemptClient, list[object]]:
        from hear.services.transcription.service import TranscriptionService
        from hear.workflows.transcription import TranscriptionWorkflow

        self.ensure_ready(WorkerRole.TRANSCRIPTION)
        qwen_module = importlib.import_module("hear.inference.qwen_asr")
        qwen_engine = qwen_module.LazyQwenAsrEngine
        scratch_root = self._settings.temp_dir
        client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(
                connect=15.0,
                read=self._settings.audio_download_read_timeout_seconds,
                write=30.0,
                pool=30.0,
            ),
        )
        native = NativeExecutor("transcription-runtime")
        engine = qwen_engine(
            model_path=self._model_path("qwen3-asr-1.7b"),
            aligner_path=self._model_path("qwen3-forced-aligner"),
            cache_dir=self._model_root,
            temp_dir=scratch_root,
            dtype=self._settings.qwen_asr_dtype,
            device_map=self._settings.qwen_asr_device_map,
            vad_onset=self._settings.whisper_vad_onset,
            vad_offset=self._settings.whisper_vad_offset,
            max_batch_size=self._settings.whisper_batch_size,
            long_audio_batch_size=self._settings.whisper_long_audio_batch_size,
            chunk_seconds=self._settings.whisper_chunk_seconds,
            idle_seconds=self._settings.pipeline_idle_ttl_seconds,
            eviction_enabled=self._settings.gpu_idle_eviction_enabled,
        )
        readiness = self.readiness(WorkerRole.TRANSCRIPTION)
        readiness.add_check("asr", lambda: self._engine_healthy(engine))
        service = TranscriptionService(
            engine,
            chunk_seconds=self._settings.whisper_chunk_seconds,
            batch_size=self._settings.whisper_batch_size,
            long_audio_batch_size=self._settings.whisper_long_audio_batch_size,
            native=native,
            min_avg_logprob=self._settings.whisper_min_avg_logprob,
        )
        audio = AudioIO(
            client,
            native,
            max_download_bytes=self._settings.audio_download_max_bytes,
            decode_timeout_seconds=self._settings.audio_decode_timeout_seconds,
        )
        workflow = TranscriptionWorkflow(
            service,
            audio,
            native,
            workspace_root=scratch_root,
        )
        backend = self._attempt_reporter(client)
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

        self._settings.required("backend_service_key")
        self.ensure_ready(WorkerRole.PIPELINE)
        scratch_root = self._settings.temp_dir
        client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(
                connect=15.0,
                read=self._settings.audio_download_read_timeout_seconds,
                write=30.0,
                pool=30.0,
            ),
        )
        model_native = NativeExecutor("pipeline-models")
        audio_native = NativeExecutor("pipeline-audio")
        catalog = PipelineCatalogClient(
            self._settings.required("backend_internal_url"),
            self._settings.required("backend_service_key"),
        ).fetch()
        if not catalog.categories:
            raise RuntimeError("pipeline_catalog_has_no_categories")
        qwen_module = importlib.import_module("hear.inference.qwen_asr")
        small_module = importlib.import_module("hear.inference.small_models")
        text_module = importlib.import_module("hear.inference.text_generation")
        asr = qwen_module.LazyQwenAsrEngine(
            model_path=self._model_path("qwen3-asr-1.7b"),
            aligner_path=self._model_path("qwen3-forced-aligner"),
            cache_dir=self._model_root,
            temp_dir=scratch_root,
            dtype=self._settings.qwen_asr_dtype,
            device_map=self._settings.qwen_asr_device_map,
            vad_onset=self._settings.whisper_vad_onset,
            vad_offset=self._settings.whisper_vad_offset,
            max_batch_size=self._settings.whisper_batch_size,
            long_audio_batch_size=self._settings.whisper_long_audio_batch_size,
            chunk_seconds=self._settings.whisper_chunk_seconds,
            idle_seconds=self._settings.pipeline_idle_ttl_seconds,
            eviction_enabled=self._settings.gpu_idle_eviction_enabled,
        )
        small_models = small_module.LazySmallModelsEngine(
            self._model_path("toxic-bert"),
            self._model_path("twitter-roberta-sentiment"),
            self._model_path("nli-distilroberta"),
            model_native,
            idle_seconds=self._settings.pipeline_idle_ttl_seconds,
            eviction_enabled=self._settings.gpu_idle_eviction_enabled,
        )
        features = self._settings.model_features
        if "qwen_llm" in features:
            text_generation = text_module.VllmTextGenerationEngine(
                self._model_path("qwen2.5-7b-instruct"),
                gpu_memory_utilization=self._settings.qwen_llm_gpu_memory_utilization,
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
            discovery_max_new_tokens=self._settings.discovery_max_new_tokens,
        )
        transcriber = TranscriptionService(
            asr,
            chunk_seconds=self._settings.whisper_chunk_seconds,
            batch_size=self._settings.whisper_batch_size,
            long_audio_batch_size=self._settings.whisper_long_audio_batch_size,
            native=audio_native,
            min_avg_logprob=self._settings.whisper_min_avg_logprob,
        )
        audio = AudioIO(
            client,
            audio_native,
            max_download_bytes=self._settings.audio_download_max_bytes,
            decode_timeout_seconds=self._settings.audio_decode_timeout_seconds,
        )
        transcription = TranscriptionWorkflow(
            transcriber,
            audio,
            audio_native,
            workspace_root=scratch_root,
        )
        pipeline = PipelineWorkflow(
            transcriber,
            ModerationService(model_client, llm, catalog.harm_keywords),
            CategorizationService(
                model_client,
                llm,
                catalog.category_catalog,
                catalog.taxonomy,
            ),
            DiscoveryService(
                llm,
                catalog.taxonomy,
                metadata_enabled=self._settings.discovery_metadata_enabled,
                max_search_phrases=self._settings.discovery_max_search_phrases,
            ),
            audio,
            audio_native,
            workspace_root=scratch_root,
        )
        backend = self._attempt_reporter(client)
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
        from hear.services.reconstruction.fish_renderer import FishReconstructionRenderer
        from hear.workflows.fish_reconstruction import FishReconstructionWorkflow

        self.ensure_ready(WorkerRole.RECONSTRUCTION)
        scratch_root = self._settings.temp_dir
        client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(
                connect=15.0,
                read=self._settings.audio_download_read_timeout_seconds,
                write=30.0,
                pool=30.0,
            ),
        )
        fish_native = NativeExecutor("reconstruction-fish")
        audio_native = NativeExecutor("reconstruction-audio")
        fish_module = importlib.import_module("hear.inference.fish_speech")
        fish_root = self._settings.fish_speech_model_root or self._model_root
        checkpoint = self._manifest.local_path(
            fish_root, "fish-speech-s2-pro", self._settings.model_paths
        )
        fish = fish_module.LazyFishSpeechEngine(
            self._settings.fish_speech_home,
            checkpoint,
            checkpoint / "codec.pth",
            fish_native,
            idle_seconds=self._settings.reconstruction_idle_ttl_seconds,
            eviction_enabled=self._settings.gpu_idle_eviction_enabled,
        )
        readiness = self.readiness(WorkerRole.RECONSTRUCTION)
        readiness.add_check("fish_speech", lambda: self._engine_healthy(fish))
        audio = AudioIO(
            client,
            audio_native,
            max_download_bytes=self._settings.audio_download_max_bytes,
            decode_timeout_seconds=self._settings.audio_decode_timeout_seconds,
        )
        workflow = FishReconstructionWorkflow(
            FishReconstructionRenderer(fish, audio_native),
            audio,
            B2StorageFactory(),
            audio_native,
            workspace_root=scratch_root,
        )
        return (
            JobExecutor({JobType.RECONSTRUCTION: workflow}),
            self._attempt_reporter(client),
            [fish, fish_native, audio_native, client],
        )

    def magic_clean_executor(
        self,
        role: WorkerRole,
    ) -> tuple[JobExecutor, BackendAttemptClient, list[object]]:
        from hear.runtime.cleaner.deepfilter_available import DeepFilterNetCleaner
        from hear.workflows.available_magic_clean import AvailableMagicCleanWorkflow

        if role != WorkerRole.MAGIC_CLEAN_NATURAL:
            raise RuntimeError("unsupported_magic_clean_role")
        sound_cleanup_service = None
        if self._settings.sound_cleanup_bundle is not None:
            from hear.services.sound_cleanup.analysis import SoundAnalyser
            from hear.services.sound_cleanup.assets import SoundCleanupAssets
            from hear.services.sound_cleanup.separator import EventSeparator
            from hear.services.sound_cleanup.service import SoundCleanupService

            assets = SoundCleanupAssets.load(
                self._settings.sound_cleanup_bundle,
                self._settings.sound_cleanup_bundle_sha256 or "",
            )
            separator = None
            if self._settings.sound_cleanup_separator_bundle is not None:
                separator = EventSeparator(
                    self._settings.sound_cleanup_separator_bundle,
                    self._settings.sound_cleanup_separator_sha256 or "",
                    self._settings.magic_clean_model_device,
                    idle_seconds=self._settings.audiosep_idle_ttl_seconds,
                    eviction_enabled=self._settings.gpu_idle_eviction_enabled,
                )
            sound_cleanup_service = SoundCleanupService(
                SoundAnalyser(assets, device=self._settings.magic_clean_model_device),
                separator=separator,
            )
        model_cleaner = DeepFilterNetCleaner(
            self._root / "deploy" / "cleaner" / "deepfilter3.ini",
            self._settings.magic_clean_model_directory,
            self._magic_clean_budget(),
            device=self._settings.magic_clean_model_device,
            sound_cleanup_service=sound_cleanup_service,
            idle_seconds=self._settings.magic_clean_idle_ttl_seconds,
            eviction_enabled=self._settings.gpu_idle_eviction_enabled,
        )
        readiness = self.readiness(role)
        readiness.add_check("ffmpeg", self._ffmpeg_ready)
        readiness.add_check("model_engine", model_cleaner.is_ready)
        readiness.initialize()
        if not readiness.is_ready():
            model_cleaner.close()
            raise RuntimeError("runtime_not_ready")
        client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(
                connect=15.0,
                read=self._settings.audio_download_read_timeout_seconds,
                write=30.0,
                pool=30.0,
            ),
        )
        native = NativeExecutor(f"{role.value}-available")
        audio = AudioIO(
            client,
            native,
            max_download_bytes=self._settings.audio_download_max_bytes,
            decode_timeout_seconds=self._settings.audio_decode_timeout_seconds,
        )
        workflow = AvailableMagicCleanWorkflow(
            audio,
            B2StorageFactory(),
            native,
            workspace_root=self._settings.temp_dir,
            timeout_seconds=self._settings.audio_decode_timeout_seconds,
            model_cleaner=model_cleaner,
        )
        return (
            JobExecutor({JobType.MAGIC_CLEAN: workflow}),
            self._attempt_reporter(client),
            [model_cleaner, native, client],
        )

    def _magic_clean_budget(self):
        from hear.runtime.cleaner.resource_guard import ResourceBudget

        return ResourceBudget(
            self._settings.magic_clean_scratch_bytes,
            self._settings.magic_clean_max_input_bytes,
            self._settings.magic_clean_max_frames,
        )

    def _model_path(self, logical_name: str) -> Path:
        return self._manifest.local_path(self._model_root, logical_name, self._settings.model_paths)

    @staticmethod
    def _ffmpeg_ready() -> bool:
        return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

    @staticmethod
    def _engine_healthy(engine) -> bool:
        check = getattr(engine, "check_health", None)
        if check is None:
            return True
        check()
        return True
