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
            self._model_root,
            self._patch_manager,
            enabled_features=self._settings.model_features,
            require_manifest_models=not self._uses_available_engine(role),
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
        if role in {
            WorkerRole.MAGIC_CLEAN_NATURAL,
        }:
            return self.magic_clean_executor(role)
        raise RuntimeError(f"unsupported_runtime_role:{role.value}")

    def transcription_executor(self) -> tuple[JobExecutor, BackendAttemptClient, list[object]]:
        from hear.services.transcription.service import TranscriptionService
        from hear.workflows.transcription import TranscriptionWorkflow

        self.ensure_ready(WorkerRole.TRANSCRIPTION)
        qwen_module = importlib.import_module("hear.inference.qwen_asr")
        qwen_engine = qwen_module.QwenAsrEngine
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
            model_path=self._model_root / "qwen3-asr-1.7b",
            aligner_path=self._model_root / "qwen3-forced-aligner",
            cache_dir=self._model_root,
            temp_dir=scratch_root,
            dtype=self._settings.qwen_asr_dtype,
            device_map=self._settings.qwen_asr_device_map,
            vad_onset=self._settings.whisper_vad_onset,
            vad_offset=self._settings.whisper_vad_offset,
            max_batch_size=self._settings.whisper_batch_size,
            long_audio_batch_size=self._settings.whisper_long_audio_batch_size,
            chunk_seconds=self._settings.whisper_chunk_seconds,
        )
        readiness = self.readiness(WorkerRole.TRANSCRIPTION)
        readiness.add_check("asr", lambda: self._engine_healthy(engine))
        service = TranscriptionService(
            engine,
            chunk_seconds=self._settings.whisper_chunk_seconds,
            batch_size=self._settings.whisper_batch_size,
            long_audio_batch_size=self._settings.whisper_long_audio_batch_size,
            native=native,
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
            B2StorageFactory(),
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
        qwen_module = importlib.import_module("hear.inference.qwen_asr")
        small_module = importlib.import_module("hear.inference.small_models")
        text_module = importlib.import_module("hear.inference.text_generation")
        asr = qwen_module.QwenAsrEngine(
            model_path=self._model_root / "qwen3-asr-1.7b",
            aligner_path=self._model_root / "qwen3-forced-aligner",
            cache_dir=self._model_root,
            temp_dir=scratch_root,
            dtype=self._settings.qwen_asr_dtype,
            device_map=self._settings.qwen_asr_device_map,
            vad_onset=self._settings.whisper_vad_onset,
            vad_offset=self._settings.whisper_vad_offset,
            max_batch_size=self._settings.whisper_batch_size,
            long_audio_batch_size=self._settings.whisper_long_audio_batch_size,
            chunk_seconds=self._settings.whisper_chunk_seconds,
        )
        small_models = small_module.SmallModelsEngine(
            self._model_root / "toxic-bert",
            self._model_root / "twitter-roberta-sentiment",
            self._model_root / "nli-distilroberta",
            model_native,
        )
        features = self._settings.model_features
        if "qwen_llm" in features:
            text_generation = text_module.VllmTextGenerationEngine(
                self._model_root / "qwen2.5-7b-instruct",
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
        catalog = PipelineCatalogClient(
            self._settings.required("backend_internal_url"),
            self._settings.required("backend_service_key"),
        ).fetch()
        if not catalog.categories:
            raise RuntimeError("pipeline_catalog_has_no_categories")
        transcriber = TranscriptionService(
            asr,
            chunk_seconds=self._settings.whisper_chunk_seconds,
            batch_size=self._settings.whisper_batch_size,
            long_audio_batch_size=self._settings.whisper_long_audio_batch_size,
            native=audio_native,
        )
        audio = AudioIO(
            client,
            audio_native,
            max_download_bytes=self._settings.audio_download_max_bytes,
            decode_timeout_seconds=self._settings.audio_decode_timeout_seconds,
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
            ModerationService(model_client, llm, catalog.harm_keywords),
            CategorizationService(
                model_client,
                llm,
                catalog.category_catalog,
                catalog.taxonomy,
            ),
            DiscoveryService(llm, catalog.taxonomy),
            audio,
            storage_factory,
            audio_native,
            workspace_root=scratch_root,
            bitrate_kbps=self._settings.pipeline_mp3_bitrate_kbps,
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
        if self._uses_available_engine(WorkerRole.RECONSTRUCTION):
            return self.available_reconstruction_executor()
        from hear.inference.client import LocalInferenceClient
        from hear.services.reconstruction.dnsmos import DNSMOSScorer
        from hear.services.reconstruction.synthesizer import SpeechSynthesizer
        from hear.workflows.reconstruction import ReconstructionWorkflow

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
        fish = fish_module.FishSpeechEngine(
            self._settings.fish_speech_home,
            self._model_root / "fish-speech" / "s2-pro",
            self._model_root / "fish-speech" / "s2-pro" / "codec.pth",
            fish_native,
            bnb_mode=self._settings.fish_speech_bnb_mode,
        )
        readiness = self.readiness(WorkerRole.RECONSTRUCTION)
        readiness.add_check("fish_speech", lambda: self._engine_healthy(fish))
        model_client = LocalInferenceClient(speech_generation=fish)
        dnsmos = DNSMOSScorer(self._model_root / "dnsmos" / "sig_bak_ovr.onnx")
        synthesizer = SpeechSynthesizer(
            model_client,
            dnsmos_scorer=dnsmos,
        )
        synthesizer.load()
        readiness.add_check("dnsmos", dnsmos.load)
        audio = AudioIO(
            client,
            audio_native,
            max_download_bytes=self._settings.audio_download_max_bytes,
            decode_timeout_seconds=self._settings.audio_decode_timeout_seconds,
        )
        workflow = ReconstructionWorkflow(
            synthesizer,
            audio,
            B2StorageFactory(),
            workspace_root=scratch_root,
        )
        backend = self._attempt_reporter(client)
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
        if self._uses_available_engine(role):
            return self.available_magic_clean_executor(role)
        from hear.inference.magic_clean import MagicCleanRuntimeFactory
        from hear.runtime.roles import WorkerCapabilityRegistry
        from hear.workflows.magic_clean import MagicCleanWorkflow

        if role not in {
            WorkerRole.MAGIC_CLEAN_NATURAL,
        }:
            raise RuntimeError("unsupported_magic_clean_role")
        self._settings.required("cleaner_certification_path")
        scratch_root = self._settings.temp_dir
        client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(connect=15.0, read=60.0, write=30.0, pool=30.0),
        )
        native = NativeExecutor(f"{role.value}-runtime")
        factory = MagicCleanRuntimeFactory(
            Path(self._settings.required("cleaner_certification_path")),
            self._settings.cleaner_lock_dir,
            self._settings.required("cleaner_certification_sha256"),
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
        backend = self._attempt_reporter(client)
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

    def available_reconstruction_executor(
        self,
    ) -> tuple[JobExecutor, BackendAttemptClient, list[object]]:
        from hear.workflows.available_reconstruction import AvailableReconstructionWorkflow

        role = WorkerRole.RECONSTRUCTION
        readiness = self.readiness(role)
        readiness.add_check("ffmpeg", self._ffmpeg_ready)
        readiness.initialize()
        if not readiness.is_ready():
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
        native = NativeExecutor("available-reconstruction")
        audio = AudioIO(
            client,
            native,
            max_download_bytes=self._settings.audio_download_max_bytes,
            decode_timeout_seconds=self._settings.audio_decode_timeout_seconds,
        )
        workflow = AvailableReconstructionWorkflow(
            audio,
            B2StorageFactory(),
            native,
            workspace_root=self._settings.temp_dir,
            timeout_seconds=self._settings.audio_decode_timeout_seconds,
        )
        return (
            JobExecutor({JobType.RECONSTRUCTION: workflow}),
            self._attempt_reporter(client),
            [native, client],
        )

    def available_magic_clean_executor(
        self,
        role: WorkerRole,
    ) -> tuple[JobExecutor, BackendAttemptClient, list[object]]:
        from hear.workflows.available_magic_clean import AvailableMagicCleanWorkflow

        model_cleaner = None
        if role == WorkerRole.MAGIC_CLEAN_NATURAL:
            from hear.runtime.cleaner.deepfilter_available import DeepFilterNetCleaner

            sound_cleanup_service = None
            if self._settings.sound_cleanup_bundle is not None:
                from hear.services.sound_cleanup.analysis import SoundAnalyser
                from hear.services.sound_cleanup.assets import SoundCleanupAssets
                from hear.services.sound_cleanup.service import SoundCleanupService

                assets = SoundCleanupAssets.load(
                    self._settings.sound_cleanup_bundle,
                    self._settings.sound_cleanup_bundle_sha256 or "",
                )
                from hear.services.sound_cleanup.separator import EventSeparator

                separator = None
                if self._settings.sound_cleanup_separator_bundle is not None:
                    separator = EventSeparator(
                        self._settings.sound_cleanup_separator_bundle,
                        self._settings.sound_cleanup_separator_sha256 or "",
                        self._settings.magic_clean_model_device,
                    )
                sound_cleanup_service = SoundCleanupService(
                    SoundAnalyser(
                        assets,
                        device=self._settings.magic_clean_model_device,
                    ),
                    separator=separator,
                )
            model_cleaner = DeepFilterNetCleaner(
                self._root / "deploy" / "cleaner" / "deepfilter3.ini",
                self._model_root / "magic-clean" / "DeepFilterNet3",
                self._magic_clean_budget(),
                device=self._settings.magic_clean_model_device,
                sound_cleanup_service=sound_cleanup_service,
            )
        readiness = self.readiness(role)
        readiness.add_check("ffmpeg", self._ffmpeg_ready)
        if model_cleaner is not None:
            readiness.add_check("model_engine", model_cleaner.is_ready)
        readiness.initialize()
        if not readiness.is_ready():
            if model_cleaner is not None:
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
        resources = [native, client]
        if model_cleaner is not None:
            resources.insert(0, model_cleaner)
        return (
            JobExecutor({JobType.MAGIC_CLEAN: workflow}),
            self._attempt_reporter(client),
            resources,
        )

    def _magic_clean_budget(self):
        from hear.runtime.cleaner.resource_guard import ResourceBudget

        return ResourceBudget(
            self._settings.magic_clean_scratch_bytes,
            self._settings.magic_clean_max_input_bytes,
            self._settings.magic_clean_max_frames,
        )

    def _uses_available_engine(self, role: WorkerRole) -> bool:
        return self._settings.optional_engine_mode == "available" and role in {
            WorkerRole.RECONSTRUCTION,
            WorkerRole.MAGIC_CLEAN_NATURAL,
        }

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
