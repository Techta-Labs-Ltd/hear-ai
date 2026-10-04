from __future__ import annotations

import inspect
import io
import logging
import multiprocessing
import sys
import threading
import time
from pathlib import Path
from typing import Any

from hear.execution.native import NativeExecutor
from hear.runtime.gpu_idle import AsyncIdleResource

MAX_SEQUENCE_TOKENS = 4096


class FishProcess:
    @staticmethod
    def serve(connection, source_root: str, checkpoint: str, codec: str):
        phase = "imports"
        try:
            sys.path.insert(0, source_root)
            import numpy as np
            import soundfile as sf
            import torch
            from fish_speech.inference_engine import TTSInferenceEngine
            from fish_speech.models.dac.inference import load_model
            from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
            from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest
            from loguru import logger

            # Third-party INFO logs include complete job transcripts.
            logger.disable("fish_speech")

            if not torch.cuda.is_available():
                raise RuntimeError("fish_speech_cuda_unavailable")
            precision = torch.bfloat16
            kwargs = {
                "checkpoint_path": checkpoint,
                "device": "cuda",
                "precision": precision,
                "compile": False,
            }
            # Upstream sizes the KV cache from config.json (32k tokens, about 10 GB).
            # Jobs are bounded to 2,000 characters plus a 20 s reference, so a
            # 4,096-token cache is ample and keeps the engine near 10 GB.
            from fish_speech.models.text2semantic.llama import DualARTransformer

            original_from_pretrained = DualARTransformer.from_pretrained

            def capped_from_pretrained(*args, **kwargs):
                model = original_from_pretrained(*args, **kwargs)
                model.config.max_seq_len = min(model.config.max_seq_len, MAX_SEQUENCE_TOKENS)
                return model

            DualARTransformer.from_pretrained = staticmethod(capped_from_pretrained)
            supported = inspect.signature(launch_thread_safe_queue).parameters
            if "max_seq_len" in supported:
                kwargs["max_seq_len"] = MAX_SEQUENCE_TOKENS
            torch.set_num_threads(2)
            torch.cuda.reset_peak_memory_stats()
            phase = "semantic_model_loading"
            queue_result = launch_thread_safe_queue(**kwargs)
            llama_queue = queue_result[0] if isinstance(queue_result, tuple) else queue_result
            phase = "codec_loading"
            decoder = load_model(config_name="modded_dac_vq", checkpoint_path=codec, device="cuda")
            engine = TTSInferenceEngine(
                llama_queue=llama_queue,
                decoder_model=decoder,
                precision=precision,
                compile=False,
            )
            connection.send(
                {
                    "status": "ready",
                    "precision": "bfloat16",
                    "startup_cuda_allocated_bytes": torch.cuda.memory_allocated(),
                    "startup_cuda_reserved_bytes": torch.cuda.memory_reserved(),
                    "startup_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "startup_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
            )
            while True:
                phase = "waiting_for_request"
                values = connection.recv()
                if values is None:
                    llama_queue.put(None)
                    return
                torch.cuda.reset_peak_memory_stats()
                phase = "inference"
                request = ServeTTSRequest(
                    text=values["text"],
                    max_new_tokens=values["max_new_tokens"],
                    references=[ServeReferenceAudio(**ref) for ref in values["references"]],
                    reference_id=None,
                    seed=values["seed"],
                    top_p=0.7,
                    temperature=0.7,
                    format="wav",
                    streaming=False,
                    use_memory_cache="off",
                    normalize=False,
                )
                # Do not inherit a companion loader's bundled default speaker.
                # Only the job's explicit reference is allowed to condition voice.
                if not values["references"]:
                    request = request.model_copy(update={"reference_id": None})
                audio = None
                rate = 0
                for result in engine.inference(request):
                    if result.code == "error":
                        raise RuntimeError("fish_speech_inference_failed")
                    if result.code == "final":
                        rate, audio = result.audio
                if (
                    audio is None
                    or rate <= 0
                    or not 0 < audio.size <= rate * 120
                    or not np.isfinite(audio).all()
                ):
                    raise RuntimeError("fish_speech_invalid_output")
                stream = io.BytesIO()
                sf.write(stream, audio, rate, format="WAV", subtype="FLOAT")
                connection.send(
                    {
                        "status": "completed",
                        "audio": stream.getvalue(),
                        "precision": "bfloat16",
                        "sample_rate": rate,
                        "frames": int(audio.size),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    }
                )
        except BaseException as exc:
            logging.getLogger(__name__).error(
                "fish_speech_process_failed phase=%s error_type=%s", phase, type(exc).__name__
            )
            try:
                # Do not send user text, references, credential-bearing paths or tracebacks.
                connection.send({"status": "failed", "error_type": type(exc).__name__})
            except (OSError, EOFError):
                pass
        finally:
            connection.close()


class FishSpeechEngine:
    """One warm, cancellable Fish process per worker running the official bf16 weights."""

    def __init__(
        self,
        source_root: Path,
        checkpoint_path: Path,
        codec_path: Path,
        native: NativeExecutor,
        *,
        startup_timeout: float = 300,
        inference_timeout: float = 180,
    ):
        if not (source_root / "fish_speech" / "inference_engine").is_dir():
            raise RuntimeError("fish_speech_source_not_provisioned")
        if (
            not checkpoint_path.is_dir()
            or not codec_path.is_file()
            or not (checkpoint_path / "config.json").is_file()
        ):
            raise RuntimeError("fish_speech_checkpoint_not_provisioned")
        if min(startup_timeout, inference_timeout) <= 0:
            raise ValueError("invalid_fish_timeout")
        self._native = native
        self._timeout = inference_timeout
        self._faulted = False
        self._closed = False
        context = multiprocessing.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(
            target=FishProcess.serve,
            args=(child, str(source_root), str(checkpoint_path), str(codec_path)),
            daemon=True,
        )
        self._process.start()
        child.close()
        try:
            self.startup_metrics = self._receive(startup_timeout, threading.Event())
            self.last_inference_metrics: dict[str, Any] = {}
            if self.startup_metrics.get("status") != "ready":
                raise RuntimeError("fish_speech_startup_failed")
        except BaseException:
            self._stop()
            raise

    def _stop(self) -> None:
        self._faulted = True
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=2)
        self._connection.close()

    def _receive(self, timeout: float, cancelled: threading.Event) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            if cancelled.is_set():
                self._stop()
                raise RuntimeError("fish_speech_cancelled")
            if time.monotonic() >= deadline:
                self._stop()
                raise RuntimeError("fish_speech_timeout")
            if self._connection.poll(0.05):
                try:
                    response = self._connection.recv()
                except (EOFError, OSError):
                    logging.getLogger(__name__).error(
                        "fish_speech_process_exited exitcode=%s", self._process.exitcode
                    )
                    self._stop()
                    raise RuntimeError("fish_speech_process_exited") from None
                if not isinstance(response, dict) or response.get("status") == "failed":
                    self._stop()
                    raise RuntimeError("fish_speech_process_failed")
                return response
            if not self._process.is_alive():
                logging.getLogger(__name__).error(
                    "fish_speech_process_exited exitcode=%s", self._process.exitcode
                )
                self._stop()
                raise RuntimeError("fish_speech_process_exited")

    @staticmethod
    def validate_request(text, references, reference_id, language, max_new_tokens):
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            raise ValueError("invalid_fish_text")
        if reference_id:
            raise ValueError("global_voice_reference_ids_not_allowed_use_job_scoped_audio")
        if (
            language not in ("en", "en-GB")
            or type(max_new_tokens) is not int
            or not 1 <= max_new_tokens <= 2048
        ):
            raise ValueError("invalid_fish_generation_options")
        if len(references or []) > 1:
            raise ValueError("one_aligned_reference_per_request")
        for ref in references or []:
            if (
                not isinstance(ref, dict)
                or not isinstance(ref.get("audio"), bytes)
                or not 0 < len(ref["audio"]) <= 4_000_000
                or not isinstance(ref.get("text"), str)
                or not ref["text"].strip()
                or len(ref["text"]) > 4000
            ):
                raise ValueError("invalid_job_scoped_voice_reference")

    async def generate_speech(
        self,
        *,
        text: str,
        max_new_tokens: int = 1024,
        references: list[dict] | None = None,
        reference_id: str | None = None,
        language: str = "en",
        seed: int | None = None,
    ) -> bytes:
        self.validate_request(text, references, reference_id, language, max_new_tokens)
        cancelled = threading.Event()
        return await self._native.run_cancellable(
            self._generate,
            {
                "text": text,
                "max_new_tokens": max_new_tokens,
                "references": references or [],
                "seed": seed,
            },
            cancelled=cancelled,
        )

    def _generate(self, request: dict, *, cancelled: threading.Event) -> bytes:
        self.check_health()
        try:
            self._connection.send(request)
            response = self._receive(self._timeout, cancelled)
            self.last_inference_metrics = {
                key: value for key, value in response.items() if key != "audio"
            }
            data = response.get("audio")
            if (
                response.get("status") != "completed"
                or not isinstance(data, bytes)
                or not 44 < len(data) <= 25_000_000
            ):
                raise RuntimeError("fish_speech_invalid_output")
            return data
        except BaseException:
            self._stop()
            raise

    def check_health(self) -> None:
        if self._closed or self._faulted or not self._process.is_alive():
            raise RuntimeError("fish_speech_unavailable")

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._faulted and self._process.is_alive():
            try:
                self._connection.send(None)
                self._process.join(timeout=2)
            except (OSError, EOFError, BrokenPipeError):
                pass
        self._stop()


class LazyFishSpeechEngine:
    """Keep the reconstruction worker alive while Fish itself is cold when idle."""

    def __init__(
        self,
        source_root: Path,
        checkpoint_path: Path,
        codec_path: Path,
        native: NativeExecutor,
        *,
        startup_timeout: float = 300,
        inference_timeout: float = 180,
        idle_seconds: float,
        eviction_enabled: bool,
    ) -> None:
        self._source_root = source_root
        self._checkpoint_path = checkpoint_path
        self._codec_path = codec_path
        self._native = native
        self._startup_timeout = startup_timeout
        self._inference_timeout = inference_timeout
        self._loader = NativeExecutor("fish-speech-lazy-loader")
        self._resource = AsyncIdleResource(
            "fish_speech",
            self._load,
            self._close_loaded,
            idle_seconds=idle_seconds,
            eviction_enabled=eviction_enabled,
        )

    async def _load(self) -> FishSpeechEngine:
        return await self._loader.run(
            lambda: FishSpeechEngine(
                self._source_root,
                self._checkpoint_path,
                self._codec_path,
                self._native,
                startup_timeout=self._startup_timeout,
                inference_timeout=self._inference_timeout,
            )
        )

    @staticmethod
    async def _close_loaded(engine: FishSpeechEngine) -> None:
        await engine.close()

    async def warmup(self) -> None:
        await self._resource.acquire()
        await self._resource.release()

    async def generate_speech(
        self,
        *,
        text: str,
        max_new_tokens: int = 1024,
        references: list[dict] | None = None,
        reference_id: str | None = None,
        language: str = "en",
        seed: int | None = None,
    ) -> bytes:
        FishSpeechEngine.validate_request(text, references, reference_id, language, max_new_tokens)
        engine = await self._resource.acquire()
        try:
            return await engine.generate_speech(
                text=text,
                max_new_tokens=max_new_tokens,
                references=references,
                reference_id=reference_id,
                language=language,
                seed=seed,
            )
        finally:
            await self._resource.release()

    def check_health(self) -> None:
        if self._resource.state == "failed":
            raise RuntimeError("fish_speech_lazy_engine_failed")
        if not (self._source_root / "fish_speech" / "inference_engine").is_dir():
            raise RuntimeError("fish_speech_source_not_provisioned")
        if (
            not self._checkpoint_path.is_dir()
            or not self._codec_path.is_file()
            or not (self._checkpoint_path / "config.json").is_file()
        ):
            raise RuntimeError("fish_speech_checkpoint_not_provisioned")

    @property
    def lifecycle(self) -> dict:
        return self._resource.snapshot

    async def evict_now(self) -> bool:
        return await self._resource.evict_now()

    async def close(self) -> None:
        try:
            await self._resource.close()
        finally:
            await self._loader.close()
