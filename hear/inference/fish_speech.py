from __future__ import annotations

import inspect
import io
import multiprocessing
import sys
import threading
import time
from pathlib import Path
from typing import Any

from hear.execution.native import NativeExecutor


class FishProcess:
    @staticmethod
    def serve(connection, source_root: str, checkpoint: str, codec: str, bnb_mode: str):
        try:
            sys.path.insert(0, source_root)
            import numpy as np
            import soundfile as sf
            import torch
            from fish_speech.inference_engine import TTSInferenceEngine
            from fish_speech.models.dac.inference import load_model
            from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
            from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

            if not torch.cuda.is_available():
                raise RuntimeError("fish_speech_cuda_unavailable")
            kwargs = {
                "checkpoint_path": checkpoint,
                "device": "cuda",
                "precision": torch.bfloat16,
                "compile": False,
            }
            supported = inspect.signature(launch_thread_safe_queue).parameters
            if bnb_mode not in ("", "none") and "bnb_mode" not in supported:
                raise RuntimeError("requested_fish_quantisation_not_supported_by_pinned_source")
            if "bnb_mode" in supported:
                kwargs["bnb_mode"] = bnb_mode if bnb_mode not in ("", "none") else None
            queue_result = launch_thread_safe_queue(**kwargs)
            llama_queue = queue_result[0] if isinstance(queue_result, tuple) else queue_result
            decoder = load_model(config_name="modded_dac_vq", checkpoint_path=codec, device="cuda")
            engine = TTSInferenceEngine(
                llama_queue=llama_queue,
                decoder_model=decoder,
                precision=torch.bfloat16,
                compile=False,
            )
            connection.send({"status": "ready"})
            while True:
                values = connection.recv()
                if values is None:
                    llama_queue.put(None)
                    return
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
                )
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
                connection.send({"status": "completed", "audio": stream.getvalue()})
        except BaseException as exc:
            try:
                # Do not send user text, references, credential-bearing paths or tracebacks.
                connection.send({"status": "failed", "error_type": type(exc).__name__})
            except (OSError, EOFError):
                pass
        finally:
            connection.close()


class FishSpeechEngine:
    """One warm, cancellable Fish process per worker; no FFmpeg-only fallback."""

    def __init__(
        self,
        source_root: Path,
        checkpoint_path: Path,
        codec_path: Path,
        native: NativeExecutor,
        *,
        bnb_mode: str = "none",
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
            args=(child, str(source_root), str(checkpoint_path), str(codec_path), bnb_mode),
            daemon=True,
        )
        self._process.start()
        child.close()
        try:
            if self._receive(startup_timeout, threading.Event()).get("status") != "ready":
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
                response = self._connection.recv()
                if not isinstance(response, dict) or response.get("status") == "failed":
                    self._stop()
                    raise RuntimeError("fish_speech_process_failed")
                return response
            if not self._process.is_alive():
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
        self._stop()
