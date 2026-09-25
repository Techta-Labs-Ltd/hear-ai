from __future__ import annotations

import gc
import inspect
import io
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from hear.execution.native import NativeExecutor


class FishSpeechEngine:
    def __init__(
        self,
        source_root: Path,
        checkpoint_path: Path,
        codec_path: Path,
        native: NativeExecutor,
        *,
        bnb_mode: str = "nf4",
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("fish_speech_cuda_unavailable")
        from fish_speech.inference_engine import TTSInferenceEngine
        from fish_speech.models.dac.inference import load_model as load_decoder_model
        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue

        self._native = native
        self._source_root = source_root
        queue_kwargs = {
            "checkpoint_path": str(checkpoint_path),
            "device": "cuda",
            "precision": torch.bfloat16,
            "compile": False,
        }
        parameters = inspect.signature(launch_thread_safe_queue).parameters
        if "bnb_mode" in parameters:
            queue_kwargs["bnb_mode"] = bnb_mode or None
        if "lazy_load" in parameters:
            queue_kwargs["lazy_load"] = False
        queue_result = launch_thread_safe_queue(**queue_kwargs)
        if isinstance(queue_result, tuple):
            llama_queue, self._llama_thread = queue_result
        else:
            llama_queue = queue_result
            self._llama_thread = None
        decoder = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=str(codec_path),
            device="cuda",
        )
        self._engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder,
            precision=torch.bfloat16,
            compile=False,
        )

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
        return await self._native.run(
            self._generate_speech,
            text,
            max_new_tokens,
            references,
            reference_id,
            language,
            seed,
        )

    def _generate_speech(
        self,
        text: str,
        max_new_tokens: int,
        references: list[dict] | None,
        reference_id: str | None,
        language: str,
        seed: int | None,
    ) -> bytes:
        from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

        refs = [
            ServeReferenceAudio(audio=item.get("audio", b""), text=item.get("text", ""))
            for item in references or []
        ]
        request = ServeTTSRequest(
            text=text,
            max_new_tokens=max_new_tokens,
            references=refs,
            reference_id=reference_id or None,
            seed=seed,
            top_p=0.7,
            temperature=0.7,
            format="wav",
            streaming=False,
        )
        sample_rate = 44100
        audio = np.zeros(0, dtype=np.float32)
        for result in self._engine.inference(request):
            if result.code == "final":
                sample_rate, audio = result.audio
            elif result.code == "error":
                raise RuntimeError("fish_speech_inference_failed")
        if sample_rate <= 0 or audio.size == 0 or not np.isfinite(audio).all():
            raise RuntimeError("fish_speech_invalid_output")
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        return buffer.getvalue()

    async def close(self) -> None:
        if hasattr(self, "_engine"):
            del self._engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
