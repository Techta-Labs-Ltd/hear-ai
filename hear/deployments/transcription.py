import gc
import json
import logging
import os
import tempfile
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import whisperx
from whisperx.asr_qwen import load_model as load_qwen_asr_model
from ray import serve

from hear.config import settings
from hear.core.hear_temp import hear_temp_directory
from hear.services.transcription.chunks import (
    adaptive_batch_size,
    append_shifted_result,
    finalize_combined_result,
    iter_audio_chunks,
)

logger = logging.getLogger(__name__)

_ORIG_PAD = F.pad

def _patched_pad(input, pad, mode="constant", value=None):
    if isinstance(input, np.ndarray):
        input = torch.from_numpy(input)
    return _ORIG_PAD(input, pad, mode=mode, value=value)

F.pad = _patched_pad

@serve.deployment(
    name="transcription",
    ray_actor_options={"num_gpus": 0.20, "num_cpus": 0.3},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 1,
        "target_num_ongoing_requests_per_replica": 1,
        "upscale_delay_s": 5.0,
        "downscale_delay_s": 600.0,
    },
    health_check_period_s=5,
    health_check_timeout_s=300,
    graceful_shutdown_timeout_s=30,
)
class TranscriptionDeployment:
    def __init__(self) -> None:
        self._cuda_healthy = True
        logger.info("Loading WhisperX Qwen3-ASR + Qwen3 ForcedAligner ...")
        self._asr = load_qwen_asr_model(
            settings.QWEN_ASR_MODEL_PATH,
            device="cuda",
            language="en",
            download_root=settings.MODEL_CACHE_DIR,
            local_files_only=True,
            vad_options={
                "vad_onset": settings.WHISPER_VAD_ONSET,
                "vad_offset": settings.WHISPER_VAD_OFFSET,
            },
            qwen_dtype=settings.QWEN_ASR_DTYPE,
            qwen_device_map=settings.QWEN_ASR_DEVICE_MAP,
            qwen_forced_aligner=settings.ALIGNER_MODEL_PATH,
            max_inference_batch_size=settings.WHISPER_BATCH_SIZE,
        )
        # qwen-asr delegates generation to its nested thinker model. Its
        # GenerationConfig does not inherit the repository's pad token, so
        # Transformers otherwise logs the same fallback warning for every
        # inference batch. 151643 is the model's declared padding/EOS token.
        qwen_wrapper = getattr(self._asr, "model", None)
        backend_model = getattr(qwen_wrapper, "model", None)
        thinker = getattr(backend_model, "thinker", None)
        generation_config = getattr(thinker, "generation_config", None)
        if generation_config is not None and generation_config.pad_token_id is None:
            generation_config.pad_token_id = 151643

        logger.info("WhisperX Qwen3-ASR + Qwen3 ForcedAligner ready")

    async def transcribe(self, audio_bytes: bytes, batch_size: int) -> str:
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, dir=hear_temp_directory()
        ) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            audio = whisperx.load_audio(tmp_path)
            duration_seconds = len(audio) / 16000
            effective_batch_size = adaptive_batch_size(
                duration_seconds,
                batch_size,
                settings.WHISPER_LONG_AUDIO_BATCH_SIZE,
            )
            combined: dict[str, Any] = {
                "segments": [],
                "language": "en",
                "audio_duration": duration_seconds,
            }
            for offset_seconds, chunk in iter_audio_chunks(
                audio,
                sample_rate=16000,
                chunk_seconds=settings.WHISPER_CHUNK_SECONDS,
            ):
                with torch.no_grad():
                    result = self._asr.transcribe(
                        chunk,
                        batch_size=effective_batch_size,
                        language="en",
                    )
                append_shifted_result(
                    combined,
                    result,
                    offset_seconds=offset_seconds,
                )
                del result
                torch.cuda.empty_cache()
            return json.dumps(finalize_combined_result(combined))
        except (IndexError, ValueError) as e:
            logger.warning("Transcription produced no output (likely no speech detected): %s", e)
            return json.dumps({"segments": [], "language": "en", "text": ""})
        except RuntimeError as exc:
            if "cuda" in str(exc).lower() or "out of memory" in str(exc).lower():
                self._cuda_healthy = False
                logger.exception("Fatal CUDA transcription error; replica will be restarted")
            raise
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            gc.collect()
            torch.cuda.empty_cache()

    def check_health(self) -> None:
        if not self._cuda_healthy:
            raise RuntimeError("transcription CUDA context requires replica restart")

    def __del__(self) -> None:
        for attr in ("_asr",):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        torch.cuda.empty_cache()

