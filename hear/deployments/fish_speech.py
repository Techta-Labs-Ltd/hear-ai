import gc
import io
import logging
import os
import time

import numpy as np
import soundfile as sf
import torch
from ray import serve

from fish_speech.inference_engine import TTSInferenceEngine
from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
from hear.config import settings

logger = logging.getLogger(__name__)

@serve.deployment(
    name="fish_speech",
    ray_actor_options={
        "num_gpus": 0.35,
        "num_cpus": 0.3,
        "runtime_env": {
            "env_vars": {
                "FISH_SPEECH_HOME": settings.FISH_SPEECH_HOME,
                "PYTHONPATH": settings.FISH_SPEECH_HOME,
            }
        },
    },
    autoscaling_config={
        "min_replicas": 0,
        "max_replicas": settings.FISH_SPEECH_REPLICA_COUNT,
        "target_num_ongoing_requests_per_replica": 1,
        "upscale_delay_s": 0.0,
        "downscale_delay_s": settings.GPU_ON_DEMAND_IDLE_SECONDS,
    },
    max_ongoing_requests=1,
    health_check_period_s=60,
    health_check_timeout_s=600,
    graceful_shutdown_timeout_s=120,
)
class FishSpeechDeployment:
    def __init__(self) -> None:
        checkpoint = settings.FISH_SPEECH_CHECKPOINT_PATH
        codec = os.path.join(checkpoint, "codec.pth")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading Fish Speech with BNB mode %s...", settings.FISH_SPEECH_BNB_MODE)
        t0 = time.time()

        llama_queue, self._llama_thread = launch_thread_safe_queue(
            checkpoint_path=checkpoint,
            device=device,
            precision=torch.bfloat16,
            compile=False,
            bnb_mode=settings.FISH_SPEECH_BNB_MODE or None,
            lazy_load=False,
        )
        decoder = load_decoder_model(
            config_name="modded_dac_vq",
            checkpoint_path=codec,
            device=device,
        )
        self._engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder,
            precision=torch.bfloat16,
            compile=False,
        )
        logger.info("Fish Speech loaded in %.1fs", time.time() - t0)

    async def generate_speech(
        self,
        text: str,
        max_new_tokens: int = 1024,
        references: list[dict] | None = None,
        reference_id: str | None = None,
        language: str = "en",
    ) -> bytes:
        refs = []
        if references:
            for r in references:
                refs.append(ServeReferenceAudio(audio=r.get("audio", b""), text=r.get("text", "")))

        req = ServeTTSRequest(
            text=text,
            max_new_tokens=max_new_tokens,
            references=refs,
            reference_id=reference_id or None,
            top_p=0.7,
            temperature=0.7,
            format="wav",
            streaming=False,
        )
        sample_rate = 44100
        audio = np.zeros(0)
        for result in self._engine.inference(req):
            if result.code == "final":
                sample_rate, audio = result.audio
        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV")
        return buf.getvalue()

    def __del__(self) -> None:
        if hasattr(self, "_engine") and self._engine is not None:
            del self._engine
        gc.collect()
        torch.cuda.empty_cache()
