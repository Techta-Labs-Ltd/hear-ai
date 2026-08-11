import os
import tempfile

from ray import serve

from hear.config import settings
from hear.core.hear_temp import hear_temp_directory
from hear.core.storage import B2Storage
from hear.models.schemas import StorageContext
from hear.services.magic_clean.service import MagicCleanAudioEnhancer


@serve.deployment(
    name="magic_clean",
    # The resident actors leave room for one 0.35-GPU on-demand actor. Using
    # the same reservation for Fish Speech prevents both loading together.
    ray_actor_options={"num_gpus": 0.35, "num_cpus": 0.5},
    autoscaling_config={
        "min_replicas": 0,
        "max_replicas": settings.MAGIC_CLEAN_REPLICA_COUNT,
        "target_num_ongoing_requests_per_replica": 1,
        "upscale_delay_s": 0.0,
        "downscale_delay_s": settings.GPU_ON_DEMAND_IDLE_SECONDS,
    },
    max_ongoing_requests=1,
    health_check_period_s=10,
    health_check_timeout_s=300,
)
class MagicCleanDeployment:
    def __init__(self) -> None:
        self._enhancer = MagicCleanAudioEnhancer()
        self._enhancer.load()

    async def enhance(
        self,
        audio_bytes: bytes,
        track_id: str,
        job_id: str,
        ai_job_id: str | None = None,
        ai_run_id: str | None = None,
        speech: int | None = None,
        music: int | None = None,
        background: int | None = None,
        cut_silence: bool = False,
        storage_context: dict | None = None,
    ) -> dict:
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, dir=hear_temp_directory()
        ) as audio_file:
            audio_file.write(audio_bytes)
            input_path = audio_file.name
        try:
            result = await self._enhancer.enhance(
                input_path=input_path,
                track_id=track_id,
                job_id=job_id,
                ai_job_id=ai_job_id,
                ai_run_id=ai_run_id,
                speech=speech,
                music=music,
                background=background,
                cut_silence=cut_silence,
                storage=B2Storage(StorageContext.model_validate(storage_context or {})),
            )
            return result.__dict__
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
