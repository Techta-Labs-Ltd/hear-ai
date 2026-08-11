import asyncio
import logging

from ray import serve

from hear.config import settings
from hear.core.hear_temp import sweep_tracked_temp_files

logger = logging.getLogger(__name__)

@serve.deployment(
    name="audio_cleanup",
    num_replicas=1,
    ray_actor_options={"num_gpus": 0.0, "num_cpus": 0.05},
    max_ongoing_requests=1,
)
class AudioCleanupDeployment:
    """Periodically remove audio abandoned by interrupted or crashed jobs."""

    def __init__(self) -> None:
        self._task = asyncio.create_task(self._cleanup_loop())

    async def _cleanup_loop(self) -> None:
        while True:
            try:
                result = await asyncio.to_thread(sweep_tracked_temp_files)
                if result["by_age"]:
                    logger.info(
                        "Removed %s stale audio entries (%s bytes)",
                        result["by_age"],
                        result["bytes_freed"],
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Stale audio cleanup failed")
            await asyncio.sleep(settings.AUDIO_CLEANUP_INTERVAL_SECONDS)

    async def run_now(self) -> dict:
        return await asyncio.to_thread(sweep_tracked_temp_files)

    def __del__(self) -> None:
        task = getattr(self, "_task", None)
        if task is not None:
            task.cancel()
