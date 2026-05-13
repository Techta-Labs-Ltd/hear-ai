import asyncio
from functools import partial

from app.config import settings
from app.core.audio_utils import export_speed_mp3_from_file, speed_layer_filename_stem
from app.core.hear_temp import drop_temp_standalone
from app.core.storage import storage


async def upload_pipeline_speed_layers(
    *,
    track_id: str,
    job_id: str,
    run_id: str,
    source_path: str,
    speed_list: list[float],
    bitrate_kbps: int | None = None,
) -> list[dict]:
    kbps = bitrate_kbps if bitrate_kbps is not None else settings.PIPELINE_MP3_BITRATE_KBPS
    loop = asyncio.get_event_loop()
    layers: list[dict] = []
    for speed in speed_list:
        stem = speed_layer_filename_stem(speed)
        b2_key = f"{settings.B2_PIPELINE_MP3_PREFIX}{track_id}/speed/{stem}.mp3"
        mp3_local = await loop.run_in_executor(
            None,
            partial(
                export_speed_mp3_from_file,
                source_path,
                speed,
                kbps,
                job_id=job_id,
                run_id=run_id,
                track_id=track_id,
                purpose=f"speed_{stem}",
            ),
        )
        try:
            url = await loop.run_in_executor(
                None,
                partial(storage.upload_file, mp3_local, b2_key, "audio/mpeg"),
            )
            layers.append({"speed": speed, "audio_url": url, "b2_key": b2_key, "audio_format": "mp3"})
        finally:
            drop_temp_standalone(mp3_local)
    return sorted(layers, key=lambda x: x["speed"])
