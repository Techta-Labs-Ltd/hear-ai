import os
import tempfile
from typing import Optional

import ray.data


async def download_audio(
    url: str,
    suffix: str = ".wav",
    *,
    db=None,
    job_id: Optional[str] = None,
    run_id: Optional[str] = None,
    track_id: Optional[str] = None,
    purpose: str = "audio",
) -> str:
    import asyncio

    ds = ray.data.from_items([{"url": url}])
    ds = ds.with_column("bytes", ray.data.expressions.download("url"))
    row = ds.take(1)[0]
    audio_bytes = row["bytes"]

    dir_path = tempfile.mkdtemp(prefix="hear_", suffix=f"_{purpose}")
    path = os.path.join(dir_path, f"{purpose}{suffix}")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _write_bytes, path, audio_bytes)
    return path


def _write_bytes(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)
