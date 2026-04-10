import os
import tempfile

import httpx


async def download_audio(url: str, suffix: str = ".wav") -> str:
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.get(url)
        response.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(response.content)
        tmp.close()
        return tmp.name


def cleanup_temp(path: str):
    if path and os.path.exists(path):
        os.unlink(path)
