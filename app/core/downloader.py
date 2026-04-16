import os
import tempfile

import httpx


def _ensure_https(url: str) -> str:
    if url.startswith("http://"):
        return "https://" + url[7:]
    return url


async def download_audio(url: str, suffix: str = ".wav") -> str:
    url = _ensure_https(url)
    async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(response.content)
        tmp.close()
        return tmp.name


def cleanup_temp(path: str):
    if path and os.path.exists(path):
        os.unlink(path)
