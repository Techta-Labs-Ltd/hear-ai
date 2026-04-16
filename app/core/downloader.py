import os
import tempfile

import httpx

AUDIO_CONTENT_TYPES = {
    "audio/wav", "audio/wave", "audio/x-wav",
    "audio/mpeg", "audio/mp3", "audio/mp4",
    "audio/ogg", "audio/flac", "audio/aac",
    "audio/webm", "application/octet-stream", "binary/octet-stream",
}


def _ensure_https(url: str) -> str:
    if url.startswith("http://"):
        return "https://" + url[7:]
    return url


async def download_audio(url: str, suffix: str = ".wav") -> str:
    url = _ensure_https(url)
    os.makedirs(tempfile.gettempdir(), exist_ok=True)

    async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        if content_type and content_type not in AUDIO_CONTENT_TYPES and "audio" not in content_type:
            raise ValueError(
                f"Expected audio file but got content-type '{content_type}' from {url}"
            )

        if len(response.content) == 0:
            raise ValueError(f"Downloaded file is empty from {url}")

        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(response.content)
        tmp.flush()
        tmp.close()

        if os.path.getsize(tmp.name) == 0:
            os.unlink(tmp.name)
            raise ValueError(f"Written temp file is empty for {url}")

        return tmp.name


def cleanup_temp(path: str):
    if path and os.path.exists(path):
        os.unlink(path)
