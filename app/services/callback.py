import asyncio
from typing import Optional

import httpx

from app.config import settings


class CallbackService:
    MAX_RETRIES = 10
    BASE_DELAY = 2
    MAX_DELAY = 300

    def _resolve_url(self, callback_url: Optional[str]) -> Optional[str]:
        url = callback_url or settings.HEAR_CALLBACK_URL
        if url and url.startswith(("http://", "https://")):
            return url
        return None

    async def send(self, callback_url: Optional[str], payload: dict) -> bool:
        url = self._resolve_url(callback_url)
        if not url:
            print(f"[CALLBACK] Invalid callback URL, skipping: {callback_url!r}")
            return False
        headers = {
            "X-Service-Key": settings.AI_SERVICE_SECRET,
            "Content-Type": "application/json",
        }
        for attempt in range(self.MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.post(
                        url, json=payload, headers=headers,
                    )
                    if response.status_code < 300:
                        return True
                    if response.status_code in (400, 401, 403, 404, 422):
                        print(
                            f"[CALLBACK] Permanent failure {response.status_code} "
                            f"for {url}, not retrying"
                        )
                        return False
            except Exception as e:
                print(
                    f"[CALLBACK] Attempt {attempt + 1}/{self.MAX_RETRIES} "
                    f"failed for {url}: {e}"
                )

            delay = min(self.BASE_DELAY * (2 ** attempt), self.MAX_DELAY)
            await asyncio.sleep(delay)

        print(f"[CALLBACK] All {self.MAX_RETRIES} attempts exhausted for {url}")
        return False


callback_service = CallbackService()
