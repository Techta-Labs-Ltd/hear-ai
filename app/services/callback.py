import asyncio

import httpx

from app.config import settings


class CallbackService:
    MAX_RETRIES = 10
    BASE_DELAY = 2
    MAX_DELAY = 300

    async def send(self, callback_url: str, payload: dict) -> bool:
        headers = {
            "X-Service-Key": settings.AI_SERVICE_SECRET,
            "Content-Type": "application/json",
        }
        for attempt in range(self.MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.post(
                        callback_url, json=payload, headers=headers,
                    )
                    if response.status_code < 300:
                        return True
                    if response.status_code in (400, 401, 403, 404, 422):
                        print(
                            f"[CALLBACK] Permanent failure {response.status_code} "
                            f"for {callback_url}, not retrying"
                        )
                        return False
            except Exception as e:
                print(
                    f"[CALLBACK] Attempt {attempt + 1}/{self.MAX_RETRIES} "
                    f"failed for {callback_url}: {e}"
                )

            delay = min(self.BASE_DELAY * (2 ** attempt), self.MAX_DELAY)
            await asyncio.sleep(delay)

        print(f"[CALLBACK] All {self.MAX_RETRIES} attempts exhausted for {callback_url}")
        return False


callback_service = CallbackService()
