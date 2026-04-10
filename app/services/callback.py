import asyncio

import httpx

from app.config import settings


class CallbackService:
    async def send(self, callback_url: str, payload: dict, retries: int = 3) -> bool:
        headers = {
            "X-Service-Key": settings.AI_SERVICE_SECRET,
            "Content-Type": "application/json",
        }
        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.post(callback_url, json=payload, headers=headers)
                    if response.status_code < 300:
                        return True
            except Exception:
                if attempt == retries - 1:
                    return False
                await asyncio.sleep(2 ** attempt)
        return False


callback_service = CallbackService()
