from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any


class RequestBodyLimitMiddleware:
    def __init__(self, app: Callable, max_bytes: int) -> None:
        if max_bytes < 1:
            raise ValueError("invalid_request_body_limit")
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope.get("type") != "http" or scope.get("path") != "/v1/attempts/stream":
            await self._app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or ())
        try:
            content_length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            await self._reject(send)
            return
        if content_length > self._max_bytes:
            await self._reject(send)
            return
        messages = []
        total = 0
        while True:
            message = await receive()
            messages.append(message)
            if message.get("type") == "http.request":
                total += len(message.get("body") or b"")
                if total > self._max_bytes:
                    await self._reject(send)
                    return
                if not message.get("more_body", False):
                    break
            else:
                break

        async def replay() -> dict[str, Any]:
            if messages:
                return messages.pop(0)
            return await receive()

        await self._app(scope, replay, send)

    @staticmethod
    async def _reject(send: Callable[[dict[str, Any]], Awaitable[None]]) -> None:
        body = b'{"detail":"request_body_too_large"}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
