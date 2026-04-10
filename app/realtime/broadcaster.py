import asyncio
import json
import time
from typing import AsyncGenerator

from fastapi import WebSocket, WebSocketDisconnect
from starlette.responses import StreamingResponse


class ConnectionManager:
    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = {}
        self._sse_queues: dict[str, list[asyncio.Queue]] = {}

    async def connect_ws(self, job_id: str, ws: WebSocket):
        await ws.accept()
        self._connections.setdefault(job_id, []).append(ws)

    def disconnect_ws(self, job_id: str, ws: WebSocket):
        if job_id in self._connections:
            self._connections[job_id].remove(ws)
            if not self._connections[job_id]:
                del self._connections[job_id]

    def add_sse_client(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._sse_queues.setdefault(job_id, []).append(q)
        return q

    def remove_sse_client(self, job_id: str, q: asyncio.Queue):
        if job_id in self._sse_queues:
            self._sse_queues[job_id].remove(q)
            if not self._sse_queues[job_id]:
                del self._sse_queues[job_id]

    async def broadcast(self, job_id: str, event: dict):
        payload = json.dumps(event)

        dead_ws = []
        for ws in self._connections.get(job_id, []):
            try:
                await ws.send_text(payload)
            except Exception:
                dead_ws.append(ws)
        for ws in dead_ws:
            self.disconnect_ws(job_id, ws)

        for q in self._sse_queues.get(job_id, []):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass


manager = ConnectionManager()


async def sse_generator(job_id: str, q: asyncio.Queue) -> AsyncGenerator[str, None]:
    try:
        while True:
            try:
                payload = await asyncio.wait_for(q.get(), timeout=30.0)
                yield f"data: {payload}\n\n"
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
    except asyncio.CancelledError:
        pass


def make_sse_response(job_id: str) -> StreamingResponse:
    q = manager.add_sse_client(job_id)

    async def _gen():
        try:
            async for chunk in sse_generator(job_id, q):
                yield chunk
        finally:
            manager.remove_sse_client(job_id, q)

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
