import asyncio
import logging
from collections.abc import Callable
from typing import Any

from .builder import ShardBuilder

logger = logging.getLogger(__name__)

RETRY_MIN_SECONDS = 5
RETRY_MAX_SECONDS = 60


class DoubleBufferManager:
    def __init__(self) -> None:
        self._slot_a: dict[str, Any] = {"version": 0}
        self._loading = False
        self._builder = ShardBuilder()
        self._retry_task: asyncio.Task | None = None
        self.on_ready: Callable[[dict[str, Any]], None] | None = None

    def get_active(self) -> dict[str, Any]:
        return self._slot_a

    def is_ready(self) -> bool:
        return bool(self._slot_a.get("version"))

    async def latest_version(self) -> int:
        return await self._builder.discover_latest_version()

    async def startup(self) -> None:
        logger.info("taxonomy loading scheduled in background")
        self._ensure_retry_task()

    async def load_version(self, version: int | None) -> bool:
        return await self._load(version)

    async def _load(self, version: int | None) -> bool:
        if self._loading:
            return False
        self._loading = True
        try:
            if version is None:
                version = await self._builder.discover_latest_version()
            if version == 0:
                logger.warning("no taxonomy version found on CDN")
                return False
            current = self._slot_a.get("version", 0)
            if version == current and current > 0:
                logger.info("version %d already active, skipping", version)
                return True
            new_index = await self._builder.build(version)
        except Exception:
            logger.exception("taxonomy load failed")
            return False
        finally:
            self._loading = False
        if not new_index.get("version"):
            return False
        self._slot_a = new_index
        logger.info("taxonomy active version=%d", new_index["version"])
        if self.on_ready:
            try:
                self.on_ready(new_index)
            except Exception:
                logger.exception("on_ready callback failed")
        return True

    def _ensure_retry_task(self) -> None:
        if self._retry_task and not self._retry_task.done():
            return
        self._retry_task = asyncio.create_task(self._retry_until_ready())

    async def _retry_until_ready(self) -> None:
        delay = 0
        while not self.is_ready():
            await asyncio.sleep(delay)
            loaded = await self._load(None)
            if loaded:
                return
            delay = (
                RETRY_MIN_SECONDS
                if delay == 0
                else min(delay * 2, RETRY_MAX_SECONDS)
            )
