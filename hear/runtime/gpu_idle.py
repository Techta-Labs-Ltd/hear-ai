"""Single-flight lazy resources with bounded idle eviction.

These lifecycle helpers keep worker/broker processes alive while allowing heavy
GPU engines to be loaded only on demand and evicted after an idle TTL.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable


class AsyncIdleResource[T]:
    def __init__(
        self,
        name: str,
        factory: Callable[[], Awaitable[T]],
        closer: Callable[[T], Awaitable[None]],
        *,
        idle_seconds: float,
        eviction_enabled: bool,
    ) -> None:
        if not name or idle_seconds <= 0:
            raise ValueError("invalid_idle_resource_configuration")
        self.name = name
        self._factory = factory
        self._closer = closer
        self._idle_seconds = idle_seconds
        self._eviction_enabled = eviction_enabled
        self._resource: T | None = None
        self._loading: asyncio.Task[T] | None = None
        self._idle_task: asyncio.Task[None] | None = None
        self._active = 0
        self._closed = False
        self._state = "cold"
        self._cold_starts = 0
        self._evictions = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        return self._state

    @property
    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "state": self._state,
            "active": self._active,
            "cold_starts": self._cold_starts,
            "evictions": self._evictions,
            "idle_seconds": self._idle_seconds,
            "eviction_enabled": self._eviction_enabled,
        }

    async def _load_and_store(self) -> T:
        try:
            resource = await self._factory()
        except BaseException:
            async with self._lock:
                self._loading = None
                self._state = "failed"
            raise
        async with self._lock:
            if self._closed:
                self._loading = None
                self._state = "closed"
                close_now = True
            else:
                self._resource = resource
                self._loading = None
                self._cold_starts += 1
                self._state = "busy" if self._active else "warm"
                close_now = False
        if close_now:
            await self._closer(resource)
            raise RuntimeError(f"{self.name}_resource_closed_during_load")
        return resource

    async def acquire(self) -> T:
        async with self._lock:
            if self._closed:
                raise RuntimeError(f"{self.name}_resource_closed")
            if self._idle_task is not None:
                self._idle_task.cancel()
                self._idle_task = None
            self._active += 1
            if self._resource is not None:
                self._state = "busy"
                return self._resource
            if self._loading is None:
                self._state = "warming"
                self._loading = asyncio.create_task(self._load_and_store())
            task = self._loading
        try:
            return await asyncio.shield(task)
        except BaseException:
            await self.release()
            raise

    async def release(self) -> None:
        async with self._lock:
            if self._active > 0:
                self._active -= 1
            if self._closed or self._active:
                return
            if self._resource is None:
                if self._loading is None and self._state != "failed":
                    self._state = "cold"
                return
            self._state = "warm"
            if self._eviction_enabled:
                self._idle_task = asyncio.create_task(self._evict_after_idle())

    async def _evict_after_idle(self) -> None:
        current = asyncio.current_task()
        try:
            await asyncio.sleep(self._idle_seconds)
            async with self._lock:
                if self._closed or self._active or self._resource is None:
                    return
                self._state = "evicting"
                resource = self._resource
                self._resource = None
                # Keep the state lock while closing so a new job cannot start
                # another heavyweight load before the previous engine releases VRAM.
                await self._closer(resource)
                self._evictions += 1
                self._state = "cold"
        except asyncio.CancelledError:
            return
        finally:
            async with self._lock:
                if self._idle_task is current:
                    self._idle_task = None

    async def evict_now(self) -> bool:
        async with self._lock:
            if self._closed or self._active or self._resource is None:
                return False
            if self._idle_task is not None:
                self._idle_task.cancel()
                self._idle_task = None
            self._state = "evicting"
            resource = self._resource
            self._resource = None
            await self._closer(resource)
            self._evictions += 1
            self._state = "cold"
            return True

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._idle_task is not None:
                self._idle_task.cancel()
                self._idle_task = None
            loading = self._loading
        if loading is not None:
            try:
                await asyncio.shield(loading)
            except BaseException:
                pass
        async with self._lock:
            resource = self._resource
            self._resource = None
            self._state = "closed"
            self._active = 0
        if resource is not None:
            await self._closer(resource)


class SyncIdleResource[T]:
    def __init__(
        self,
        name: str,
        factory: Callable[[], T],
        closer: Callable[[T], None],
        *,
        idle_seconds: float,
        eviction_enabled: bool,
    ) -> None:
        if not name or idle_seconds <= 0:
            raise ValueError("invalid_idle_resource_configuration")
        self.name = name
        self._factory = factory
        self._closer = closer
        self._idle_seconds = idle_seconds
        self._eviction_enabled = eviction_enabled
        self._resource: T | None = None
        self._timer: threading.Timer | None = None
        self._active = 0
        self._closed = False
        self._state = "cold"
        self._cold_starts = 0
        self._evictions = 0
        self._lock = threading.RLock()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "name": self.name,
                "state": self._state,
                "active": self._active,
                "cold_starts": self._cold_starts,
                "evictions": self._evictions,
                "idle_seconds": self._idle_seconds,
                "eviction_enabled": self._eviction_enabled,
            }

    def acquire(self) -> T:
        with self._lock:
            if self._closed:
                raise RuntimeError(f"{self.name}_resource_closed")
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._active += 1
            if self._resource is None:
                self._state = "warming"
                try:
                    self._resource = self._factory()
                except BaseException:
                    self._active -= 1
                    self._state = "failed"
                    raise
                self._cold_starts += 1
            self._state = "busy"
            return self._resource

    def release(self) -> None:
        with self._lock:
            if self._active > 0:
                self._active -= 1
            if self._closed or self._active or self._resource is None:
                return
            self._state = "warm"
            if self._eviction_enabled:
                timer = threading.Timer(self._idle_seconds, self._evict_timer)
                timer.daemon = True
                self._timer = timer
                timer.start()

    def _evict_timer(self) -> None:
        with self._lock:
            self._timer = None
            if self._closed or self._active or self._resource is None:
                return
            self._state = "evicting"
            resource = self._resource
            self._resource = None
            self._closer(resource)
            self._evictions += 1
            self._state = "cold"

    def evict_now(self) -> bool:
        with self._lock:
            if self._closed or self._active or self._resource is None:
                return False
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._state = "evicting"
            resource = self._resource
            self._resource = None
            self._closer(resource)
            self._evictions += 1
            self._state = "cold"
            return True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            resource = self._resource
            self._resource = None
            self._active = 0
            self._state = "closed"
            if resource is not None:
                self._closer(resource)
