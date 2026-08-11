import hashlib
import json
import logging
import time

import ray

from hear.config import settings

logger = logging.getLogger(__name__)

_ACTOR_NAME = "resolver_kv_cache"
_ACTOR_NAMESPACE = "resolver"


def _norm(utterance: str) -> str:
    return " ".join(utterance.lower().strip().split())


@ray.remote(num_cpus=0)
class _KVCacheActor:
    def __init__(self) -> None:
        self._store: dict[str, tuple[str, float | None]] = {}

    def get(self, key: str) -> str | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and time.time() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: str, ttl: int | None) -> None:
        expires_at = time.time() + ttl if ttl else None
        self._store[key] = (value, expires_at)

    def clear_prefix(self, prefix: str) -> int:
        keys = [k for k in self._store if k.startswith(prefix)]
        for k in keys:
            del self._store[k]
        return len(keys)


def get_cache_actor():
    return _KVCacheActor.options(
        name=_ACTOR_NAME,
        namespace=_ACTOR_NAMESPACE,
        get_if_exists=True,
        lifetime="detached",
    ).remote()


class RayCache:
    def __init__(self) -> None:
        self._actor = None
        self.enabled = True

    async def connect(self) -> None:
        try:
            self._actor = get_cache_actor()
        except Exception:
            logger.exception("ray_cache actor unavailable; Tier-2 disabled")
            self._actor = None
            self.enabled = False

    def _key(self, utterance: str, country: str) -> str:
        digest = hashlib.sha256(_norm(utterance).encode()).hexdigest()[:24]
        return f"{settings.RESOLVER_CACHE_NAMESPACE}utt:{country}:{digest}"

    async def get_utterance(self, utterance: str, country: str) -> dict | None:
        if not self._actor:
            return None
        try:
            raw = await self._actor.get.remote(self._key(utterance, country))
            return json.loads(raw) if raw else None
        except Exception:
            logger.warning("ray_cache get failed", exc_info=True)
            return None

    async def set_utterance(self, utterance: str, country: str, payload: dict) -> None:
        if not self._actor:
            return
        try:
            await self._actor.set.remote(
                self._key(utterance, country),
                json.dumps(payload, default=str),
                settings.RESOLVER_CACHE_TTL_UTTERANCE,
            )
        except Exception:
            logger.warning("ray_cache set failed", exc_info=True)

    async def flush_namespace(self) -> int:
        if not self._actor:
            return 0
        try:
            cleared = await self._actor.clear_prefix.remote(settings.RESOLVER_CACHE_NAMESPACE)
            logger.info("ray_cache flushed keys=%d", cleared)
            return cleared
        except Exception:
            logger.warning("ray_cache flush failed", exc_info=True)
            return 0

    async def close(self) -> None:
        self._actor = None
