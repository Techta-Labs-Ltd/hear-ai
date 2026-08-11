import asyncio
import hashlib
import json
import logging
from typing import Any

import httpx

from hear.config import settings

logger = logging.getLogger(__name__)


class ShardBuilder:
    def __init__(self) -> None:
        self._cdn_base = ""

    async def discover_latest_version(self) -> int:
        self._cdn_base = settings.RESOLVER_CDN_BASE_URL.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(f"{self._cdn_base}/version.json")
                resp.raise_for_status()
                data = resp.json()
                return int(data.get("version", 0))
        except Exception as exc:
            logger.warning(
                "taxonomy version unavailable; background retry scheduled: %s", exc
            )
            return 0

    async def build(self, version: int) -> dict[str, Any]:
        base = self._cdn_base
        manifest_url = f"{base}/v{version}/manifest.json"

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(manifest_url)
            resp.raise_for_status()
            manifest = resp.json()

        shards = manifest.get("files", [])
        if not shards:
            logger.warning("No shards in manifest v%s", version)
            return {}

        index: dict[str, Any] = {"version": version}

        for shard in shards:
            name = shard["name"]
            url = f"{base}/v{version}/{name}"
            expected_hash = shard.get("hash")

            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=60) as client:
                        resp = await client.get(url)
                        resp.raise_for_status()
                        raw = resp.content

                    if expected_hash:
                        actual_hash = hashlib.sha256(raw).hexdigest()[:len(expected_hash)]
                        if actual_hash != expected_hash:
                            raise ValueError(
                                f"hash mismatch for {name}: expected {expected_hash}, "
                                f"got {actual_hash}"
                            )

                    index[name] = json.loads(raw)
                    logger.info("loaded shard %s (v%s) %d bytes", name, version, len(raw))
                    break
                except Exception as exc:
                    logger.warning("shard %s attempt %d/3 failed: %s", name, attempt + 1, exc)
                    if attempt == 2:
                        logger.error("failed to load shard %s after 3 attempts", name)
                        return {}
                    await asyncio.sleep(2 ** attempt)

        return index
