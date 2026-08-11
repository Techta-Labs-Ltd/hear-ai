import asyncio
import hashlib
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import ray
import torch
from rapidfuzz import distance as rapidfuzz_dist, fuzz as rapidfuzz
from sentence_transformers import SentenceTransformer

from ..cache.ray_cache import get_cache_actor
from hear.config import settings

logger = logging.getLogger(__name__)

class _BatchProcessor:
    """Collects tokens from concurrent tasks, encodes them in a single GPU call."""

    def __init__(self, model: Any, model_lock: Any, max_batch_size: int = 16, max_wait_ms: float = 8.0) -> None:
        self._model = model
        self._model_lock = model_lock
        self._max_batch = max_batch_size
        self._max_wait = max_wait_ms / 1000.0
        self._queue: list[tuple[str, asyncio.Future]] = []
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _encode_batch(self, tokens: list[str]) -> Any:
        with self._model_lock:
            return self._model.encode(tokens, convert_to_numpy=True, normalize_embeddings=True)

    async def _process(self, batch: list[tuple[str, asyncio.Future]]) -> None:
        tokens = [item[0] for item in batch]
        try:
            loop = asyncio.get_running_loop()
            embeddings = await loop.run_in_executor(self._executor, self._encode_batch, tokens)
            for (_, fut), emb in zip(batch, embeddings):
                if not fut.done():
                    fut.set_result(emb)
        except Exception as e:
            for _, fut in batch:
                if not fut.done():
                    fut.set_exception(e)

    async def _drain_loop(self) -> None:
        while True:
            await asyncio.sleep(self._max_wait)
            async with self._lock:
                batch = self._queue[:]
                self._queue.clear()
            if batch:
                asyncio.create_task(self._process(batch))

    async def submit(self, token: str) -> Any:
        fut: asyncio.Future = asyncio.Future()
        async with self._lock:
            self._queue.append((token, fut))
            if len(self._queue) >= self._max_batch:
                batch = self._queue[:]
                self._queue.clear()
                asyncio.create_task(self._process(batch))
        return await fut

_ENTITY_LABEL = {
    "category": ("categories.json", "canonical"),
    "creator": ("creators.json", "normalized"),
    "org": ("organisations.json", "normalized"),
    "tag": ("tags.json", "normalized"),
    "location": ("locations.json", "normalized"),
}


class SemanticMatcher:
    def __init__(self) -> None:
        self._model: Any = None
        self._model_staging: Any = None
        self._active_lock = threading.Lock()
        self._staging_lock = threading.Lock()
        self._device = "cpu"
        self._batcher: _BatchProcessor | None = None
        self._matrices: dict[str, Any] = {}
        self._records: dict[str, list[dict]] = {}
        self._cache: dict[tuple[str, str], tuple[dict | None, float]] = {}
        self.enabled = settings.RESOLVER_SEMANTIC_ENABLED
        self._kv = None

    def _load_model_instance(self, device: str, label: str) -> Any:
        m = SentenceTransformer(
            settings.RESOLVER_SEMANTIC_MODEL,
            device=device,
            local_files_only=True,
        )
        if device == "cuda":
            m.half()
        logger.info("semantic_model_loaded_%s model=%s device=%s", label, settings.RESOLVER_SEMANTIC_MODEL, device)
        return m

    def load_model(self) -> None:
        """Load the transformer once at startup (before forking workers ideally)."""
        if not self.enabled or self._model is not None:
            return
        try:
            self._kv = get_cache_actor()
            device = settings.RESOLVER_SEMANTIC_DEVICE
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device
            if device == "cuda" and settings.RESOLVER_GPU_MEM_FRACTION > 0:
                try:
                    torch.cuda.set_per_process_memory_fraction(settings.RESOLVER_GPU_MEM_FRACTION, 0)
                    logger.info("gpu_mem_fraction set=%.3f", settings.RESOLVER_GPU_MEM_FRACTION)
                except Exception:
                    logger.warning("gpu_mem_fraction set failed", exc_info=True)
            self._model = self._load_model_instance(device, "active")
            self._model_staging = self._load_model_instance(device, "staging")
            self._batcher = _BatchProcessor(self._model, self._active_lock)
            asyncio.create_task(self._batcher._drain_loop())
            logger.info("semantic_batcher_started")
        except Exception:
            logger.exception("semantic model load failed; Tier-3 disabled")
            self.enabled = False

    def build(self, index: dict) -> None:
        """Precompute entity embeddings on staging model, then hot-swap."""
        if not self.enabled:
            return
        if self._model is None:
            self.load_model()
        if self._model is None:
            return
        matrices, records = {}, {}
        for etype, (shard, field) in _ENTITY_LABEL.items():
            recs = index.get(shard, []) or []
            labels = [str(r.get(field, "")).strip() for r in recs]
            keep = [(r, l) for r, l in zip(recs, labels) if l]
            if not keep:
                continue
            recs2, labels2 = zip(*keep)
            with self._staging_lock:
                emb = self._model_staging.encode(
                    list(labels2), convert_to_numpy=True, normalize_embeddings=True, batch_size=256
                )
            matrices[etype] = emb.astype("float32")
            records[etype] = list(recs2)
        self._matrices = matrices
        self._records = records
        self._cache.clear()
        if self._kv is not None:
            try:
                ray.get(self._kv.clear_prefix.remote(f"{settings.RESOLVER_CACHE_NAMESPACE}sem:"))
            except Exception:
                logger.warning("semantic_cache_flush_failed", exc_info=True)
        # Hot-swap: staging becomes active (zero-downtime, active keeps serving)
        self._model, self._model_staging = self._model_staging, self._model
        self._active_lock, self._staging_lock = self._staging_lock, self._active_lock
        self._batcher._model = self._model
        self._batcher._model_lock = self._active_lock
        logger.info("semantic_embeddings_built types=%s", {k: len(v) for k, v in records.items()})

    def _cache_key(self, token: str, entity_type: str) -> str:
        digest = hashlib.md5(token.lower().encode()).hexdigest()[:16]
        return f"{settings.RESOLVER_CACHE_NAMESPACE}sem:{entity_type}:{digest}"

    async def match(self, token: str, entity_type: str) -> tuple[dict | None, float, list[tuple[dict, float]]]:
        if not self.enabled or not token or self._model is None:
            return None, 0.0, []
        key = (token, entity_type)
        cached = self._cache.get(key)
        if cached is not None:
            rec, score = cached
            return rec, score, []
        rkey = self._cache_key(token, entity_type)
        if self._kv is not None:
            try:
                raw = await self._kv.get.remote(rkey)
                if raw:
                    data = json.loads(raw)
                    result = (data.get("record"), data.get("score", 0.0))
                    self._cache[key] = result
                    return result[0], result[1], []
            except Exception:
                pass
        mat = self._matrices.get(entity_type)
        recs = self._records.get(entity_type)
        if mat is None or not recs:
            self._cache[key] = (None, 0.0)
            return None, 0.0, []
        try:
            q = await self._batcher.submit(token)
            sims = mat @ q
            n_candidates = min(3, len(recs))
            indices = np.argsort(sims)[-n_candidates:][::-1]
            idx0 = int(indices[0])
            score0 = float(sims[idx0])
            score0_pct = round(score0 * 100, 1)

            def _reject(rec: dict, score_pct: float) -> bool:
                if score_pct < settings.RESOLVER_SEMANTIC_THRESHOLD * 100:
                    return True
                primary = str(
                    rec.get("canonical")
                    or rec.get("normalized")
                    or rec.get("city")
                    or rec.get("name")
                    or ""
                )
                if score_pct < 85.0 and entity_type == "location":
                    if rapidfuzz_dist.Levenshtein.distance(token.lower(), primary.lower()) > 3:
                        return True
                if rapidfuzz.partial_ratio(token.lower(), primary.lower()) < 20.0 and score_pct < 60.0:
                    return True
                return False

            best = (recs[idx0], score0_pct) if score0 >= settings.RESOLVER_SEMANTIC_THRESHOLD else (None, 0.0)
            if best[0] is not None and _reject(best[0], best[1]):
                best = (None, 0.0)

            candidates_list: list[tuple[dict, float]] = []
            floor = score0_pct - 10.0
            for i in indices[1:]:
                s = round(float(sims[i]) * 100, 1)
                if s < floor:
                    break
                rec_i = recs[i]
                if rec_i is best[0] or _reject(rec_i, s):
                    continue
                candidates_list.append((rec_i, s))

            result = best if best[0] else (None, 0.0)
            self._cache[key] = result
            if best[0] and self._kv is not None:
                try:
                    await self._kv.set.remote(rkey, json.dumps({"record": best[0], "score": best[1]}), settings.RESOLVER_CACHE_TTL_ENTITY)
                except Exception:
                    pass
            return best[0], best[1], candidates_list
        except Exception:
            logger.warning("semantic match failed", exc_info=True)
            self._cache[key] = (None, 0.0)
            return None, 0.0, []
