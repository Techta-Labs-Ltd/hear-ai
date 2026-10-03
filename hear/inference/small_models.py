from __future__ import annotations

import gc
from pathlib import Path
from threading import Lock
from typing import Any, cast

import torch
from transformers import pipeline

from hear.execution.native import NativeExecutor
from hear.runtime.gpu_idle import SyncIdleResource


class SmallModelsEngine:
    def __init__(
        self,
        toxic_path: Path,
        sentiment_path: Path,
        nli_path: Path,
        native: NativeExecutor,
        *,
        device: int = 0,
    ) -> None:
        if device >= 0 and not torch.cuda.is_available():
            raise RuntimeError("small_models_cuda_unavailable")
        self._native = native
        self._lock = Lock()
        pipeline_factory = cast(Any, pipeline)
        self._toxic = pipeline_factory(
            "text-classification",
            model=str(toxic_path),
            device=device,
            model_kwargs={"local_files_only": True},
        )
        self._sentiment = pipeline_factory(
            "sentiment-analysis",
            model=str(sentiment_path),
            device=device,
            model_kwargs={"local_files_only": True},
        )
        self._nli = pipeline_factory(
            "zero-shot-classification",
            model=str(nli_path),
            device=device,
            model_kwargs={"local_files_only": True},
        )

    async def infer(
        self,
        model_name: str,
        text: str,
        candidates: list[str] | None = None,
        hypothesis_template: str | None = None,
        *,
        multi_label: bool = False,
    ) -> dict:
        return await self._native.run(
            self.infer_sync,
            model_name,
            text,
            candidates,
            hypothesis_template,
            multi_label=multi_label,
        )

    def infer_sync(
        self,
        model_name: str,
        text: str,
        candidates: list[str] | None = None,
        hypothesis_template: str | None = None,
        *,
        multi_label: bool = False,
    ) -> dict:
        with self._lock:
            return self._infer(model_name, text, candidates, hypothesis_template, multi_label)

    def _infer(
        self,
        model_name: str,
        text: str,
        candidates: list[str] | None,
        hypothesis_template: str | None,
        multi_label: bool = False,
    ) -> dict:
        if model_name == "toxic_bert":
            result = self._toxic(text[:512], truncation=True, top_k=None)
            return {
                "labels": [item["label"] for item in result],
                "scores": [float(item["score"]) for item in result],
            }
        if model_name == "sentiment":
            result = self._sentiment(text[:512], truncation=True)
            return {
                "label": result[0]["label"],
                "labels": [item["label"] for item in result],
                "scores": [float(item["score"]) for item in result],
            }
        if model_name == "nli":
            kwargs: dict[str, Any] = {}
            if hypothesis_template:
                kwargs["hypothesis_template"] = hypothesis_template
            result = self._nli(
                text[:1024], candidates or [], multi_label=multi_label, batch_size=16, **kwargs
            )
            return {
                "labels": list(result["labels"]),
                "scores": [float(score) for score in result["scores"]],
            }
        raise ValueError("unsupported_small_model")

    def check_health(self) -> None:
        if not all(hasattr(self, name) for name in ("_toxic", "_sentiment", "_nli")):
            raise RuntimeError("small_models_unavailable")

    def unload(self) -> None:
        for name in ("_toxic", "_sentiment", "_nli"):
            if hasattr(self, name):
                delattr(self, name)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    async def close(self) -> None:
        self.unload()


class LazySmallModelsEngine:
    """Lazy-load GPU classifiers and evict them after the pipeline is idle."""

    def __init__(
        self,
        toxic_path: Path,
        sentiment_path: Path,
        nli_path: Path,
        native: NativeExecutor,
        *,
        device: int = 0,
        idle_seconds: float,
        eviction_enabled: bool,
    ) -> None:
        self._paths = (toxic_path, sentiment_path, nli_path)
        self._native = native
        self._device = device
        self._resource = SyncIdleResource(
            "small_models",
            lambda: SmallModelsEngine(*self._paths, native, device=device),
            lambda engine: engine.unload(),
            idle_seconds=idle_seconds,
            eviction_enabled=eviction_enabled,
        )

    async def warmup(self) -> None:
        def load() -> None:
            self._resource.acquire()
            self._resource.release()

        await self._native.run(load)

    def infer_sync(
        self,
        model_name: str,
        text: str,
        candidates: list[str] | None = None,
        hypothesis_template: str | None = None,
        *,
        multi_label: bool = False,
    ) -> dict:
        engine = self._resource.acquire()
        try:
            return engine.infer_sync(
                model_name, text, candidates, hypothesis_template, multi_label=multi_label
            )
        finally:
            self._resource.release()

    async def infer(
        self,
        model_name: str,
        text: str,
        candidates: list[str] | None = None,
        hypothesis_template: str | None = None,
        *,
        multi_label: bool = False,
    ) -> dict:
        return await self._native.run(
            self.infer_sync,
            model_name,
            text,
            candidates,
            hypothesis_template,
            multi_label=multi_label,
        )

    def check_health(self) -> None:
        if self._resource.state == "failed":
            raise RuntimeError("small_models_lazy_engine_failed")
        if any(not path.is_dir() for path in self._paths):
            raise RuntimeError("small_models_assets_missing")
        if self._device >= 0 and not torch.cuda.is_available():
            raise RuntimeError("small_models_cuda_unavailable")

    @property
    def lifecycle(self) -> dict:
        return self._resource.snapshot

    def evict_now(self) -> bool:
        return self._resource.evict_now()

    async def close(self) -> None:
        self._resource.close()
