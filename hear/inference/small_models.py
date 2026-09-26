from __future__ import annotations

import gc
from pathlib import Path
from threading import Lock
from typing import Any, cast

import torch
from transformers import pipeline

from hear.execution.native import NativeExecutor


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
    ) -> dict:
        return await self._native.run(
            self._infer,
            model_name,
            text,
            candidates,
            hypothesis_template,
        )

    def infer_sync(
        self,
        model_name: str,
        text: str,
        candidates: list[str] | None = None,
        hypothesis_template: str | None = None,
    ) -> dict:
        with self._lock:
            return self._infer(model_name, text, candidates, hypothesis_template)

    def _infer(
        self,
        model_name: str,
        text: str,
        candidates: list[str] | None,
        hypothesis_template: str | None,
    ) -> dict:
        if model_name == "toxic_bert":
            result = self._toxic(text[:512], truncation=True)
            return {
                "labels": [item["label"] for item in result],
                "scores": [float(item["score"]) for item in result],
            }
        if model_name == "sentiment":
            result = self._sentiment(text[:512], truncation=True)
            return {
                "labels": [item["label"] for item in result],
                "scores": [float(item["score"]) for item in result],
            }
        if model_name == "nli":
            kwargs: dict[str, Any] = {}
            if hypothesis_template:
                kwargs["hypothesis_template"] = hypothesis_template
            result = self._nli(text[:1024], candidates or [], **kwargs)
            return {
                "labels": list(result["labels"]),
                "scores": [float(score) for score in result["scores"]],
            }
        raise ValueError("unsupported_small_model")

    def check_health(self) -> None:
        if not all(hasattr(self, name) for name in ("_toxic", "_sentiment", "_nli")):
            raise RuntimeError("small_models_unavailable")

    async def close(self) -> None:
        for name in ("_toxic", "_sentiment", "_nli"):
            if hasattr(self, name):
                delattr(self, name)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
