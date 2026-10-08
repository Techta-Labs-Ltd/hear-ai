"""AudioSet sound tagger (Audio Spectrogram Transformer) for telling music from speech."""

from __future__ import annotations

import gc
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from hear.execution.native import NativeExecutor
from hear.runtime.gpu_idle import SyncIdleResource

SAMPLE_RATE = 16000


class AudioTaggerEngine:
    """Multi-label AudioSet scores (sigmoid, not softmax) for 16 kHz mono clips."""

    def __init__(self, model_path: Path, *, device: int = 0) -> None:
        if device >= 0 and not torch.cuda.is_available():
            raise RuntimeError("audio_tagger_cuda_unavailable")
        self._lock = Lock()
        self._device = torch.device(f"cuda:{device}" if device >= 0 else "cpu")
        self._features = AutoFeatureExtractor.from_pretrained(
            str(model_path), local_files_only=True
        )
        model = AutoModelForAudioClassification.from_pretrained(
            str(model_path), local_files_only=True
        )
        self._model = model.to(self._device).eval()
        self._labels = {int(index): str(name) for index, name in model.config.id2label.items()}

    def tag_sync(self, clips: list[np.ndarray], labels: tuple[str, ...]) -> list[dict[str, float]]:
        if not clips:
            return []
        wanted = [(name, index) for index, name in self._labels.items() if name in labels]
        with self._lock, torch.inference_mode():
            features = self._features(
                [np.asarray(clip, dtype=np.float32) for clip in clips],
                sampling_rate=SAMPLE_RATE,
                return_tensors="pt",
            )
            logits = self._model(**{k: v.to(self._device) for k, v in features.items()}).logits
            probabilities = torch.sigmoid(logits.float()).cpu().numpy()
        return [
            {name: round(float(row[index]), 4) for name, index in wanted} for row in probabilities
        ]

    def unload(self) -> None:
        if hasattr(self, "_model"):
            del self._model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class LazyAudioTaggerEngine:
    """Lazy-load the tagger on first use and evict it after the pipeline is idle."""

    def __init__(
        self,
        model_path: Path,
        native: NativeExecutor,
        *,
        device: int = 0,
        idle_seconds: float,
        eviction_enabled: bool,
    ) -> None:
        self._path = model_path
        self._native = native
        self._device = device
        self._resource = SyncIdleResource(
            "audio_tagger",
            lambda: AudioTaggerEngine(model_path, device=device),
            lambda engine: engine.unload(),
            idle_seconds=idle_seconds,
            eviction_enabled=eviction_enabled,
        )

    async def warmup(self) -> None:
        def load() -> None:
            self._resource.acquire()
            self._resource.release()

        await self._native.run(load)

    def tag_sync(self, clips: list[np.ndarray], labels: tuple[str, ...]) -> list[dict[str, float]]:
        engine = self._resource.acquire()
        try:
            return engine.tag_sync(clips, labels)
        finally:
            self._resource.release()

    def check_health(self) -> None:
        if self._resource.state == "failed":
            raise RuntimeError("audio_tagger_lazy_engine_failed")
        if not self._path.is_dir():
            raise RuntimeError("audio_tagger_assets_missing")
        if self._device >= 0 and not torch.cuda.is_available():
            raise RuntimeError("audio_tagger_cuda_unavailable")

    @property
    def lifecycle(self) -> dict:
        return self._resource.snapshot

    def evict_now(self) -> bool:
        return self._resource.evict_now()

    async def close(self) -> None:
        self._resource.close()
