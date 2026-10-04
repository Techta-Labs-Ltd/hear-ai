"""Voice activity detection for transcription windows, run ahead of the GPU.

Silero VAD is a sequential CPU loop that costs about 3.5 s per 240 s of audio,
more than the ASR forward pass itself. Windows are read, resampled and
segmented in worker processes while the GPU transcribes the previous window, so
the job's wall time is bounded by the model and not by the VAD.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import time
from collections.abc import AsyncIterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from hear.inference.qwen_asr import LocalSileroVad
from hear.services.transcription.service import TranscriptionService

SAMPLE_RATE = 16000


@dataclass(frozen=True)
class WindowTask:
    path: str
    start_frame: int
    frames: int


@dataclass
class WindowResult:
    start_frame: int
    samples: np.ndarray
    segments: list[tuple[float, float]]
    read_seconds: float
    vad_seconds: float


class VadWindowPool:
    # The worker process's detector, built once by `initialize`.
    detector: LocalSileroVad | None = None

    def __init__(self, workers: int, onset: float, segment_seconds: int) -> None:
        if workers < 1:
            raise ValueError("vad_pool_needs_a_worker")
        self._workers = workers
        self._onset = onset
        self._segment_seconds = segment_seconds
        self._executor: ProcessPoolExecutor | None = None

    @classmethod
    def initialize(cls, onset: float, segment_seconds: int) -> None:
        cls.detector = LocalSileroVad(onset, segment_seconds)

    @classmethod
    def run_window(cls, task: WindowTask) -> WindowResult:
        if cls.detector is None:
            raise RuntimeError("vad_worker_not_initialized")
        started = time.perf_counter()
        with sf.SoundFile(task.path) as source:
            source.seek(task.start_frame)
            samples = TranscriptionService._read_window(source, task.frames)
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        read_seconds = time.perf_counter() - started
        started = time.perf_counter()
        found = cls.detector({"waveform": samples, "sample_rate": SAMPLE_RATE})
        return WindowResult(
            task.start_frame,
            samples,
            [(item.start, item.end) for item in found],
            read_seconds,
            time.perf_counter() - started,
        )

    def _ensure(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self._workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=self.initialize,
                initargs=(self._onset, self._segment_seconds),
            )
        return self._executor

    async def stream(self, path: str, windows: list[tuple[int, int]]) -> AsyncIterator[WindowResult]:
        """Yield windows in file order; all of them are queued up front."""
        executor = self._ensure()
        futures = [
            executor.submit(self.run_window, WindowTask(path, start, frames))
            for start, frames in windows
        ]
        try:
            for future in futures:
                yield await asyncio.wrap_future(future)
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    def close(self) -> None:
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
