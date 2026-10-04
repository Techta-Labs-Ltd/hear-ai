from __future__ import annotations
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from hear.runtime.cleaner.pool import WorkerPool
from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.services.magic_clean.contracts import CleanExecutionError, CleanPlan, ErrorCode
from hear.services.magic_clean.loudness import BlockMeasurement, KWeightedMeter
from hear.services.magic_clean.mastering import AudioMasteringService
from hear.services.magic_clean.profile_dsp import ProfileDspService
from hear.services.magic_clean.quality import AudioQualityGate, BlockEnergies

RATE = 48000
HANDLE_FRAMES = 6 * RATE
CROSSFADE_FRAMES = RATE // 10
COPY_FRAMES = 1 << 20


@dataclass(frozen=True)
class ChunkEngineConfig:
    """Everything a worker process needs to build its own engines."""

    config_path: str
    model_directory: str
    device: str
    sound_cleanup_bundle: str | None
    sound_cleanup_sha256: str
    budget: tuple[int, int, int]


@dataclass(frozen=True)
class ChunkTask:
    index: int
    start: int
    end: int
    left: int
    right: int
    prepared: str
    workspace: str
    deadline_epoch: float
    options: dict
    plan_json: str
    finishing: bool
    channels: int


@dataclass
class ChunkResult:
    index: int
    start: int
    end: int
    left: int
    output: str
    energies: BlockEnergies
    voiced_frames: int | None
    lost_voiced_frames: int | None
    input_measurement: BlockMeasurement
    output_measurement: BlockMeasurement
    timings: dict[str, float] = field(default_factory=dict)


class ChunkEngines:
    """The per-process engines a chunk uses; built once per worker or borrowed inline."""

    def __init__(self, dsp: ProfileDspService, engine, analyser) -> None:
        self.dsp = dsp
        self.engine = engine
        self.analyser = analyser


class ChunkPlanner:
    @staticmethod
    def plan(frames: int, chunk_frames: int) -> list[tuple[int, int, int, int]]:
        """(start, end, left, right) per chunk; chunks are near-equal, never tiny."""
        count = max(1, round(frames / chunk_frames))
        bounds = [round(index * frames / count) for index in range(count + 1)]
        chunks = []
        for start, end in zip(bounds[:-1], bounds[1:], strict=True):
            chunks.append((start, end, max(0, start - HANDLE_FRAMES), min(frames, end + HANDLE_FRAMES)))
        return chunks


class ChunkWorker:
    """Runs one chunk with the engines installed in this process."""

    engines: ChunkEngines | None = None

    @classmethod
    def use(cls, engines: ChunkEngines | None) -> None:
        cls.engines = engines

    @classmethod
    def clean(cls, task: ChunkTask) -> ChunkResult:
        engines = cls.engines
        if engines is None:
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "chunk worker has no engines")
        workspace = Path(task.workspace) / f"chunk-{task.index:04d}"
        guard = WorkerPool.guard(workspace, task.deadline_epoch)
        plan = CleanPlan.model_validate_json(task.plan_json)
        timings: dict[str, float] = {}
        clock = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal clock
            now = time.perf_counter()
            timings[name] = now - clock
            clock = now

        chunk_in = workspace / "in.wav"
        cls.copy_range(Path(task.prepared), chunk_in, task.left, task.right, guard)
        crop = slice(task.start - task.left, task.end - task.left)
        lap("read_seconds")
        model_input = chunk_in
        filters = engines.dsp.preparation_filters(task.options)
        if filters:
            model_input = workspace / "conditioned.wav"
            engines.dsp.render(chunk_in, model_input, filters, guard)
        lap("preparation_seconds")
        processed = workspace / "processed.wav"
        session = engines.engine.open_session(plan, guard)
        try:
            session.process(model_input, processed, plan, guard)
        finally:
            session.close()
        AudioMasteringService.scan(
            processed, guard, rate=RATE, channels=task.channels, frames=task.right - task.left
        )
        lap("denoising_seconds")
        input_crop = workspace / "input-crop.wav"
        processed_crop = workspace / "processed-crop.wav"
        cls.copy_range(model_input, input_crop, crop.start, crop.stop, guard)
        cls.copy_range(processed, processed_crop, crop.start, crop.stop, guard)
        energies = AudioQualityGate.block_energies(input_crop, processed_crop, guard)
        voiced = lost = None
        if engines.analyser is not None:
            before = engines.analyser.speech_probability(input_crop, guard)
            after = engines.analyser.speech_probability(processed_crop, guard)
            anchors = before >= 0.8
            voiced = int(anchors.sum())
            lost = int((anchors & (after < 0.1)).sum())
        lap("quality_checks_seconds")
        final = processed
        finishing = engines.dsp.finishing_filters(task.options) if task.finishing else []
        if finishing:
            final = workspace / "finished.wav"
            engines.dsp.render(processed, final, finishing, guard)
            AudioMasteringService.scan(
                final, guard, rate=RATE, channels=task.channels, frames=task.right - task.left
            )
        lap("finishing_seconds")
        with sf.SoundFile(input_crop) as stream:
            input_measurement = KWeightedMeter.measure(stream.read(dtype="float32", always_2d=True), RATE)
        with sf.SoundFile(final) as stream:
            stream.seek(crop.start)
            output_measurement = KWeightedMeter.measure(
                stream.read(crop.stop - crop.start, dtype="float32", always_2d=True), RATE
            )
        lap("metering_seconds")
        for path in (chunk_in, input_crop, processed_crop):
            path.unlink(missing_ok=True)
        if model_input != chunk_in:
            model_input.unlink(missing_ok=True)
        if final != processed:
            processed.unlink(missing_ok=True)
        guard.check()
        return ChunkResult(
            task.index,
            task.start,
            task.end,
            task.left,
            str(final),
            energies,
            voiced,
            lost,
            input_measurement,
            output_measurement,
            timings,
        )


    @staticmethod
    def copy_range(source: Path, target: Path, start: int, end: int, guard: ResourceGuard) -> None:
        with sf.SoundFile(source) as stream, sf.SoundFile(
            target, "w", samplerate=stream.samplerate, channels=stream.channels, subtype="FLOAT"
        ) as out:
            stream.seek(start)
            remaining = end - start
            while remaining > 0:
                guard.check()
                data = stream.read(min(COPY_FRAMES, remaining), dtype="float32", always_2d=True)
                if not len(data):
                    raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "chunk source is truncated")
                if not np.isfinite(data).all():
                    raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "invalid source samples")
                out.write(data)
                remaining -= len(data)


class ParallelCleaner:
    """Runs the chunk plan and stitches the result for one job."""

    def __init__(self, workers: int, chunk_seconds: int) -> None:
        self.workers = max(1, workers)
        self.chunk_frames = max(RATE * 60, chunk_seconds * RATE)

    def run(
        self,
        tasks_template: dict,
        frames: int,
        guard: ResourceGuard,
        *,
        initializer: Callable[[ChunkEngineConfig], None],
        config_factory: Callable[[], ChunkEngineConfig],
        inline_engines: ChunkEngines,
    ) -> list[ChunkResult]:
        chunks = ChunkPlanner.plan(frames, self.chunk_frames)
        deadline = guard.wall_deadline.timestamp() if guard.wall_deadline else time.time() + 86400
        tasks = [
            ChunkTask(index, start, end, left, right, deadline_epoch=deadline, **tasks_template)
            for index, (start, end, left, right) in enumerate(chunks)
        ]
        if len(tasks) == 1 or self.workers == 1:
            ChunkWorker.use(inline_engines)
            try:
                return [ChunkWorker.clean(task) for task in tasks]
            finally:
                ChunkWorker.use(None)
        with WorkerPool(
            min(self.workers, len(tasks)), initializer=initializer, initargs=(config_factory(),)
        ) as pool:
            return pool.map(ChunkWorker.clean, tasks, guard)

    @staticmethod
    def stitch(results: list[ChunkResult], target: Path, channels: int, guard: ResourceGuard) -> int:
        """Write the handled chunk outputs as one file with crossfades at the joins."""
        ordered = sorted(results, key=lambda item: item.index)
        half = CROSSFADE_FRAMES // 2
        written = 0
        with sf.SoundFile(
            target, "w", samplerate=RATE, channels=channels, format="RF64", subtype="FLOAT"
        ) as out:
            for index, item in enumerate(ordered):
                with sf.SoundFile(item.output) as current:
                    lo = item.start if index == 0 else item.start + half
                    hi = item.end if index == len(ordered) - 1 else item.end - half
                    if index > 0:
                        previous = ordered[index - 1]
                        with sf.SoundFile(previous.output) as before:
                            before.seek(item.start - half - previous.left)
                            tail = before.read(2 * half, dtype="float32", always_2d=True)
                        current.seek(item.start - half - item.left)
                        head = current.read(2 * half, dtype="float32", always_2d=True)
                        if len(tail) != 2 * half or len(head) != 2 * half:
                            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "chunk handles too short")
                        ramp = np.linspace(0.0, 1.0, 2 * half, dtype=np.float32)[:, None]
                        out.write(tail * (1 - ramp) + head * ramp)
                        written += 2 * half
                    current.seek(lo - item.left)
                    remaining = hi - lo
                    while remaining > 0:
                        guard.check()
                        data = current.read(min(COPY_FRAMES, remaining), dtype="float32", always_2d=True)
                        if not len(data):
                            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "chunk output is truncated")
                        out.write(data)
                        remaining -= len(data)
                        written += len(data)
        return written

    @staticmethod
    def speech_report(results: list[ChunkResult]) -> dict:
        """Fail when confidently voiced frames vanish; energy checks cannot see this."""
        if any(item.voiced_frames is None for item in results):
            return {"status": "analyser_not_provisioned"}
        anchor_count = sum(item.voiced_frames or 0 for item in results)
        lost_count = sum(item.lost_voiced_frames or 0 for item in results)
        allowed = max(2, anchor_count // 100)
        if lost_count > allowed:
            raise CleanExecutionError(
                ErrorCode.INVALID_AUDIO,
                f"speech_activity_lost:{lost_count}_of_{anchor_count}_voiced_frames",
            )
        return {
            "status": "review_required" if lost_count else "passed",
            "voiced_frames": anchor_count,
            "lost_voiced_frames": lost_count,
            "allowed_lost_frames": allowed,
            "step_frames": 1536,
        }

    @staticmethod
    def default_workers() -> int:
        return max(1, min(8, (os.cpu_count() or 2) // 2))

    @staticmethod
    def chunk_timings(results: list[ChunkResult]) -> dict[str, float]:
        """Slowest chunk per stage: the parallel wall time each stage contributed."""
        names = sorted({name for item in results for name in item.timings})
        return {
            f"chunk_{name}": round(max(item.timings.get(name, 0.0) for item in results), 6)
            for name in names
        }

