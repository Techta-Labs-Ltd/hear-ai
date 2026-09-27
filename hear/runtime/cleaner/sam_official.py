from __future__ import annotations

import importlib
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from hear.runtime.cleaner.path_cleanup import AttemptPathCleanup, OwnedFileIdentity
from hear.runtime.cleaner.resampling import AudioResampler
from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.services.magic_clean.contracts import CleanExecutionError, CleanPlan, ErrorCode


class SamOfficialPipeline:
    SAMPLE_RATE = 48_000
    CHUNK_SECONDS = 75
    OVERLAP_SECONDS = 5
    AMBIENT_RERANKING_CANDIDATES = 2
    EVENT_RERANKING_CANDIDATES = 1
    MAX_RESIDUAL_COLLAPSE_SECONDS = 2
    MIN_RESIDUAL_RMS_RATIO = 0.05
    MIN_TARGET_RMS = 10 ** (-55 / 20)
    MIN_TARGET_RMS_RATIO = 0.01
    TARGET_ANALYSIS_SECONDS = 1
    POLICY = "meta-sam-audio-clap-pe-long-context-v4"
    DUAL_MONO_MIN_CORRELATION = 0.98
    CHANNEL_POLICY = "validated-dual-mono-average-v1"

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.device = next(model.parameters()).device
        if processor.audio_sampling_rate != self.SAMPLE_RATE:
            raise ValueError("SAM Audio processor must use 48 kHz")

    @classmethod
    def preflight(cls, frames: int, sample_rate: int, guard: ResourceGuard) -> int:
        guard.check()
        if type(frames) is not int or frames <= 0 or not 8_000 <= sample_rate <= 96_000:
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "invalid SAM audio geometry")
        prepared_frames = AudioResampler.frame_count(frames, sample_rate, cls.SAMPLE_RATE)
        if prepared_frames > guard.budget.max_frames:
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED, "SAM input exceeds frame budget"
            )
        chunk_frames = cls.CHUNK_SECONDS * cls.SAMPLE_RATE

        scratch = min(prepared_frames, chunk_frames) * 4 * 4
        guard.reserve_scratch(scratch)
        return scratch

    @classmethod
    def prepare_input(
        cls,
        source: Path,
        destination: Path,
        *,
        sample_rate: int,
        channels: int,
        frames: int,
        channel_correlation: float | None,
        plan: CleanPlan,
        guard: ResourceGuard,
    ) -> Path:
        guard.check()
        workspace = guard.workspace.resolve()
        if source.is_symlink() or not source.resolve().is_relative_to(workspace):
            raise CleanExecutionError(ErrorCode.SOURCE_MISMATCH, "SAM input outside workspace")
        if channels == 1:
            if plan.channel_policy != "mono":
                raise CleanExecutionError(
                    ErrorCode.INVALID_AUDIO, "mono SAM Audio input has a stereo policy"
                )
            return source
        if (
            channels != 2
            or plan.channel_policy != "validated_dual_mono"
            or not plan.mono_acknowledged
            or channel_correlation is None
            or channel_correlation < cls.DUAL_MONO_MIN_CORRELATION
        ):
            raise CleanExecutionError(
                ErrorCode.INVALID_AUDIO, "SAM Audio requires mono or correlated dual-mono input"
            )
        if destination.exists() or destination.is_symlink():
            raise CleanExecutionError(ErrorCode.ARTIFACT_CONFLICT, "mono input already exists")
        if not destination.resolve().is_relative_to(workspace):
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "mono input outside workspace")
        try:
            with sf.SoundFile(source) as audio:
                if (audio.samplerate, audio.channels, audio.frames) != (
                    sample_rate,
                    channels,
                    frames,
                ):
                    raise CleanExecutionError(
                        ErrorCode.SOURCE_MISMATCH, "stereo input changed after inspection"
                    )
                with sf.SoundFile(
                    destination,
                    "w",
                    samplerate=sample_rate,
                    channels=1,
                    format="RF64",
                    subtype="FLOAT",
                ) as mono:
                    while True:
                        guard.check()
                        block = audio.read(32768, dtype="float32", always_2d=True)
                        if not len(block):
                            break
                        if block.shape[1] != 2 or not np.isfinite(block).all():
                            raise CleanExecutionError(
                                ErrorCode.INVALID_AUDIO, "invalid dual-mono samples"
                            )
                        mono.write(block.mean(axis=1, dtype=np.float64).astype(np.float32))
            guard.check_scratch()
            return destination
        except CleanExecutionError:
            raise
        except Exception:
            raise CleanExecutionError(
                ErrorCode.INVALID_AUDIO, "dual-mono conversion failed"
            ) from None

    def separate_plan(
        self,
        source: Path,
        destination: Path,
        *,
        plan: CleanPlan,
        expected_runtime,
        guard: ResourceGuard,
    ) -> str:
        guard.check()
        if plan.profile != "sam_audio" or plan.runtime != expected_runtime:
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "SAM plan identity mismatch")
        if plan.channel_policy not in ("mono", "validated_dual_mono"):
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "SAM requires mono audio")
        if destination.exists() or destination.is_symlink():
            raise CleanExecutionError(ErrorCode.ARTIFACT_CONFLICT, "SAM output already exists")
        if source.is_symlink() or not source.resolve().is_relative_to(guard.workspace.resolve()):
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "SAM input outside workspace")
        if not destination.resolve().is_relative_to(guard.workspace.resolve()):
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "SAM output outside workspace")

        description = (plan.prompt_text or "").strip().lower()
        if not description or len(description.encode("utf-8")) > 512:
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "invalid SAM sound description")
        stream = "target" if plan.prompt_action == "isolate" else "residual"
        published: OwnedFileIdentity | None = None
        try:
            with sf.SoundFile(source) as audio:
                rate, frames, channels = audio.samplerate, audio.frames, audio.channels
                if channels != 1 or not 8_000 <= rate <= 96_000 or frames <= 0:
                    raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "SAM requires mono audio")
            with tempfile.TemporaryDirectory(
                prefix="sam-official-", dir=guard.workspace
            ) as directory:
                directory_path = Path(directory)
                prepared = source
                if rate != self.SAMPLE_RATE:
                    prepared = directory_path / "prepared.wav"
                    AudioResampler(CancellableProcessRunner()).convert(
                        source, prepared, self.SAMPLE_RATE, guard
                    )
                separated = directory_path / "separated.wav"
                self._separate_48k(
                    prepared,
                    separated,
                    description=description,
                    stream=stream,
                    seed=plan.seed,
                    predict_spans=plan.prompt_mode == "event",
                    guard=guard,
                )
                if rate == self.SAMPLE_RATE:
                    os.link(separated, destination)
                else:
                    AudioResampler(CancellableProcessRunner()).convert(
                        separated, destination, rate, guard, exact_frames=frames
                    )
                published = AttemptPathCleanup.identity(destination)
            guard.check()
            return self.POLICY
        except BaseException as exc:
            if published is not None:
                AttemptPathCleanup.remove_if_owned(destination, published, exc)
            if isinstance(exc, CleanExecutionError):
                raise
            if isinstance(exc, FileExistsError):
                raise CleanExecutionError(
                    ErrorCode.ARTIFACT_CONFLICT, "SAM output already exists"
                ) from None
            if not isinstance(exc, Exception):
                raise
            raise CleanExecutionError(
                ErrorCode.PROCESS_FAILED,
                f"Meta SAM Audio inference failed ({type(exc).__name__})",
            ) from None

    def _separate_48k(
        self,
        source: Path,
        destination: Path,
        *,
        description: str,
        stream: str,
        seed: int,
        predict_spans: bool,
        guard: ResourceGuard,
    ) -> None:
        torch = importlib.import_module("torch")

        chunk_frames = self.CHUNK_SECONDS * self.SAMPLE_RATE
        overlap_frames = self.OVERLAP_SECONDS * self.SAMPLE_RATE
        stride = chunk_frames - overlap_frames
        pending: np.ndarray | None = None
        target_detected = stream != "target"
        reranking_candidates = (
            self.EVENT_RERANKING_CANDIDATES if predict_spans else self.AMBIENT_RERANKING_CANDIDATES
        )
        with sf.SoundFile(source) as audio:
            if audio.samplerate != self.SAMPLE_RATE or audio.channels != 1:
                raise CleanExecutionError(
                    ErrorCode.INVALID_AUDIO, "SAM preparation is not mono 48 kHz"
                )
            total_frames = audio.frames
            validate_stitched_residual = total_frames > chunk_frames
            staged = destination.with_suffix(".partial.wav")
            staged.unlink(missing_ok=True)
            try:
                with sf.SoundFile(
                    staged,
                    "w",
                    samplerate=self.SAMPLE_RATE,
                    channels=1,
                    format="RF64",
                    subtype="FLOAT",
                ) as output:
                    start = index = 0
                    while start < total_frames:
                        guard.check()
                        audio.seek(start)
                        values = audio.read(
                            min(chunk_frames, total_frames - start),
                            dtype="float32",
                            always_2d=False,
                        )
                        if values.size == 0:
                            raise CleanExecutionError(
                                ErrorCode.INVALID_AUDIO, "short SAM input read"
                            )
                        waveform = torch.from_numpy(np.array(values, copy=True)).unsqueeze(0)
                        batch = self.processor(audios=[waveform], descriptions=[description]).to(
                            self.device
                        )
                        span_on_device = False
                        try:
                            if predict_spans and self.device.type == "cuda":
                                self.model.span_predictor.to(
                                    device=self.device,
                                    dtype=torch.float32,
                                )
                                span_on_device = True
                            with torch.random.fork_rng(
                                devices=[self.device.index or 0]
                                if self.device.type == "cuda"
                                else []
                            ):
                                torch.manual_seed(seed)
                                if self.device.type == "cuda":
                                    torch.cuda.manual_seed_all(seed)
                                with (
                                    torch.inference_mode(),
                                    torch.autocast(device_type=self.device.type, enabled=False),
                                ):
                                    result = self.model.separate(
                                        batch,
                                        predict_spans=predict_spans,
                                        reranking_candidates=reranking_candidates,
                                    )
                        finally:
                            if span_on_device:
                                self.model.span_predictor.to(
                                    device="cpu",
                                    dtype=torch.float32,
                                )
                                torch.cuda.empty_cache()
                        output_list = getattr(result, stream)
                        if len(output_list) != 1:
                            raise CleanExecutionError(
                                ErrorCode.PROCESS_FAILED, "SAM batch size mismatch"
                            )
                        current = output_list[0].detach().float().cpu().numpy().reshape(-1)

                        codec_hop = self.processor.audio_hop_length
                        if (
                            current.size < values.size
                            or current.size - values.size >= codec_hop
                            or not np.isfinite(current[: values.size]).all()
                        ):
                            raise CleanExecutionError(
                                ErrorCode.PROCESS_FAILED,
                                "SAM output duration or samples invalid "
                                f"(chunk={index}, input={values.size}, output={current.size})",
                            )
                        current = current[: values.size]
                        if stream == "target" and self._target_detected(values, current):
                            target_detected = True
                        if stream == "residual" and validate_stitched_residual:
                            self._validate_residual(values, current)
                        if pending is None:
                            pending = current
                        else:
                            count = min(overlap_frames, pending.size, current.size)
                            if count:
                                output.write(pending[:-count])
                                ramp = np.linspace(0, 1, count, endpoint=True, dtype=np.float32)
                                output.write(pending[-count:] * (1 - ramp) + current[:count] * ramp)
                                pending = current[count:]
                            else:
                                output.write(pending)
                                pending = current
                        del batch, result, waveform
                        start += stride
                        index += 1
                    if pending is not None:
                        output.write(pending)
                if not target_detected:
                    raise CleanExecutionError(
                        ErrorCode.TARGET_NOT_DETECTED,
                        "SAM Audio target was not detected in the source",
                    )
                with sf.SoundFile(staged) as check:
                    if check.frames != total_frames or check.samplerate != self.SAMPLE_RATE:
                        raise CleanExecutionError(
                            ErrorCode.PROCESS_FAILED, "SAM output timeline mismatch"
                        )
                os.link(staged, destination)
            finally:
                staged.unlink(missing_ok=True)

    @classmethod
    def _target_detected(cls, source: np.ndarray, target: np.ndarray) -> bool:
        block_frames = cls.TARGET_ANALYSIS_SECONDS * cls.SAMPLE_RATE
        for start in range(0, source.size, block_frames):
            source_block = source[start : start + block_frames].astype(np.float64)
            target_block = target[start : start + block_frames].astype(np.float64)
            source_rms = float(np.sqrt(np.mean(source_block * source_block)))
            target_rms = float(np.sqrt(np.mean(target_block * target_block)))
            if (
                source_rms >= 1e-4
                and target_rms >= cls.MIN_TARGET_RMS
                and target_rms >= source_rms * cls.MIN_TARGET_RMS_RATIO
            ):
                return True
        return False

    @classmethod
    def _validate_residual(cls, source: np.ndarray, residual: np.ndarray) -> None:
        collapsed_frames = 0
        block_frames = cls.SAMPLE_RATE
        for start in range(0, source.size, block_frames):
            source_block = source[start : start + block_frames].astype(np.float64)
            residual_block = residual[start : start + block_frames].astype(np.float64)
            source_rms = float(np.sqrt(np.mean(source_block * source_block)))
            residual_rms = float(np.sqrt(np.mean(residual_block * residual_block)))
            if source_rms > 1e-4 and residual_rms < source_rms * cls.MIN_RESIDUAL_RMS_RATIO:
                collapsed_frames += source_block.size
                if collapsed_frames >= cls.MAX_RESIDUAL_COLLAPSE_SECONDS * cls.SAMPLE_RATE:
                    raise CleanExecutionError(
                        ErrorCode.INVALID_AUDIO,
                        "SAM residual collapsed active source audio",
                    )
            else:
                collapsed_frames = 0
