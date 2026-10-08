import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from math import gcd
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from hear.execution.native import NativeExecutor
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode
from hear.utils.transcription_chunks import (
    adaptive_batch_size,
    append_shifted_result,
    finalize_combined_result,
)

FFMPEG_FALLBACK_TIMEOUT_SECONDS = 1800
FFMPEG_FALLBACK_PARTS = 8
FFMPEG_FALLBACK_PART_SECONDS = 600
# A one-second frame louder than this holds sound; a file with at most
# SILENT_MAX_AUDIBLE_SECONDS such frames (a click, a pop) is silent.
AUDIBLE_DBFS = -50.0
SILENT_MAX_AUDIBLE_SECONDS = 1
SILENCE_FLOOR_DBFS = -120.0

_HALLUCINATION_ONLY = {
    "thank you",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
}


class TranscriptionModel(Protocol):
    async def transcribe(self, audio_bytes: bytes, batch_size: int) -> dict: ...

    async def transcribe_window(
        self,
        samples,
        batch_size: int,
        language: str,
        segments: list[tuple[float, float]] | None = None,
    ) -> dict: ...


class TranscriptionProgressSink(Protocol):
    async def publish(self, progress: float) -> None: ...


class TranscriptionResultPolicy:
    def __init__(self, min_avg_logprob: float = -0.75) -> None:
        self._min_avg_logprob = float(min_avg_logprob)

    @staticmethod
    def _normalized_text(value: str) -> str:
        return re.sub("[^a-z0-9 ]+", "", value.lower()).strip()

    def _credible_segments(self, result: dict, *, short_utterance: bool) -> list[dict]:
        segments = list(result.get("segments") or [])
        credible = [
            segment
            for segment in segments
            if "avg_logprob" not in segment
            or float(segment.get("avg_logprob", -99.0)) >= self._min_avg_logprob
        ]
        if not credible or short_utterance:
            return credible
        text = " ".join(str(segment.get("text") or "").strip() for segment in credible)
        speech_seconds = sum(
            max(0.0, float(segment.get("end", 0)) - float(segment.get("start", 0)))
            for segment in credible
        )
        audio_seconds = float(result.get("audio_duration") or 0.0)
        if (
            TranscriptionResultPolicy._normalized_text(text) in _HALLUCINATION_ONLY
            and audio_seconds >= 5.0
            and (speech_seconds <= 3.0)
        ):
            return []
        return credible


@dataclass
class _WindowRun:
    """What one file's windows share, across a retry from the ffmpeg-decoded WAV."""

    combined: dict
    state: dict
    batch_size: int
    language: str | None
    performance: dict
    progress: TranscriptionProgressSink | None


class TranscriptionService:
    def __init__(
        self,
        model_client: TranscriptionModel,
        chunk_seconds: int = 60,
        batch_size: int = 36,
        long_audio_batch_size: int = 4,
        native: NativeExecutor | None = None,
        min_avg_logprob: float = -0.75,
        vad_pool=None,
    ):
        if not 1 <= chunk_seconds <= 600 or batch_size < 1 or long_audio_batch_size < 1:
            raise ValueError("invalid_transcription_window_policy")
        self._model_client = model_client
        self._chunk_seconds = chunk_seconds
        self._batch_size = batch_size
        self._long_audio_batch_size = long_audio_batch_size
        self._native = native
        self._policy = TranscriptionResultPolicy(min_avg_logprob)
        self._vad_pool = vad_pool

    @staticmethod
    def _libsndfile_can_read(path: str) -> bool:
        """Header probe only (about a millisecond); the fast path stays libsndfile."""
        try:
            sf.info(path)
        except (RuntimeError, OSError):
            return False
        return True

    @classmethod
    def _decode_with_ffmpeg(cls, path: str) -> str:
        """Decode what libsndfile cannot open to the model's own 16 kHz mono float WAV.

        ffmpeg resyncs past leading junk (legacy padded uploads) and reads AAC/M4A
        and other containers. A single ffmpeg decodes about 5 minutes of MP3 per
        second, so long files are cut into ranges decoded in parallel and joined in
        order; the job then reads the WAV exactly like any readable source.
        """
        target = f"{path}.decoded.wav"
        duration = cls._probe_duration(path)
        parts = max(1, min(FFMPEG_FALLBACK_PARTS, int(duration // FFMPEG_FALLBACK_PART_SECONDS)))
        bounds = [duration * index / parts for index in range(parts)]
        raws = [f"{path}.part{index:02d}.f32" for index in range(parts)]
        try:
            with ThreadPoolExecutor(max_workers=parts) as pool:
                list(
                    pool.map(
                        lambda index: cls._decode_range(
                            path,
                            raws[index],
                            bounds[index],
                            None if index == parts - 1 else bounds[index + 1] - bounds[index],
                        ),
                        range(parts),
                    )
                )
            with sf.SoundFile(
                target, "w", samplerate=16000, channels=1, subtype="FLOAT", format="RF64"
            ) as out:
                for raw in raws:
                    with open(raw, "rb") as stream:
                        while block := stream.read(1 << 24):
                            out.write(np.frombuffer(block, dtype=np.float32))
        finally:
            for raw in raws:
                Path(raw).unlink(missing_ok=True)
        if sf.info(target).frames == 0:
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "source_audio_undecodable")
        return target

    @staticmethod
    def _probe_duration(path: str) -> float:
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, check=True, timeout=60,
            )
            return max(0.0, float(probe.stdout.strip() or 0))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as exc:
            raise CleanExecutionError(
                ErrorCode.INVALID_AUDIO, "source_audio_undecodable"
            ) from exc

    @staticmethod
    def _decode_range(path: str, target: str, start: float, length: float | None) -> None:
        try:
            subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-y",
                 *(["-ss", f"{start:.6f}"] if start > 0 else []), "-i", path,
                 *(["-t", f"{length:.6f}"] if length is not None else []),
                 "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", target],
                capture_output=True, check=True, timeout=FFMPEG_FALLBACK_TIMEOUT_SECONDS,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise CleanExecutionError(
                ErrorCode.INVALID_AUDIO, "source_audio_undecodable"
            ) from exc

    @staticmethod
    def _read_window(source, frames: int):
        samples = source.read(frames, dtype="float32", always_2d=True).mean(axis=1)
        if source.samplerate != 16000:
            divisor = gcd(source.samplerate, 16000)
            samples = resample_poly(samples, 16000 // divisor, source.samplerate // divisor)
        return samples

    async def transcribe_file(
        self,
        path: str,
        *,
        job_id: str | None = None,
        run_id: str | None = None,
        track_id: str | None = None,
        short_utterance: bool = False,
        language: str | None = None,
        progress: TranscriptionProgressSink | None = None,
    ) -> dict:
        started = time.perf_counter()
        performance = {
            "decode_resample_seconds": 0.0,
            "asr_and_alignment_seconds": 0.0,
            "model_window_seconds": 0.0,
            "executor_wait_seconds": 0.0,
            "windows": 0,
        }
        decoded = False
        if not self._libsndfile_can_read(path):
            # Rare inputs only (padded legacy MP3s, AAC/M4A); readable files never get here.
            fallback_started = time.perf_counter()
            path = await self._ffmpeg_decode(path)
            decoded = True
            performance["ffmpeg_fallback_seconds"] = time.perf_counter() - fallback_started
        with sf.SoundFile(path) as source:
            duration = source.frames / source.samplerate
        batch_size = adaptive_batch_size(duration, self._batch_size, self._long_audio_batch_size)
        combined = {"segments": [], "audio_duration": duration, "language": language or "en"}
        state = {"decoded_seconds": 0.0, "audible_seconds": 0, "loudest_dbfs": SILENCE_FLOOR_DBFS}
        windows = _WindowRun(combined, state, batch_size, language, performance, progress)
        try:
            await self._transcribe_windows(path, 0.0, windows)
        except sf.SoundFileRuntimeError:
            if decoded:
                raise
            # Damaged frames part-way through: libsndfile gives up where ffmpeg resyncs.
            # Keep the windows already transcribed and read the rest from ffmpeg's WAV.
            fallback_started = time.perf_counter()
            path = await self._ffmpeg_decode(path)
            performance["ffmpeg_retry_seconds"] = time.perf_counter() - fallback_started
            performance["ffmpeg_retry_from_seconds"] = state["decoded_seconds"]
            await self._transcribe_windows(path, state["decoded_seconds"], windows)
        decoded_seconds = state["decoded_seconds"]
        if decoded_seconds < duration - 1.0:
            # MP3 frame counts come from the header; report the audio that actually decoded.
            performance["header_overstated_seconds"] = duration - decoded_seconds
            duration = combined["audio_duration"] = decoded_seconds
        result = self._process_result(
            finalize_combined_result(combined), language=language, short_utterance=short_utterance
        )
        # "No speech" and "silent" differ: music beds and intros are audible without words.
        result["silent"] = result["no_speech"] and state["audible_seconds"] <= SILENT_MAX_AUDIBLE_SECONDS
        result["audio_level"] = {
            "audible_seconds": state["audible_seconds"],
            "loudest_second_dbfs": round(state["loudest_dbfs"], 1),
        }
        result["audio_duration"] = duration
        result["performance"] = {
            **{k: round(v, 6) for k, v in performance.items()},
            "total_seconds": round(time.perf_counter() - started, 6),
            "batch_size": batch_size,
            "window_seconds": self._chunk_seconds,
            "alignment_enabled": True,
        }
        return result

    async def _ffmpeg_decode(self, path: str) -> str:
        if self._native is not None:
            return await self._native.run(self._decode_with_ffmpeg, path)
        return await NativeExecutor.run_blocking_to_completion(
            partial(self._decode_with_ffmpeg, path)
        )

    async def _transcribe_windows(self, path: str, start_seconds: float, run: _WindowRun) -> None:
        """Transcribe `path` from `start_seconds` on, appending to the run's transcript."""
        with sf.SoundFile(path) as source:
            rate, total_frames = source.samplerate, source.frames
            frames = rate * self._chunk_seconds
            first = min(total_frames, int(round(start_seconds * rate)))
            if self._vad_pool is None:
                source.seek(first)
                while source.tell() < total_frames:
                    offset = source.tell() / rate
                    window_started = time.perf_counter()
                    operation = partial(self._read_window, source, frames)
                    samples = (
                        await self._native.run(self._read_window, source, frames)
                        if self._native is not None
                        else await NativeExecutor.run_blocking_to_completion(operation)
                    )
                    run.performance["decode_resample_seconds"] += time.perf_counter() - window_started
                    if samples.size == 0:
                        # The header overstates the length (truncated MP3): the audio has ended.
                        break
                    result = await self._transcribe_window(samples, run.batch_size, run.language)
                    self._append(run, result, samples, offset)
                    if run.progress is not None:
                        await run.progress.publish(source.tell() / total_frames * 100.0)
                return
        # Windows are read and VAD-segmented in worker processes ahead of the GPU.
        windows = [(start, min(frames, total_frames - start)) for start in range(first, total_frames, frames)]
        async for window in self._vad_pool.stream(path, windows):
            run.performance["decode_resample_seconds"] += window.read_seconds
            run.performance["vad_seconds"] = run.performance.get("vad_seconds", 0.0) + window.vad_seconds
            if window.samples.size == 0:
                # Planned from an overstated header; the audio ended before this window.
                continue
            result = await self._transcribe_window(
                window.samples, run.batch_size, run.language, segments=window.segments
            )
            self._append(run, result, window.samples, window.start_frame / rate)
            if run.progress is not None:
                done = window.start_frame + min(frames, total_frames - window.start_frame)
                await run.progress.publish(done / total_frames * 100.0)

    def _append(self, run: _WindowRun, result: dict, samples, offset: float) -> None:
        self._account(run.performance, result)
        append_shifted_result(run.combined, result, offset_seconds=offset)
        run.state["decoded_seconds"] = offset + samples.size / 16000
        audible, loudest = self._measure_level(samples)
        run.state["audible_seconds"] += audible
        run.state["loudest_dbfs"] = max(run.state["loudest_dbfs"], loudest)

    @staticmethod
    def _measure_level(samples) -> tuple[int, float]:
        """Count the 16 kHz window's audible one-second frames and its loudest frame."""
        whole = samples.size // 16000
        frames = [samples[: whole * 16000].reshape(whole, 16000)] if whole else []
        if samples.size - whole * 16000 >= 4000:
            frames.append(samples[whole * 16000 :].reshape(1, -1))
        if not frames:
            return 0, SILENCE_FLOOR_DBFS
        levels = np.concatenate(
            [np.sqrt(np.mean(np.square(block, dtype=np.float64), axis=1)) for block in frames]
        )
        dbfs = 20.0 * np.log10(np.maximum(levels, 1e-10))
        return int(np.count_nonzero(dbfs > AUDIBLE_DBFS)), float(max(dbfs.max(), SILENCE_FLOOR_DBFS))

    async def _transcribe_window(self, samples, batch_size: int, language: str | None, *, segments=None) -> dict:
        started = time.perf_counter()
        if segments is None:
            result = await self._model_client.transcribe_window(samples, batch_size, language or "en")
        else:
            result = await self._model_client.transcribe_window(
                samples, batch_size, language or "en", segments=segments
            )
        if not isinstance(result, dict) or not isinstance(result.get("segments"), list):
            raise RuntimeError("invalid_transcription_window_result")
        result.setdefault("_runtime_timing", {})["asr_and_alignment_seconds"] = (
            time.perf_counter() - started
        )
        return result

    @staticmethod
    def _account(performance: dict, result: dict) -> None:
        timing = result.get("_runtime_timing", {})
        performance["asr_and_alignment_seconds"] += timing.get("asr_and_alignment_seconds", 0.0)
        for name in ("model_window_seconds", "executor_wait_seconds"):
            performance[name] += timing.get(name, 0.0)
        performance["windows"] += 1

    async def transcribe(
        self,
        audio_bytes: bytes,
        *,
        job_id: str | None = None,
        run_id: str | None = None,
        track_id: str | None = None,
        short_utterance: bool = False,
        language: str | None = None,
    ) -> dict:
        if len(audio_bytes) > 16 * 1024 * 1024:
            raise ValueError("reference_audio_too_large")
        result = await self._model_client.transcribe(audio_bytes, self._batch_size)
        if not isinstance(result, dict) or not isinstance(result.get("segments"), list):
            raise RuntimeError("invalid_transcription_result")
        return self._process_result(result, language=language, short_utterance=short_utterance)

    def _process_result(
        self, result: dict, language: str | None = None, short_utterance: bool = False
    ) -> dict:
        _silent: dict = {
            "transcript": "",
            "segments": [],
            "language": None,
            "language_probability": 0.0,
            "duration": 0.0,
            "confidence": 0.0,
            "silent": True,
            "no_speech": True,
        }
        if not result:
            return _silent
        segments_list = self._policy._credible_segments(result, short_utterance=short_utterance)
        detected_language = result.get("language", language or "en")
        segments: list[dict] = []
        full_text_parts = []
        total_conf = 0.0
        word_count = 0
        scored_words = 0
        for seg in segments_list:
            text = seg.get("text", "").strip()
            if not text:
                continue
            words = []
            raw_words = [w for w in (seg.get("words") or []) if str(w.get("word", "")).strip()]
            for index, w in enumerate(raw_words):
                start = float(w["start"])
                end = float(w["end"])
                if end <= start:
                    # The aligner emits a zero-length stamp for some function words; give
                    # them a 20 ms span bounded by the next word so timelines stay monotonic.
                    limit = float(raw_words[index + 1]["start"]) if index + 1 < len(raw_words) else start + 0.02
                    end = max(start, min(start + 0.02, limit))
                score = w.get("score")
                # The forced aligner places words but has no per-word confidence; the
                # patch emits a placeholder 1.0 which must not masquerade as a measurement.
                real_score = score is not None and float(score) < 1.0
                words.append(
                    {
                        "word": w["word"],
                        "start": start,
                        "end": end,
                        "prob": float(score) if real_score else None,
                    }
                )
                if real_score:
                    total_conf += float(score)
                    scored_words += 1
                word_count += 1
            if not words:
                avg_logprob = seg.get("avg_logprob", -1.0)
                prob = max(float(avg_logprob) + 1.0, 0.1)
                words.append(
                    {
                        "word": text,
                        "start": seg.get("start", 0),
                        "end": seg.get("end", 0),
                        "prob": prob,
                    }
                )
                total_conf += prob
                scored_words += 1
                word_count += 1
            segments.append(
                {
                    "id": seg.get("id", len(segments)),
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": text,
                    "words": words,
                }
            )
            full_text_parts.append(text)
        if not full_text_parts:
            return _silent
        transcript = " ".join(full_text_parts)
        confidence = round(total_conf / scored_words, 4) if scored_words else None
        duration = segments[-1]["end"] if segments else 0.0
        return {
            "transcript": transcript,
            "segments": segments,
            "word_segments": result.get("word_segments", []),
            "language": detected_language,
            "language_probability": 1.0,
            "duration": duration,
            "confidence": confidence,
            "word_confidence_available": scored_words > 0,
            "silent": False,
            "no_speech": False,
        }
