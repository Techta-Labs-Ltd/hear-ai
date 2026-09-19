from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import soundfile as sf
import torch

from hear.core.hear_temp import drop_temp_standalone, hear_temp_standalone_dir
from hear.services.magic_clean.models import ContentMode, StemLevels
from hear.services.magic_clean.pipeline import MagicCleanPipeline
from hear.services.magic_clean.processing.audio_io import AudioIO

TARGET_LUFS = -16.0
MAX_GAIN_DB = 6.0
MAX_ATTENUATION_DB = 12.0
# Leave a full decibel of codec headroom beneath the public -1 dBTP gate.
PRE_ENCODE_PEAK_LIMIT = 10 ** (-2.0 / 20.0)
DELIVERED_TRUE_PEAK_CEILING_DBTP = -1.0
CODEC_TRUE_PEAK_TARGET_DBTP = -1.25
MAX_CODEC_TRUE_PEAK_REENCODES = 2
MAX_CODEC_CORRECTION_DB = 6.0
SUPPORTED_MP3_SAMPLE_RATES = (
    8_000,
    11_025,
    12_000,
    16_000,
    22_050,
    24_000,
    32_000,
    44_100,
    48_000,
)
FFMPEG_TIMEOUT_SECONDS = 4 * 60 * 60


class MagicCleanProcessingCancelled(RuntimeError):
    """A cooperative cancellation request reached a safe processing boundary."""


@dataclass(frozen=True, slots=True)
class StreamingCleanResult:
    output_path: str
    input_duration_seconds: float
    output_duration_seconds: float
    peak_db: float
    integrated_lufs: float
    chunks_processed: int
    sample_rate: int
    delivery_sample_rate: int
    channels: int
    input_samples: int
    output_samples: int
    stage_times: dict[str, float] = field(default_factory=dict)


def clean_file_streaming(
    pipeline: MagicCleanPipeline,
    input_path: str,
    output_path: str,
    *,
    device: torch.device,
    mode: ContentMode = ContentMode.SPEECH,
    levels: StemLevels | None = None,
    cut_silence: bool = False,
    chunk_seconds: float = 60.0,
    overlap_seconds: float = 2.0,
    bitrate_kbps: int = 96,
    progress: Callable[[int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
    validation_reference_path: str | None = None,
) -> StreamingCleanResult:
    """Clean one file with context/core stitching and one global master."""
    work_dir = hear_temp_standalone_dir("magic_clean_stream")
    decoded_path = os.path.join(work_dir, "decoded.wav")
    processed_path = os.path.join(work_dir, "processed.wav")
    silence_path = os.path.join(work_dir, "silence-edited.wav")
    temporary_output = output_path + ".partial.mp3"
    temporary_reference = (
        validation_reference_path + ".partial"
        if validation_reference_path is not None
        else None
    )
    stage_times: dict[str, float] = {}
    completed = False
    try:
        _raise_if_cancelled(cancel_event)
        started = time.perf_counter()
        _decode_lossless(input_path, decoded_path)
        _raise_if_cancelled(cancel_event)
        stage_times["decode"] = _elapsed(started)

        with sf.SoundFile(decoded_path) as decoded:
            sample_rate = int(decoded.samplerate)
            channels = int(decoded.channels)
            input_samples = int(len(decoded))
        if sample_rate <= 0 or channels not in {1, 2} or input_samples <= 0:
            raise ValueError("Magic Clean requires non-empty mono or stereo source audio")

        chunk_samples, context_samples = pipeline._chunk_sizes(
            sample_rate,
            chunk_seconds,
            overlap_seconds,
        )
        started = time.perf_counter()
        chunks = _process_decoded_file(
            pipeline,
            decoded_path,
            processed_path,
            device=device,
            mode=mode,
            levels=levels,
            chunk_samples=chunk_samples,
            context_samples=context_samples,
            progress=progress,
            cancel_event=cancel_event,
        )
        _raise_if_cancelled(cancel_event)
        stage_times["enhance_and_stitch"] = _elapsed(started)

        with sf.SoundFile(processed_path) as processed:
            output_samples = int(len(processed))
            processed_channels = int(processed.channels)
            processed_rate = int(processed.samplerate)
        if processed_rate != sample_rate or processed_channels != channels:
            raise RuntimeError("Magic Clean changed sample rate or channel layout")
        if output_samples != input_samples:
            raise RuntimeError(
                "Magic Clean context stitching changed the timeline: "
                f"expected={input_samples}, actual={output_samples}"
            )

        if cut_silence:
            _raise_if_cancelled(cancel_event)
            started = time.perf_counter()
            output_samples = _apply_global_silence_edit(
                pipeline,
                decoded_path,
                processed_path,
                silence_path,
                device=device,
                mode=mode,
            )
            _raise_if_cancelled(cancel_event)
            os.replace(silence_path, processed_path)
            stage_times["silence_edit"] = _elapsed(started)

        _raise_if_cancelled(cancel_event)
        started = time.perf_counter()
        measurements = _measure_loudness(processed_path)
        _raise_if_cancelled(cancel_event)
        stage_times["measure"] = _elapsed(started)

        started = time.perf_counter()
        delivery_sample_rate = _mp3_delivery_sample_rate(sample_rate)
        delivered_measurements = _encode_with_true_peak_guard(
            processed_path,
            temporary_output,
            measurements,
            sample_rate=delivery_sample_rate,
            channels=channels,
            bitrate_kbps=bitrate_kbps,
        )
        _raise_if_cancelled(cancel_event)
        os.replace(temporary_output, output_path)
        stage_times["master_and_encode"] = _elapsed(started)

        if validation_reference_path is not None and temporary_reference is not None:
            shutil.copyfile(processed_path, temporary_reference)
            os.replace(temporary_reference, validation_reference_path)

        source_probe = _probe(input_path)
        output_probe = _probe(output_path)
        integrated_lufs = _finite_float(
            delivered_measurements.get("input_i"),
            default=-99.0,
        )
        delivered_peak = _finite_float(
            delivered_measurements.get("input_tp"),
            default=-99.0,
        )
        result = StreamingCleanResult(
            output_path=output_path,
            input_duration_seconds=source_probe["duration"],
            output_duration_seconds=output_probe["duration"],
            peak_db=delivered_peak,
            integrated_lufs=integrated_lufs,
            chunks_processed=chunks,
            sample_rate=sample_rate,
            delivery_sample_rate=delivery_sample_rate,
            channels=channels,
            input_samples=input_samples,
            output_samples=output_samples,
            stage_times=stage_times,
        )
        completed = True
        return result
    finally:
        for path in (
            temporary_output,
            decoded_path,
            processed_path,
            silence_path,
        ):
            try:
                os.unlink(path)
            except OSError:
                pass
        if temporary_reference is not None:
            try:
                os.unlink(temporary_reference)
            except OSError:
                pass
        if not completed and validation_reference_path is not None:
            try:
                os.unlink(validation_reference_path)
            except OSError:
                pass
        drop_temp_standalone(work_dir)


def _decode_lossless(input_path: str, decoded_path: str) -> None:
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-xerror",
            "-i",
            input_path,
            "-map",
            "0:a:0",
            "-vn",
            "-c:a",
            "pcm_f32le",
            "-rf64",
            "auto",
            decoded_path,
        ]
    )


def _process_decoded_file(
    pipeline: MagicCleanPipeline,
    decoded_path: str,
    processed_path: str,
    *,
    device: torch.device,
    mode: ContentMode,
    levels: StemLevels | None,
    chunk_samples: int,
    context_samples: int,
    progress: Callable[[int, int], None] | None,
    cancel_event: threading.Event | None,
) -> int:
    """Process overlapping context but write each exact core only once."""
    core_samples = chunk_samples - (2 * context_samples)
    if core_samples <= 0:
        raise ValueError("Magic Clean chunk context leaves no valid core")

    with sf.SoundFile(decoded_path) as source:
        sample_rate = int(source.samplerate)
        channels = int(source.channels)
        total_samples = int(len(source))
        total_chunks = max(1, math.ceil(total_samples / core_samples))
        processed_chunks = 0
        with sf.SoundFile(
            processed_path,
            mode="w",
            samplerate=sample_rate,
            channels=channels,
            format="RF64",
            subtype="FLOAT",
        ) as destination:
            for core_start in range(0, total_samples, core_samples):
                _raise_if_cancelled(cancel_event)
                core_end = min(core_start + core_samples, total_samples)
                window_start = max(0, core_start - context_samples)
                window_end = min(total_samples, core_end + context_samples)
                source.seek(window_start)
                raw = source.read(
                    window_end - window_start,
                    dtype="float32",
                    always_2d=True,
                )
                waveform = torch.from_numpy(raw.T.copy()).to(device)
                cleaned = pipeline.process(
                    waveform,
                    sample_rate,
                    mode,
                    levels,
                    False,
                    False,
                )
                _raise_if_cancelled(cancel_event)
                if tuple(cleaned.shape) != tuple(waveform.shape):
                    raise RuntimeError("Magic Clean model window changed shape")
                if not torch.isfinite(cleaned).all():
                    raise RuntimeError("Magic Clean model window contains non-finite audio")

                core_offset = core_start - window_start
                core_length = core_end - core_start
                core = cleaned[:, core_offset : core_offset + core_length]
                if core.shape[1] != core_length:
                    raise RuntimeError("Magic Clean model window did not cover its core")
                destination.write(core.detach().cpu().T.numpy())
                _raise_if_cancelled(cancel_event)
                del waveform, cleaned, core
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                processed_chunks += 1
                if progress is not None:
                    progress(processed_chunks, total_chunks)
    return processed_chunks


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise MagicCleanProcessingCancelled("Magic Clean processing was cancelled")


def _apply_global_silence_edit(
    pipeline: MagicCleanPipeline,
    source_path: str,
    input_path: str,
    output_path: str,
    *,
    device: torch.device,
    mode: ContentMode,
) -> int:
    file_editor = getattr(pipeline, "strip_silence_file", None)
    if callable(file_editor):
        return int(file_editor(source_path, input_path, output_path))
    waveform, sample_rate = AudioIO.load(input_path)
    edited = pipeline.finalise(
        waveform.to(device),
        sample_rate,
        mode,
        cut_silence=True,
        master=False,
    )
    if edited.ndim != 2 or edited.shape[1] < 1 or not torch.isfinite(edited).all():
        raise RuntimeError("Magic Clean global silence edit returned invalid audio")
    sf.write(
        output_path,
        edited.detach().cpu().T.numpy(),
        sample_rate,
        format="RF64",
        subtype="FLOAT",
    )
    return int(edited.shape[1])


def _measure_loudness(path: str) -> dict[str, str]:
    completed = _run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-xerror",
            "-i",
            path,
            "-af",
            "loudnorm=I=-16:LRA=11:TP=-2:print_format=json",
            "-f",
            "null",
            "-",
        ]
    )
    matches = re.findall(r"\{[^{]*\}", completed.stderr, flags=re.DOTALL)
    if not matches:
        raise RuntimeError("ffmpeg did not return loudness measurements")
    return json.loads(matches[-1])


def _bounded_master_gain_db(input_lufs: float) -> float:
    if not math.isfinite(input_lufs) or input_lufs <= -70.0:
        return 0.0
    requested = TARGET_LUFS - input_lufs
    return max(-MAX_ATTENUATION_DB, min(requested, MAX_GAIN_DB))


def _encode_normalised(
    input_path: str,
    output_path: str,
    measurements: dict[str, str],
    *,
    sample_rate: int,
    channels: int,
    bitrate_kbps: int,
    codec_correction_db: float = 0.0,
) -> None:
    input_lufs = _finite_float(measurements.get("input_i"), default=-99.0)
    gain_db = _bounded_master_gain_db(input_lufs)
    filters = [
        f"volume={gain_db:.6f}dB,"
        f"alimiter=limit={PRE_ENCODE_PEAK_LIMIT:.8f}:level=false:latency=true"
    ]
    if codec_correction_db:
        filters.append(f"volume={codec_correction_db:.6f}dB")
    audio_filter = ",".join(filters)
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-xerror",
            "-i",
            input_path,
            "-af",
            audio_filter,
            "-c:a",
            "libmp3lame",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-b:a",
            f"{bitrate_kbps}k",
            output_path,
        ]
    )


def _encode_with_true_peak_guard(
    input_path: str,
    output_path: str,
    measurements: dict[str, str],
    *,
    sample_rate: int,
    channels: int,
    bitrate_kbps: int,
) -> dict[str, str]:
    """Bound decoded MP3 true peak with at most two corrective re-encodes."""
    correction_db = 0.0
    for attempt in range(MAX_CODEC_TRUE_PEAK_REENCODES + 1):
        _encode_normalised(
            input_path,
            output_path,
            measurements,
            sample_rate=sample_rate,
            channels=channels,
            bitrate_kbps=bitrate_kbps,
            codec_correction_db=correction_db,
        )
        delivered = _measure_loudness(output_path)
        try:
            true_peak = float(delivered["input_tp"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "ffmpeg did not return a delivered true-peak measurement"
            ) from exc
        if true_peak == -math.inf:
            # Digital silence has no defined peak and needs no correction.
            return delivered
        if not math.isfinite(true_peak):
            raise RuntimeError("ffmpeg returned an invalid delivered true peak")
        if true_peak <= DELIVERED_TRUE_PEAK_CEILING_DBTP:
            return delivered
        if attempt == MAX_CODEC_TRUE_PEAK_REENCODES:
            break
        correction_db += CODEC_TRUE_PEAK_TARGET_DBTP - true_peak
        if correction_db < -MAX_CODEC_CORRECTION_DB:
            break
    raise RuntimeError(
        "Magic Clean could not encode an MP3 below the delivered true-peak ceiling"
    )


def _probe(path: str) -> dict[str, float]:
    completed = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            path,
        ]
    )
    return {"duration": float(json.loads(completed.stdout)["format"]["duration"])}


def _mp3_delivery_sample_rate(source_sample_rate: int) -> int:
    """Choose the nearest rate supported by MPEG Layer III.

    Enhancement remains at the decoded source rate; only the final delivery
    encode is converted. Log-ratio distance treats an octave consistently and
    avoids invalid LAME requests for 88.2/96/192 kHz sources.
    """
    if source_sample_rate <= 0:
        raise ValueError("source sample rate must be positive")
    return min(
        SUPPORTED_MP3_SAMPLE_RATES,
        key=lambda candidate: abs(math.log(source_sample_rate / candidate)),
    )


def _finite_float(value: object, *, default: float) -> float:
    if not isinstance(value, (int, float, str)) or isinstance(value, bool):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 3)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        check=True,
        text=True,
        timeout=FFMPEG_TIMEOUT_SECONDS,
    )
