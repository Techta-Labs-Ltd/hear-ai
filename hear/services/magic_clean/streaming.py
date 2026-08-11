"""Disk-backed Magic Clean processing for long recordings."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch

from hear.core.hear_temp import drop_temp_standalone, hear_temp_standalone_dir
from hear.services.magic_clean.models import ContentMode, StemLevels
from hear.services.magic_clean.pipeline import MagicCleanPipeline
from hear.services.magic_clean.processing.audio_io import AudioIO


@dataclass(frozen=True, slots=True)
class StreamingCleanResult:
    output_path: str
    input_duration_seconds: float
    output_duration_seconds: float
    peak_db: float
    integrated_lufs: float
    chunks_processed: int


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
) -> StreamingCleanResult:
    """Clean a file without keeping the decoded recording or result in RAM."""
    sample_rate = AudioIO.TARGET_SR
    chunk_samples, overlap_samples = pipeline._chunk_sizes(
        sample_rate, chunk_seconds, overlap_seconds
    )
    if cut_silence:
        overlap_samples = 0
    step_samples = chunk_samples - overlap_samples

    work_dir = hear_temp_standalone_dir("magic_clean_stream")
    decoded_path = os.path.join(work_dir, "decoded.wav")
    processed_path = os.path.join(work_dir, "processed.wav")
    temporary_output = output_path + ".partial.mp3"
    try:
        _run(
            [
                "ffmpeg", "-nostdin", "-y", "-v", "error", "-i", input_path,
                "-map", "0:a:0", "-ac", "1", "-ar", str(sample_rate),
                "-c:a", "pcm_f32le", decoded_path,
            ]
        )
        chunks = _process_decoded_file(
            pipeline,
            decoded_path,
            processed_path,
            device=device,
            mode=mode,
            levels=levels,
            cut_silence=cut_silence,
            chunk_samples=chunk_samples,
            overlap_samples=overlap_samples,
            step_samples=step_samples,
            progress=progress,
        )
        measurements = _measure_loudness(processed_path)
        _encode_normalised(
            processed_path,
            temporary_output,
            measurements,
            bitrate_kbps=bitrate_kbps,
        )
        os.replace(temporary_output, output_path)
        source_probe = _probe(input_path)
        output_probe = _probe(output_path)
        return StreamingCleanResult(
            output_path=output_path,
            input_duration_seconds=source_probe["duration"],
            output_duration_seconds=output_probe["duration"],
            peak_db=-1.0,
            integrated_lufs=-16.0,
            chunks_processed=chunks,
        )
    finally:
        for path in (temporary_output, decoded_path, processed_path):
            try:
                os.unlink(path)
            except OSError:
                pass
        drop_temp_standalone(work_dir)


def _process_decoded_file(
    pipeline: MagicCleanPipeline,
    decoded_path: str,
    processed_path: str,
    *,
    device: torch.device,
    mode: ContentMode,
    levels: StemLevels | None,
    cut_silence: bool,
    chunk_samples: int,
    overlap_samples: int,
    step_samples: int,
    progress: Callable[[int, int], None] | None,
) -> int:
    sample_rate = AudioIO.TARGET_SR
    pending: np.ndarray | None = None
    processed_chunks = 0
    with sf.SoundFile(decoded_path) as source, sf.SoundFile(
        processed_path,
        mode="w",
        samplerate=sample_rate,
        channels=1,
        subtype="FLOAT",
    ) as destination:
        total_chunks = max(1, (len(source) + step_samples - 1) // step_samples)
        start = 0
        while start < len(source):
            source.seek(start)
            raw = source.read(chunk_samples, dtype="float32", always_2d=False)
            if raw.size == 0:
                break
            waveform = torch.from_numpy(raw.copy()).unsqueeze(0).to(device)
            cleaned = pipeline.process(
                waveform,
                sample_rate,
                mode,
                levels,
                cut_silence,
                False,
            ).detach().cpu().squeeze(0).numpy()
            del waveform
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if pending is None:
                pending = cleaned
            else:
                fade_samples = min(overlap_samples, pending.size, cleaned.size)
                if fade_samples:
                    phase = np.linspace(0.0, np.pi / 2, fade_samples, dtype=np.float32)
                    fade_out = np.cos(phase) ** 2
                    fade_in = np.sin(phase) ** 2
                    destination.write(pending[:-fade_samples])
                    destination.write(
                        pending[-fade_samples:] * fade_out
                        + cleaned[:fade_samples] * fade_in
                    )
                    pending = cleaned[fade_samples:]
                else:
                    destination.write(pending)
                    pending = cleaned
            processed_chunks += 1
            if progress is not None:
                progress(processed_chunks, total_chunks)
            start += step_samples
        if pending is not None:
            destination.write(pending)
    return processed_chunks


def _measure_loudness(path: str) -> dict[str, str]:
    completed = _run(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-i", path,
            "-af", "loudnorm=I=-16:LRA=11:TP=-1:print_format=json",
            "-f", "null", "-",
        ]
    )
    matches = re.findall(r"\{[^{]*\}", completed.stderr, flags=re.DOTALL)
    if not matches:
        raise RuntimeError("ffmpeg did not return loudness measurements")
    return json.loads(matches[-1])


def _encode_normalised(
    input_path: str,
    output_path: str,
    measurements: dict[str, str],
    *,
    bitrate_kbps: int,
) -> None:
    loudnorm = (
        "loudnorm=I=-16:LRA=11:TP=-1:linear=true:"
        f"measured_I={measurements['input_i']}:"
        f"measured_LRA={measurements['input_lra']}:"
        f"measured_TP={measurements['input_tp']}:"
        f"measured_thresh={measurements['input_thresh']}:"
        f"offset={measurements['target_offset']}:print_format=summary"
    )
    _run(
        [
            "ffmpeg", "-nostdin", "-y", "-v", "error", "-i", input_path,
            "-af", loudnorm, "-ar", str(AudioIO.TARGET_SR), "-ac", "1",
            "-b:a", f"{bitrate_kbps}k", output_path,
        ]
    )


def _probe(path: str) -> dict[str, float]:
    completed = _run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", path,
        ]
    )
    return {"duration": float(json.loads(completed.stdout)["format"]["duration"])}


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, check=True, text=True)
