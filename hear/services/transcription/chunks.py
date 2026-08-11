from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterator

import numpy as np


def adaptive_batch_size(
    duration_seconds: float,
    requested_batch_size: int,
    long_audio_batch_size: int,
) -> int:
    requested = max(1, int(requested_batch_size))
    long_batch = max(1, min(int(long_audio_batch_size), requested))
    if duration_seconds > 3600:
        return long_batch
    if duration_seconds > 1800:
        return min(requested, max(long_batch, 8))
    if duration_seconds > 600:
        return min(requested, max(long_batch, 16))
    return requested


def iter_audio_chunks(
    audio: np.ndarray,
    *,
    sample_rate: int,
    chunk_seconds: int,
) -> Iterator[tuple[float, np.ndarray]]:
    samples_per_chunk = max(sample_rate, int(chunk_seconds) * sample_rate)
    for start in range(0, len(audio), samples_per_chunk):
        yield start / sample_rate, audio[start : start + samples_per_chunk]


def append_shifted_result(
    combined: dict[str, Any],
    chunk_result: dict[str, Any],
    *,
    offset_seconds: float,
) -> None:
    combined.setdefault("segments", [])
    if not combined.get("language") and chunk_result.get("language"):
        combined["language"] = chunk_result["language"]

    for raw_segment in chunk_result.get("segments") or []:
        segment = deepcopy(raw_segment)
        _shift_timestamps(segment, offset_seconds)
        for word in segment.get("words") or []:
            _shift_timestamps(word, offset_seconds)
        segment["id"] = len(combined["segments"])
        combined["segments"].append(segment)

def finalize_combined_result(combined: dict[str, Any]) -> dict[str, Any]:
    segments = combined.get("segments") or []
    combined["text"] = " ".join(
        str(segment.get("text") or "").strip()
        for segment in segments
        if str(segment.get("text") or "").strip()
    )
    return combined


def _shift_timestamps(value: dict[str, Any], offset_seconds: float) -> None:
    for field in ("start", "end"):
        timestamp = value.get(field)
        if isinstance(timestamp, (int, float)):
            value[field] = round(float(timestamp) + offset_seconds, 3)
