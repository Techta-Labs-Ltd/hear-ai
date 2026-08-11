import numpy as np

from hear.services.transcription.chunks import (
    adaptive_batch_size,
    append_shifted_result,
    finalize_combined_result,
    iter_audio_chunks,
)


def test_adaptive_batch_size_reduces_long_audio_gpu_pressure():
    assert adaptive_batch_size(300, 36, 4) == 36
    assert adaptive_batch_size(1200, 36, 4) == 16
    assert adaptive_batch_size(2400, 36, 4) == 8
    assert adaptive_batch_size(10800, 36, 4) == 4


def test_audio_chunks_are_bounded_and_have_absolute_offsets():
    audio = np.arange(25, dtype=np.float32)

    chunks = list(iter_audio_chunks(audio, sample_rate=2, chunk_seconds=5))

    assert [offset for offset, _ in chunks] == [0.0, 5.0, 10.0]
    assert [len(chunk) for _, chunk in chunks] == [10, 10, 5]


def test_chunk_results_merge_with_shifted_segment_and_word_timestamps():
    combined = {"segments": [], "language": "en"}
    chunk = {
        "segments": [
            {
                "id": 9,
                "start": 1.0,
                "end": 2.0,
                "text": "hello",
                "words": [{"word": "hello", "start": 1.1, "end": 1.9}],
            }
        ],
    }

    append_shifted_result(combined, chunk, offset_seconds=600.0)
    result = finalize_combined_result(combined)

    assert result["segments"][0]["id"] == 0
    assert result["segments"][0]["start"] == 601.0
    assert result["segments"][0]["words"][0]["end"] == 601.9
    assert result["text"] == "hello"
