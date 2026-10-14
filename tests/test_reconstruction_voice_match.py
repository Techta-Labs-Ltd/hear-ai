"""Voice-reference policy, take reproducibility, acoustic matching and cheap previews."""

import asyncio
import hashlib
import io
import threading
import time

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from hear.contracts.reconstruction import ReconstructionOptions, VoiceReference
from hear.execution.native import NativeExecutor
from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.services.reconstruction.acoustic_match import AcousticMatcher
from hear.services.reconstruction.fish_renderer import FishReconstructionRenderer
from hear.workflows.fish_reconstruction import FishReconstructionWorkflow
from tests.test_fish_tts_contract import (
    REFERENCE,
    REFERENCE_TEXT,
    FileSource,
    MemoryStorage,
    SyntheticSpeechDouble,
    request,
    source_file,
)

RATE = 48000


def short_edit(**extra):
    return {
        "same_speaker": True,
        "changes": [
            {"segment_start": 6, "segment_end": 7, "new_text": "New.", "original_text": "Old."}
        ],
        **extra,
    }


def test_short_same_speaker_edit_needs_an_explicit_reference():
    with pytest.raises(ValueError, match="explicit_reference_or_5_to_20"):
        ReconstructionOptions.validate_operation("replace_segments", short_edit())
    ReconstructionOptions.validate_operation("replace_segments", short_edit(reference=REFERENCE))


def test_edit_interval_of_five_seconds_may_reference_itself():
    raw = {
        "same_speaker": True,
        "changes": [
            {
                "segment_start": 2,
                "segment_end": 7,
                "new_text": "New.",
                "original_text": "Old words.",
            }
        ],
    }
    assert ReconstructionOptions.validate_operation("preview", raw).changes[0].can_self_reference


@pytest.mark.parametrize("end", [3.0, 4.99, 25.5])
def test_reference_outside_five_to_twenty_seconds_rejected(end):
    with pytest.raises(ValueError, match="5_to_20"):
        VoiceReference(start_seconds=0, end_seconds=end, text="words")


@pytest.mark.parametrize("seed", [-1, 2**31, "7"])
def test_invalid_seed_rejected(seed):
    with pytest.raises(ValueError):
        ReconstructionOptions.validate_operation(
            "replace_segments", short_edit(reference=REFERENCE, seed=seed)
        )


def test_one_seed_per_edit_and_caller_seed_reproduces_it():
    derived = [FishReconstructionRenderer.edit_seed(None, "job:abc", i) for i in range(3)]
    assert len(set(derived)) == 3 and all(0 <= s < 2**31 for s in derived)
    assert FishReconstructionRenderer.edit_seed(None, "job:abc", 0) == derived[0]
    assert [FishReconstructionRenderer.edit_seed(41, "job:abc", i) for i in range(2)] == [41, 57]


def test_text_chunks_stay_inside_the_token_budget():
    text = "This is a precise replacement sentence. " * 100
    chunks = FishReconstructionRenderer.text_chunks(text)
    assert " ".join(chunks).split() == text.split()
    assert max(map(len, chunks)) <= FishReconstructionRenderer.MAX_CHUNK_CHARACTERS == 200


def speech_like(seconds, level=0.1, rate=RATE, seed=0):
    """Band-limited noise with syllable-rate amplitude modulation; not silence."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    envelope = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 4 * t))
    tone = sum(np.sin(2 * np.pi * f * t) for f in (140, 280, 560, 1120, 2240))
    noise = rng.standard_normal(len(t)) * 0.3
    signal = (tone / 5 + noise) * envelope
    return (signal / np.max(np.abs(signal)) * level).astype("float32")


def guard_for(tmp_path):
    (tmp_path / "ws").mkdir()
    return ResourceGuard(
        ResourceBudget(8 * 1024**3, 2 * 1024**3, RATE * 3600),
        tmp_path / "ws",
        time.monotonic() + 60,
        threading.Event(),
    )


def write(path, values):
    sf.write(path, values, RATE, subtype="FLOAT")
    return path


def test_mostly_silent_reference_rejected(tmp_path):
    values = np.zeros(RATE * 6, dtype="float32")
    values[: RATE * 2] = speech_like(2)
    source = write(tmp_path / "s.wav", values)
    reference = VoiceReference(start_seconds=0, end_seconds=6, text=REFERENCE_TEXT)
    with pytest.raises(ValueError, match="mostly_silent"):
        FishReconstructionRenderer.reference_bytes(source, reference, guard_for(tmp_path))


def test_reference_text_must_fit_its_duration(tmp_path):
    source = write(tmp_path / "s.wav", speech_like(6))
    guard = guard_for(tmp_path)
    with pytest.raises(ValueError, match="text_length_mismatch"):
        FishReconstructionRenderer.reference_bytes(
            source, VoiceReference(start_seconds=0, end_seconds=6, text="Hi."), guard
        )
    with pytest.raises(ValueError, match="text_length_mismatch"):
        FishReconstructionRenderer.reference_bytes(
            source, VoiceReference(start_seconds=0, end_seconds=6, text="x" * 400), guard
        )
    payload, quality = FishReconstructionRenderer.reference_bytes(
        source, VoiceReference(start_seconds=0, end_seconds=6, text=REFERENCE_TEXT), guard
    )
    assert sf.info(io.BytesIO(payload)).duration == 6.0
    assert quality.warnings == ("reference_shorter_than_recommended",)


def test_clipped_reference_rejected(tmp_path):
    values = speech_like(6, level=1.0)
    values = np.clip(values * 4, -1, 1)
    source = write(tmp_path / "s.wav", values)
    with pytest.raises(ValueError, match="clipped"):
        FishReconstructionRenderer.reference_bytes(
            source,
            VoiceReference(start_seconds=0, end_seconds=6, text=REFERENCE_TEXT),
            guard_for(tmp_path),
        )


def level_dbfs(values):
    rms, _ = AcousticMatcher.voiced_level(values)
    return 20 * np.log10(rms)


def test_generated_level_is_matched_to_the_surrounding_speech():
    context = speech_like(4, level=0.3, seed=1)
    generated = speech_like(1, level=0.1, seed=2)
    matched, report = AcousticMatcher.match_channel(generated, context)
    assert abs(level_dbfs(matched) - level_dbfs(context)) < 1.0
    assert 8 < report.gain_db < 12 and report.lowpass_hz is None and not report.room_tone_added


def test_gain_is_bounded_and_peak_safe():
    context = speech_like(4, level=0.9, seed=1)
    generated = speech_like(1, level=0.001, seed=2)
    matched, report = AcousticMatcher.match_channel(generated, context)
    assert report.gain_db == AcousticMatcher.MAX_GAIN_DB
    assert np.max(np.abs(matched)) <= 0.98


def test_bandwidth_is_matched_when_the_source_is_band_limited():
    t = np.arange(RATE * 4) / RATE
    rng = np.random.default_rng(3)
    # A narrow-band source (telephone-like): energy only below ~3.4 kHz.
    sos = butter(8, 3400, btype="low", fs=RATE, output="sos")
    context = sosfiltfilt(sos, rng.standard_normal(len(t))).astype("float32")
    context *= 0.1 / np.max(np.abs(context))
    generated = (0.1 * np.sin(2 * np.pi * 10000 * t[:RATE])).astype("float32")
    matched, report = AcousticMatcher.match_channel(generated, context)
    assert report.lowpass_hz is not None and report.lowpass_hz < 4000
    # The 10 kHz tone is far above the source bandwidth and must be removed.
    assert np.sqrt(np.mean(matched**2)) < 0.01 * np.sqrt(np.mean(generated**2))


def test_room_tone_is_added_under_clean_generated_speech():
    rng = np.random.default_rng(4)
    context = speech_like(4, level=0.2, seed=5) + (0.003 * rng.standard_normal(RATE * 4)).astype(
        "float32"
    )
    context[RATE : RATE + RATE // 2] = (0.003 * rng.standard_normal(RATE // 2)).astype("float32")
    generated = speech_like(1, level=0.2, seed=6)
    generated[: RATE // 10] = 0  # a clean pause that would otherwise drop to digital silence
    matched, report = AcousticMatcher.match_channel(generated, context)
    assert report.room_tone_added
    pause = matched[: RATE // 10]
    assert 0.001 < np.sqrt(np.mean(pause**2)) < 0.01


def test_room_tone_tiling_is_continuous():
    window = np.ones(RATE // 2, dtype="float32")
    tiled = AcousticMatcher.tile_room_tone(window, RATE * 2)
    assert len(tiled) == RATE * 2
    interior = tiled[AcousticMatcher.CROSSFADE : -AcousticMatcher.CROSSFADE]
    assert np.allclose(interior, 1.0, atol=1e-5)


def run_workflow(tmp_path, operation, options, fish=None, storage=None):
    path, _ = source_file(tmp_path, seconds=8)
    fish = fish or SyntheticSpeechDouble()
    storage = storage or MemoryStorage()

    async def run():
        native = NativeExecutor("voice-match")
        try:
            workflow = FishReconstructionWorkflow(
                FishReconstructionRenderer(fish, native),
                FileSource(path),
                storage,
                native,
                workspace_root=tmp_path / "jobs",
            )
            events = [e async for e in workflow.stream(request(path, operation, options))]
            return events[-1].data["outcome"]
        finally:
            await native.close()

    return asyncio.run(run()), fish, storage


def test_preview_uploads_only_the_generated_segment(tmp_path):
    outcome, fish, storage = run_workflow(
        tmp_path, "preview", short_edit(reference=REFERENCE, seed=9)
    )
    result = outcome["result"]["reconstructed_audio"]
    assert result["preview"] is True and result["delivery"] is None
    assert list(storage.uploads) == [
        "creators/owner/audio/jobs/job/attempt/segments/segment-000.mp3"
    ]
    row = result["segments"][0]
    assert row["seed"] == 9 and row["audio_url"].endswith("segment-000.mp3")
    assert row["output_start_frame"] == 6 * RATE and row["output_end_frame"] == 6 * RATE + 24000
    assert result["output_frames"] == 8 * RATE - RATE + 24000
    assert fish.calls[0]["seed"] == 9 and fish.calls[0]["references"][0]["text"] == REFERENCE_TEXT
    assert row["crossfade_frames"] == 960


def test_confirm_with_the_preview_seed_sends_identical_requests(tmp_path):
    _, preview, _ = run_workflow(tmp_path, "preview", short_edit(reference=REFERENCE, seed=9))
    (tmp_path / "confirm").mkdir()
    _, confirm, storage = run_workflow(
        tmp_path / "confirm", "replace_segments", short_edit(reference=REFERENCE, seed=9)
    )
    strip = lambda calls: [{k: v for k, v in c.items()} for c in calls]  # noqa: E731
    assert strip(preview.calls) == strip(confirm.calls)
    assert len(storage.uploads) == 2


def test_every_chunk_of_an_edit_shares_one_seed(tmp_path):
    options = {
        "same_speaker": True,
        "reference": REFERENCE,
        "changes": [
            {
                "segment_start": 6,
                "segment_end": 6.5,
                "new_text": "word " * 100,
                "original_text": "x",
            },
            {"segment_start": 7, "segment_end": 7.5, "new_text": "other", "original_text": "y"},
        ],
    }
    outcome, fish, _ = run_workflow(tmp_path, "edit_transcript", options)
    rows = outcome["result"]["reconstructed_audio"]["segments"]
    assert rows[0]["text_chunks"] == 3 and rows[1]["text_chunks"] == 1
    seeds = [c["seed"] for c in fish.calls]
    assert len(seeds) == 4 and len(set(seeds[:3])) == 1 and seeds[3] != seeds[0]
    assert rows[0]["seed"] == seeds[0] and rows[1]["seed"] == seeds[3]
    digest = hashlib.sha256(fish.calls[0]["references"][0]["audio"]).hexdigest()
    assert {r["reference_sha256"] for r in rows} == {digest}
