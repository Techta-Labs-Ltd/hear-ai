import asyncio
from types import SimpleNamespace

import pytest
import torch

from hear.services.model_client import RayModelClient
from hear.services.reconstruction.synthesizer import SpeechSynthesizer


def _tone(
    duration_seconds: float,
    sample_rate: int = 44100,
) -> torch.Tensor:
    samples = int(duration_seconds * sample_rate)
    time = torch.arange(samples, dtype=torch.float32) / sample_rate

    return (0.2 * torch.sin(2 * torch.pi * 220 * time)).unsqueeze(0)


def test_reconstruction_seed_is_stable_across_retries_and_python_processes():
    synthesizer = SpeechSynthesizer()
    first_seed = synthesizer._compute_seed("job-1", "track-1")

    assert first_seed == synthesizer._compute_seed("job-1", "track-1")
    assert first_seed == synthesizer._compute_seed("job-2", "track-1")
    assert first_seed != synthesizer._compute_seed("job-1", "track-2")

def test_reconstruct_segments_uses_immutable_waveform_for_voice_reference(
    monkeypatch,
):
    synthesizer = SpeechSynthesizer()
    current_waveform = torch.zeros((1, 2 * synthesizer.TARGET_SR))
    immutable_waveform = torch.ones((1, 2 * synthesizer.TARGET_SR))
    observed = {}

    def fake_load(path):
        if path == "current.wav":
            return current_waveform, synthesizer.TARGET_SR
        if path == "immutable.wav":
            return immutable_waveform, synthesizer.TARGET_SR
        raise AssertionError(f"unexpected audio load: {path}")

    async def observe_reference(waveform, start, end, **_kwargs):
        observed.update(waveform=waveform, start=start, end=end)
        raise RuntimeError("reference observed")

    monkeypatch.setattr(
        "hear.services.reconstruction.synthesizer.torchaudio.load",
        fake_load,
    )
    monkeypatch.setattr(
        synthesizer,
        "_prepare_voice_reference",
        observe_reference,
    )

    with pytest.raises(RuntimeError, match="reference observed"):
        asyncio.run(
            synthesizer.reconstruct_segments(
                original_audio_path="current.wav",
                voice_reference_audio_path="immutable.wav",
                track_id="track-1",
                changes=[{
                    "segment_start": 0.25,
                    "segment_end": 1.25,
                    "new_text": "replacement",
                    "original_text": "original",
                }],
                storage=SimpleNamespace(),
                same_speaker=True,
                job_id="job-1",
            )
        )

    assert observed["waveform"] is immutable_waveform
    assert observed["start"] == int(0.25 * synthesizer.TARGET_SR)
    assert observed["end"] == int(1.25 * synthesizer.TARGET_SR)


def test_model_client_forwards_seed_to_fish_deployment():
    captured = {}

    class RemoteMethod:
        def remote(self, *args):
            captured["args"] = args

            async def resolve():
                return b"audio"

            return resolve()

    handle = SimpleNamespace(generate_speech=RemoteMethod())
    client = RayModelClient({"fish_speech": handle})

    result = asyncio.run(client.generate_speech(text="hello", seed=12345))

    assert result == b"audio"
    assert captured["args"] == (
        "hello",
        1024,
        None,
        None,
        "en",
        12345,
    )


def test_voice_reference_uses_ten_seconds_before_a_middle_edit():
    synthesizer = SpeechSynthesizer()
    waveform = _tone(30.0)
    edit_start = int(14.8 * synthesizer.TARGET_SR)
    edit_end = int(15.2 * synthesizer.TARGET_SR)

    start, end = synthesizer._reference_clip_bounds(
        waveform,
        edit_start,
        edit_end,
    )

    assert (end - start) / synthesizer.TARGET_SR == 10.0
    assert end == edit_start - int(
        synthesizer.VOICE_REFERENCE_GUARD_SECONDS * synthesizer.TARGET_SR
    )
    assert end < edit_start


def test_voice_reference_uses_clean_audio_after_an_edit_at_track_start():
    synthesizer = SpeechSynthesizer()
    waveform = _tone(30.0)
    edit_end = int(0.3 * synthesizer.TARGET_SR)

    start, end = synthesizer._reference_clip_bounds(
        waveform,
        0,
        edit_end,
    )

    assert start == edit_end + int(
        synthesizer.VOICE_REFERENCE_GUARD_SECONDS * synthesizer.TARGET_SR
    )
    assert end - start == int(10.0 * synthesizer.TARGET_SR)


def test_voice_reference_excludes_the_latest_jobs_original_hello_interval():
    synthesizer = SpeechSynthesizer()
    waveform = _tone(30.0)
    edit_start = int(1.921 * synthesizer.TARGET_SR)
    edit_end = int(8.721 * synthesizer.TARGET_SR)

    start, end = synthesizer._reference_clip_bounds(
        waveform,
        edit_start,
        edit_end,
    )

    assert start >= edit_end + int(
        synthesizer.VOICE_REFERENCE_GUARD_SECONDS * synthesizer.TARGET_SR
    )
    assert end - start == int(10.0 * synthesizer.TARGET_SR)


def test_voice_reference_uses_clean_audio_before_an_edit_at_track_end():
    synthesizer = SpeechSynthesizer()
    waveform = _tone(30.0)
    edit_start = int(29.5 * synthesizer.TARGET_SR)

    start, end = synthesizer._reference_clip_bounds(
        waveform,
        edit_start,
        int(30.0 * synthesizer.TARGET_SR),
    )

    assert end == edit_start - int(
        synthesizer.VOICE_REFERENCE_GUARD_SECONDS * synthesizer.TARGET_SR
    )
    assert end - start == int(10.0 * synthesizer.TARGET_SR)


def test_voice_reference_uses_transcript_aligned_to_expanded_clip(
    monkeypatch, tmp_path
):
    synthesizer = SpeechSynthesizer()
    waveform = _tone(30.0)
    reference_path = tmp_path / "reference.wav"
    reference_path.write_bytes(b"reference audio")
    exported_bounds = {}

    def fake_export(_waveform, start, end, *, track_id):
        exported_bounds.update(start=start, end=end, track_id=track_id)
        return str(reference_path)

    async def fake_transcribe(_audio_bytes, **_kwargs):
        return {"transcript": "The words spoken across the full reference clip."}

    monkeypatch.setattr(synthesizer, "_export_reference_clip", fake_export)
    monkeypatch.setattr(
        "hear.services.reconstruction.synthesizer._get_transcriber",
        lambda: SimpleNamespace(transcribe=fake_transcribe),
    )

    path, text, speaking_rate = asyncio.run(
        synthesizer._prepare_voice_reference(
            waveform,
            int(14.8 * synthesizer.TARGET_SR),
            int(15.2 * synthesizer.TARGET_SR),
            track_id="track-1",
        )
    )

    assert path == str(reference_path)
    assert text == "The words spoken across the full reference clip."
    assert speaking_rate is None
    assert (exported_bounds["end"] - exported_bounds["start"]) == int(
        10.0 * synthesizer.TARGET_SR
    )
    assert exported_bounds["end"] < int(14.8 * synthesizer.TARGET_SR)


def test_voice_reference_is_skipped_when_edit_leaves_no_clean_context(monkeypatch):
    synthesizer = SpeechSynthesizer()
    waveform = _tone(2.0)

    def fail_export(*_args, **_kwargs):
        raise AssertionError("empty reference must not be exported")

    monkeypatch.setattr(synthesizer, "_export_reference_clip", fail_export)

    path, text, speaking_rate = asyncio.run(
        synthesizer._prepare_voice_reference(
            waveform,
            0,
            int(2.0 * synthesizer.TARGET_SR),
            track_id="track-1",
        )
    )

    assert path is None
    assert text == ""
    assert speaking_rate is None


def test_voice_reference_prefers_aligned_edited_speaker_and_returns_rate(
    monkeypatch, tmp_path
):
    synthesizer = SpeechSynthesizer()
    waveform = _tone(30.0)
    reference_path = tmp_path / "reference.wav"
    reference_path.write_bytes(b"reference audio")
    exported_bounds = {}

    def fake_export(_waveform, start, end, *, track_id):
        exported_bounds.update(start=start, end=end, track_id=track_id)
        return str(reference_path)

    async def fake_transcribe(_audio_bytes, **_kwargs):
        return {
            "transcript": "one two three four",
            "segments": [{
                "start": 0.0,
                "end": 3.0,
                "text": "one two three four",
                "words": [
                    {"word": "one", "start": 0.0, "end": 0.5},
                    {"word": "two", "start": 0.8, "end": 1.3},
                    {"word": "three", "start": 1.8, "end": 2.3},
                    {"word": "four", "start": 2.5, "end": 3.0},
                ],
            }],
        }

    monkeypatch.setattr(synthesizer, "_export_reference_clip", fake_export)
    monkeypatch.setattr(
        "hear.services.reconstruction.synthesizer._get_transcriber",
        lambda: SimpleNamespace(transcribe=fake_transcribe),
    )

    edit_start = int(10.0 * synthesizer.TARGET_SR)
    edit_end = int(16.0 * synthesizer.TARGET_SR)
    path, text, speaking_rate = asyncio.run(
        synthesizer._prepare_voice_reference(
            waveform,
            edit_start,
            edit_end,
            track_id="track-1",
            original_text="the trusted original words",
        )
    )

    assert path == str(reference_path)
    assert text == "one two three four"
    assert speaking_rate is not None and abs(speaking_rate - (4.0 / 3.0)) < 1e-9
    assert exported_bounds["start"] == edit_start
    assert exported_bounds["end"] == edit_start + 3 * synthesizer.TARGET_SR


def test_long_edit_uses_full_pacing_window_and_aligned_ten_second_clone(
    monkeypatch, tmp_path
):
    synthesizer = SpeechSynthesizer()
    waveform = _tone(40.0)
    reference_path = tmp_path / "reference.wav"
    reference_path.write_bytes(b"reference audio")
    observed = {}

    def fake_export(_waveform, start, end, *, track_id):
        observed.update(reference_start=start, reference_end=end, track_id=track_id)
        return str(reference_path)

    def fake_wav_bytes(audio, sampling_rate):
        observed.update(
            pacing_samples=len(audio),
            pacing_sample_rate=sampling_rate,
        )
        return b"pacing audio"

    async def fake_transcribe(_audio_bytes, **_kwargs):
        observed["short_utterance"] = _kwargs["short_utterance"]
        return {
            "transcript": "one two three four",
            "segments": [{
                "start": 1.0,
                "end": 29.0,
                "text": "one two three four",
                "words": [
                    {"word": "one", "start": 1.0, "end": 2.0},
                    {"word": "two", "start": 8.0, "end": 9.0},
                    {"word": "three", "start": 13.0, "end": 14.0},
                    {"word": "four", "start": 28.0, "end": 29.0},
                ],
            }],
        }

    monkeypatch.setattr(synthesizer, "_export_reference_clip", fake_export)
    monkeypatch.setattr(synthesizer, "_wav_bytes_from_audio", fake_wav_bytes)
    monkeypatch.setattr(
        "hear.services.reconstruction.synthesizer._get_transcriber",
        lambda: SimpleNamespace(transcribe=fake_transcribe),
    )

    edit_start = 0
    edit_end = 40 * synthesizer.TARGET_SR
    path, text, speaking_rate = asyncio.run(
        synthesizer._prepare_voice_reference(
            waveform,
            edit_start,
            edit_end,
            track_id="track-1",
            original_text="trusted original words",
        )
    )

    assert path == str(reference_path)
    assert text == "three"
    assert speaking_rate is None
    assert observed["pacing_samples"] == int(
        synthesizer.MAX_PACING_REFERENCE_SECONDS * synthesizer.TARGET_SR
    )
    assert observed["pacing_sample_rate"] == synthesizer.TARGET_SR
    assert observed["short_utterance"] is False
    assert observed["reference_start"] == 18 * synthesizer.TARGET_SR
    assert observed["reference_end"] == 19 * synthesizer.TARGET_SR


def test_reference_window_excludes_words_cut_by_audio_boundaries():
    transcription = {
        "segments": [{
            "words": [
                {"word": "partial-left", "start": 0.8, "end": 1.2},
                {"word": "one", "start": 1.3, "end": 2.0},
                {"word": "two", "start": 2.2, "end": 3.5},
                {"word": "partial-right", "start": 3.8, "end": 4.2},
            ]
        }]
    }

    aligned = SpeechSynthesizer._complete_word_reference_window(
        transcription,
        1.0,
        4.0,
    )

    assert aligned == (1.3, 3.5, "one two")


def test_speech_rate_match_safely_moves_fast_tts_toward_source_rate():
    synthesizer = SpeechSynthesizer()

    stretched = synthesizer._time_stretch_to_match(
        _tone(1.0),
        _tone(1.0),
        original_text="one two",
        new_text="one two three four",
        source_speaking_rate=2.0,
    )

    assert abs(stretched.shape[1] / synthesizer.TARGET_SR - (1.0 / 0.75)) < 0.01


def test_speech_rate_match_safely_moves_slow_tts_toward_source_rate():
    synthesizer = SpeechSynthesizer()

    stretched = synthesizer._time_stretch_to_match(
        _tone(2.0),
        _tone(1.0),
        original_text="one two",
        new_text="one two",
        source_speaking_rate=2.0,
    )

    assert abs(stretched.shape[1] / synthesizer.TARGET_SR - (2.0 / 1.35)) < 0.01


def test_longer_replacement_is_not_forced_into_the_original_interval():
    synthesizer = SpeechSynthesizer()

    stretched = synthesizer._time_stretch_to_match(
        _tone(2.0),
        _tone(1.0),
        original_text="one two",
        new_text="one two three four",
        source_speaking_rate=2.0,
    )

    assert stretched.shape[1] == 2 * synthesizer.TARGET_SR


def test_latest_live_job_uses_aligned_span_without_hitting_slowdown_floor():
    synthesizer = SpeechSynthesizer()
    text = "word " * 40

    stretched = synthesizer._time_stretch_to_match(
        _tone(15.975),
        _tone(20.480),
        original_text=text,
        new_text=text,
        source_speaking_rate=40.0 / 20.480,
    )

    duration = stretched.shape[1] / synthesizer.TARGET_SR
    assert abs(duration - 20.480) < 0.01

def test_splice_never_searches_for_or_deletes_untouched_following_audio(monkeypatch):
    synthesizer = SpeechSynthesizer()
    sample_rate = synthesizer.TARGET_SR
    original = _tone(3.0)
    replacement = _tone(1.0)

    def fail_correlation(*_args, **_kwargs):
        raise AssertionError("splice must not correlate into untouched following audio")

    monkeypatch.setattr(torch.nn.functional, "conv1d", fail_correlation)

    result = synthesizer._splice_segment(
        original.clone(), replacement.clone(), sample_rate, 2 * sample_rate
    )
    crossfade_samples = int(0.03 * sample_rate)
    expected_samples = 3 * sample_rate - 2 * crossfade_samples

    assert result.shape[1] == expected_samples


def test_speech_rate_fallback_preserves_the_complete_source_span():
    synthesizer = SpeechSynthesizer()
    reference = torch.cat(
        [
            torch.zeros((1, synthesizer.TARGET_SR)),
            _tone(1.0),
            torch.zeros((1, synthesizer.TARGET_SR)),
        ],
        dim=1,
    )

    stretched = synthesizer._time_stretch_to_match(
        _tone(1.0),
        reference,
        original_text="one two",
        new_text="one two",
    )

    duration = stretched.shape[1] / synthesizer.TARGET_SR
    assert abs(duration - (1.0 / 0.75)) < 0.01


def test_speech_rate_match_bounds_extreme_tempo_changes():
    synthesizer = SpeechSynthesizer()
    text = "one two three four"

    shortened = synthesizer._time_stretch_to_match(
        _tone(4.0),
        _tone(1.0),
        original_text=text,
        new_text=text,
        source_speaking_rate=4.0,
    )
    lengthened = synthesizer._time_stretch_to_match(
        _tone(1.0),
        _tone(4.0),
        original_text=text,
        new_text=text,
        source_speaking_rate=1.0,
    )

    assert abs(shortened.shape[1] / synthesizer.TARGET_SR - (4.0 / 1.35)) < 0.01
    assert abs(lengthened.shape[1] / synthesizer.TARGET_SR - (1.0 / 0.75)) < 0.01


def test_speech_rate_match_prefers_aligned_asr_rate_over_noisy_reference():
    synthesizer = SpeechSynthesizer()

    stretched = synthesizer._time_stretch_to_match(
        _tone(2.0),
        _tone(10.0),
        original_text="one two",
        new_text="one two three four",
        source_speaking_rate=3.0,
    )

    assert abs(stretched.shape[1] / synthesizer.TARGET_SR - (2.0 / 1.35)) < 0.01


def test_speech_rate_match_failure_preserves_natural_tts_speed(monkeypatch):
    synthesizer = SpeechSynthesizer()

    def fail_time_stretch(*_args, **_kwargs):
        raise RuntimeError("test failure")

    monkeypatch.setattr(
        "hear.services.reconstruction.synthesizer.torchaudio.sox_effects.apply_effects_tensor",
        fail_time_stretch,
    )

    shortened = synthesizer._time_stretch_to_match(
        _tone(3.0),
        _tone(1.0),
        original_text="one two",
        new_text="one two",
        source_speaking_rate=2.0,
    )
    lengthened = synthesizer._time_stretch_to_match(
        _tone(1.0),
        _tone(3.0),
        original_text="one two",
        new_text="one two",
        source_speaking_rate=1.0,
    )

    assert shortened.shape[1] == 3 * synthesizer.TARGET_SR
    assert lengthened.shape[1] == synthesizer.TARGET_SR


def test_speech_rate_match_without_aligned_text_keeps_natural_tts_speed():
    synthesizer = SpeechSynthesizer()

    result = synthesizer._time_stretch_to_match(_tone(2.0), _tone(1.0))

    assert result.shape[1] == 2 * synthesizer.TARGET_SR


def test_active_speech_duration_excludes_long_silence():
    synthesizer = SpeechSynthesizer()
    waveform = torch.cat(
        [
            torch.zeros((1, synthesizer.TARGET_SR)),
            _tone(1.0),
            torch.zeros((1, synthesizer.TARGET_SR)),
        ],
        dim=1,
    )

    duration = synthesizer._active_speech_duration(waveform, synthesizer.TARGET_SR)

    assert 0.95 <= duration <= 1.1


def test_source_activity_detector_separates_speech_from_stationary_background():
    synthesizer = SpeechSynthesizer()
    background = torch.full(
        (1, 10 * synthesizer.TARGET_SR),
        0.01,
        dtype=torch.float32,
    )
    speech_start = 4 * synthesizer.TARGET_SR
    speech_end = 5 * synthesizer.TARGET_SR
    background[:, speech_start:speech_end] += _tone(1.0)

    duration = synthesizer._active_speech_duration(
        background,
        synthesizer.TARGET_SR,
        allow_uniform_activity=False,
    )

    assert 0.95 <= duration <= 1.1


def test_source_activity_detector_rejects_uniform_background():
    synthesizer = SpeechSynthesizer()
    background = torch.full(
        (1, 10 * synthesizer.TARGET_SR),
        0.01,
        dtype=torch.float32,
    )

    duration = synthesizer._active_speech_duration(
        background,
        synthesizer.TARGET_SR,
        allow_uniform_activity=False,
    )

    assert duration == 0.0


def test_speech_units_ignore_pacing_control_tokens():
    synthesizer = SpeechSynthesizer()

    assert synthesizer._speech_units("Hello [pause] there [speaking slowly].") == 2


def test_preprocessor_does_not_force_a_pause_after_every_punctuation_mark():
    synthesizer = SpeechSynthesizer()

    processed = asyncio.run(synthesizer._preprocess_for_s2("Hello, world. How are you?"))

    assert processed == "Hello, world. How are you?"


def test_preprocessor_preserves_explicit_paragraph_pause():
    synthesizer = SpeechSynthesizer()

    processed = asyncio.run(synthesizer._preprocess_for_s2("First paragraph.\n\nSecond."))

    assert processed == "First paragraph. [pause] Second."
