import numpy as np
import torch

from hear.services.reconstruction.tts_post_processor import TTSPostProcessor

SAMPLE_RATE = 44100


def _tone(frequency: float, duration_seconds: float = 1.0) -> torch.Tensor:
    time = np.arange(round(SAMPLE_RATE * duration_seconds)) / SAMPLE_RATE
    samples = (0.25 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
    return torch.from_numpy(samples).unsqueeze(0)


def test_pitch_estimator_detects_fundamental_frequency():
    pitch = TTSPostProcessor._estimate_median_pitch_hz(_tone(220.0), SAMPLE_RATE)

    assert pitch is not None
    assert abs(pitch - 220.0) < 3.0


def test_pitch_match_moves_tts_toward_reference_and_preserves_duration():
    tts = _tone(180.0, 2.0)
    reference = _tone(240.0, 2.0)

    matched = TTSPostProcessor.match_pitch(tts, reference, SAMPLE_RATE)
    matched_pitch = TTSPostProcessor._estimate_median_pitch_hz(matched, SAMPLE_RATE)

    assert matched.shape == tts.shape
    assert matched_pitch is not None
    assert abs(matched_pitch - 240.0) < abs(180.0 - 240.0)
    assert abs(matched_pitch - 240.0) < 5.0
    assert torch.isfinite(matched).all()


def test_pitch_match_leaves_already_matched_voice_unchanged():
    tts = _tone(220.0)
    reference = _tone(222.0)

    matched = TTSPostProcessor.match_pitch(tts, reference, SAMPLE_RATE)

    assert torch.equal(matched, tts)


def test_pitch_match_skips_silence_without_fabricating_pitch():
    silence = torch.zeros((1, SAMPLE_RATE), dtype=torch.float32)

    matched = TTSPostProcessor.match_pitch(silence, _tone(220.0), SAMPLE_RATE)

    assert torch.equal(matched, silence)


def test_boundary_pitch_match_corrects_joins_without_shifting_middle():
    tts = torch.cat([_tone(250.0, 1.5), _tone(200.0, 2.0), _tone(250.0, 1.5)], dim=1)
    reference = _tone(200.0, 5.0)

    matched = TTSPostProcessor.match_boundary_pitch(tts, reference, SAMPLE_RATE)
    opening_pitch = TTSPostProcessor._estimate_median_pitch_hz(
        matched[:, :SAMPLE_RATE],
        SAMPLE_RATE,
    )
    middle_pitch = TTSPostProcessor._estimate_median_pitch_hz(
        matched[:, 2 * SAMPLE_RATE:3 * SAMPLE_RATE],
        SAMPLE_RATE,
    )
    closing_pitch = TTSPostProcessor._estimate_median_pitch_hz(
        matched[:, -SAMPLE_RATE:],
        SAMPLE_RATE,
    )

    assert matched.shape == tts.shape
    assert opening_pitch is not None
    assert middle_pitch is not None
    assert closing_pitch is not None
    assert abs(opening_pitch - 200.0) < 5.0
    assert abs(middle_pitch - 200.0) < 3.0
    assert abs(closing_pitch - 200.0) < 5.0


def test_quality_fallback_does_not_reapply_rejected_pitch_shift(monkeypatch):
    tts = _tone(180.0)
    reference = _tone(240.0)
    scores = iter((4.1, 3.3))
    pitch_calls = 0

    def distort_pitch(waveform, _reference, _sample_rate):
        nonlocal pitch_calls
        pitch_calls += 1
        return waveform * 0.5

    monkeypatch.setattr(
        TTSPostProcessor,
        "_score_dnsmos",
        lambda *_args: next(scores),
    )
    monkeypatch.setattr(TTSPostProcessor, "_trim_digital_silence", lambda value, _sr: value)
    monkeypatch.setattr(
        TTSPostProcessor,
        "_compress_internal_silence",
        lambda value, _sr: value,
    )
    monkeypatch.setattr(TTSPostProcessor, "match_pitch", distort_pitch)
    monkeypatch.setattr(
        TTSPostProcessor,
        "match_boundary_pitch",
        lambda value, _reference, _sr: value,
    )
    monkeypatch.setattr(TTSPostProcessor, "_apply_edge_fades", lambda value, _sr: value)
    monkeypatch.setattr(
        TTSPostProcessor,
        "match_loudness",
        lambda value, _reference, _sr: value,
    )
    monkeypatch.setattr(
        TTSPostProcessor,
        "match_spectral_envelope",
        lambda value, _reference, _sr: value,
    )

    result = TTSPostProcessor.process(tts, reference, SAMPLE_RATE)

    assert pitch_calls == 1
    assert torch.equal(result, tts)


def test_quality_gate_keeps_live_pitch_candidate_when_dnsmos_is_acceptable(monkeypatch):
    tts = _tone(180.0)
    reference = _tone(240.0)
    scores = iter((4.1, 3.62))
    pitched = tts * 0.8

    monkeypatch.setattr(
        TTSPostProcessor,
        "_score_dnsmos",
        lambda *_args: next(scores),
    )
    monkeypatch.setattr(
        TTSPostProcessor,
        "_trim_digital_silence",
        lambda value, _sr: value,
    )
    monkeypatch.setattr(
        TTSPostProcessor,
        "match_pitch",
        lambda _value, _reference, _sr: pitched,
    )
    monkeypatch.setattr(
        TTSPostProcessor,
        "_apply_edge_fades",
        lambda value, _sr: value,
    )
    monkeypatch.setattr(
        TTSPostProcessor,
        "match_loudness",
        lambda value, _reference, _sr: value,
    )

    def fail_if_spectrally_equalized(*_args, **_kwargs):
        raise AssertionError("reference-cloned speech must not be spectrally equalized")

    monkeypatch.setattr(
        TTSPostProcessor,
        "match_spectral_envelope",
        fail_if_spectrally_equalized,
    )

    result = TTSPostProcessor.process(tts, reference, SAMPLE_RATE)

    assert torch.equal(result, pitched)


def test_post_processor_preserves_internal_sentence_pause(monkeypatch):
    pause = torch.zeros((1, SAMPLE_RATE // 2), dtype=torch.float32)
    tts = torch.cat([_tone(220.0, 0.25), pause, _tone(220.0, 0.25)], dim=1)

    def fail_if_compressed(*_args, **_kwargs):
        raise AssertionError("internal pause compression must not run")

    monkeypatch.setattr(
        TTSPostProcessor,
        "_compress_internal_silence",
        fail_if_compressed,
    )
    monkeypatch.setattr(TTSPostProcessor, "_score_dnsmos", lambda *_args: 0.0)
    monkeypatch.setattr(
        TTSPostProcessor,
        "_trim_digital_silence",
        lambda value, _sr: value,
    )
    monkeypatch.setattr(
        TTSPostProcessor,
        "match_pitch",
        lambda value, _reference, _sr: value,
    )
    monkeypatch.setattr(
        TTSPostProcessor,
        "match_boundary_pitch",
        lambda value, _reference, _sr: value,
    )
    monkeypatch.setattr(
        TTSPostProcessor,
        "_apply_edge_fades",
        lambda value, _sr: value,
    )
    monkeypatch.setattr(
        TTSPostProcessor,
        "match_loudness",
        lambda value, _reference, _sr: value,
    )
    monkeypatch.setattr(
        TTSPostProcessor,
        "match_spectral_envelope",
        lambda value, _reference, _sr: value,
    )

    result = TTSPostProcessor.process(tts, _tone(220.0), SAMPLE_RATE)

    assert torch.equal(result, tts)


def test_post_processor_does_not_compound_pitch_shift_at_boundaries(monkeypatch):
    tts = _tone(180.0, 3.0)
    reference = _tone(360.0, 3.0)

    def fail_if_boundary_shifted(*_args, **_kwargs):
        raise AssertionError("global pitch correction must not be applied twice")

    monkeypatch.setattr(
        TTSPostProcessor,
        "match_boundary_pitch",
        fail_if_boundary_shifted,
    )
    monkeypatch.setattr(TTSPostProcessor, "_score_dnsmos", lambda *_args: 0.0)
    monkeypatch.setattr(
        TTSPostProcessor,
        "_trim_digital_silence",
        lambda value, _sr: value,
    )
    monkeypatch.setattr(
        TTSPostProcessor,
        "_apply_edge_fades",
        lambda value, _sr: value,
    )
    monkeypatch.setattr(
        TTSPostProcessor,
        "match_loudness",
        lambda value, _reference, _sr: value,
    )
    monkeypatch.setattr(
        TTSPostProcessor,
        "match_spectral_envelope",
        lambda value, _reference, _sr: value,
    )

    result = TTSPostProcessor.process(tts, reference, SAMPLE_RATE)
    opening_pitch = TTSPostProcessor._estimate_median_pitch_hz(
        result[:, :SAMPLE_RATE],
        SAMPLE_RATE,
    )
    middle_pitch = TTSPostProcessor._estimate_median_pitch_hz(
        result[:, SAMPLE_RATE:2 * SAMPLE_RATE],
        SAMPLE_RATE,
    )

    assert opening_pitch is not None
    assert middle_pitch is not None
    assert abs(opening_pitch - middle_pitch) < 3.0
    assert 235.0 <= middle_pitch <= 245.0
