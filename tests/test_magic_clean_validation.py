from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from hear.services.magic_clean.processing.audio_io import AudioIO
from hear.services.magic_clean.processing.quality import QualityMetrics
from hear.services.magic_clean.processing.validation import AudioValidationError, AudioValidator


@pytest.fixture(scope="module")
def ffmpeg_path() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        pytest.skip("ffmpeg is required for deterministic delivered-MP3 validation tests")
    completed = subprocess.run(
        [executable, "-hide_banner", "-encoders"], capture_output=True, check=False, text=True
    )
    if completed.returncode != 0 or "libmp3lame" not in completed.stdout:
        pytest.skip("ffmpeg with the libmp3lame encoder is required for this test")
    return executable


def _tone(
    *,
    sample_rate: int = 44100,
    duration_seconds: float = 1.0,
    amplitude: float = 0.08,
    channels: int = 1,
) -> np.ndarray:
    sample_count = round(sample_rate * duration_seconds)
    time = np.arange(sample_count, dtype=np.float64) / sample_rate
    left = amplitude * np.sin(2 * np.pi * 997.0 * time)
    if channels == 1:
        return left.astype(np.float32)
    right = amplitude * np.sin(2 * np.pi * 613.0 * time + 0.3)
    return np.column_stack((left, right)).astype(np.float32)


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int = 44100) -> None:
    sf.write(path, audio, sample_rate, format="WAV", subtype="FLOAT")


def _encode_mp3(
    ffmpeg_path: str,
    source_path: Path,
    output_path: Path,
    *,
    channels: int | None = None,
    output_sample_rate: int | None = None,
) -> None:
    command = [
        ffmpeg_path,
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-i",
        str(source_path),
        "-c:a",
        "libmp3lame",
        "-b:a",
        "192k",
    ]
    if channels is not None:
        command.extend(("-ac", str(channels)))
    if output_sample_rate is not None:
        command.extend(("-ar", str(output_sample_rate)))
    command.append(str(output_path))
    subprocess.run(command, capture_output=True, check=True)


@pytest.mark.parametrize(
    "waveform",
    [
        torch.ones(8),
        torch.empty((0, 8)),
        torch.empty((1, 0)),
        torch.tensor([[float("nan")]]),
        torch.tensor([[float("inf")]]),
    ],
)
def test_validate_pcm_rejects_invalid_shape_empty_and_nonfinite(waveform: torch.Tensor) -> None:
    with pytest.raises(AudioValidationError):
        AudioValidator.validate_pcm(waveform, expected_samples=None, preserve_timeline=False)


def test_validate_pcm_requires_and_enforces_exact_preserved_length() -> None:
    waveform = torch.zeros((2, 100), dtype=torch.float32)
    AudioValidator.validate_pcm(waveform, expected_samples=100, preserve_timeline=True)
    AudioValidator.validate_pcm(waveform, expected_samples=None, preserve_timeline=False)
    with pytest.raises(AudioValidationError, match="Expected PCM sample count"):
        AudioValidator.validate_pcm(waveform, expected_samples=None, preserve_timeline=True)
    with pytest.raises(AudioValidationError, match="expected=99, actual=100"):
        AudioValidator.validate_pcm(waveform, expected_samples=99, preserve_timeline=True)


def test_mp3_duration_allowance_is_one_frame_or_30ms_and_never_proportional() -> None:
    assert AudioValidator.mp3_duration_tolerance_seconds(48000) == pytest.approx(0.03)
    assert AudioValidator.mp3_duration_tolerance_seconds(44100) == pytest.approx(0.03)
    assert AudioValidator.mp3_duration_tolerance_seconds(32000) == pytest.approx(0.036)
    assert AudioValidator.mp3_duration_tolerance_seconds(16000) == pytest.approx(0.036)
    with pytest.raises(ValueError, match="sample rate"):
        AudioValidator.mp3_duration_tolerance_seconds(0)


def test_unavailable_snr_sentinel_is_excluded_from_quality_score() -> None:
    metrics = QualityMetrics()
    unavailable = metrics.compute_quality_score(0.0, False, -16.0, snr_available=False)
    treated_as_measurement = metrics.compute_quality_score(0.0, False, -16.0, snr_available=True)
    assert unavailable == 0.4
    assert treated_as_measurement > unavailable


def test_delivered_mp3_returns_measurements_from_exact_decoded_artifact(
    tmp_path: Path, ffmpeg_path: str
) -> None:
    source_path = tmp_path / "stereo-source.wav"
    output_path = tmp_path / "delivered.mp3"
    _write_wav(source_path, _tone(duration_seconds=1.2, channels=2))
    _encode_mp3(ffmpeg_path, source_path, output_path)
    result = AudioValidator.validate_delivered_audio(
        str(source_path),
        str(output_path),
        cut_silence=False,
        expect_audible=True,
        metrics=QualityMetrics(),
    )
    decoded, decoded_sample_rate = AudioIO.load(str(output_path))
    decoded_peak = QualityMetrics.compute_true_peak_db(decoded, decoded_sample_rate)
    assert result.sample_rate == decoded_sample_rate
    assert result.channels == decoded.shape[0] == 2
    assert result.sample_count == decoded.shape[1]
    assert result.duration_seconds == pytest.approx(decoded.shape[1] / decoded_sample_rate)
    assert result.duration_delta_seconds <= AudioValidator.mp3_duration_tolerance_seconds(
        decoded_sample_rate
    )
    assert result.peak_db == pytest.approx(round(decoded_peak, 2), abs=0.05)
    assert result.peak_db < -10.0
    assert result.lufs < -10.0
    assert result.clipping_detected is False
    assert result.clipped_sample_count == 0
    assert np.isfinite(result.quality_score)


def test_delivered_mp3_rejects_duration_beyond_fixed_codec_allowance(
    tmp_path: Path, ffmpeg_path: str
) -> None:
    source_path = tmp_path / "source.wav"
    longer_path = tmp_path / "longer.wav"
    output_path = tmp_path / "longer.mp3"
    _write_wav(source_path, _tone(duration_seconds=1.0))
    _write_wav(longer_path, _tone(duration_seconds=1.08))
    _encode_mp3(ffmpeg_path, longer_path, output_path)
    with pytest.raises(AudioValidationError, match="duration mismatch"):
        AudioValidator.validate_delivered_audio(
            str(source_path),
            str(output_path),
            cut_silence=False,
            expect_audible=True,
            metrics=QualityMetrics(),
        )


def test_silence_edit_allows_delivery_resampling_rounding(tmp_path: Path, ffmpeg_path: str) -> None:
    source_rate = 96000
    source_samples = source_rate + 1
    source_path = tmp_path / "source-96k.wav"
    retained_path = tmp_path / "retained-96k.wav"
    output_path = tmp_path / "delivered-48k.mp3"
    source = _tone(sample_rate=source_rate, duration_seconds=source_samples / source_rate)
    _write_wav(source_path, source, source_rate)
    _write_wav(retained_path, source, source_rate)
    _encode_mp3(ffmpeg_path, retained_path, output_path, output_sample_rate=48000)
    result = AudioValidator.validate_delivered_audio(
        str(source_path),
        str(output_path),
        cut_silence=True,
        expect_audible=True,
        metrics=QualityMetrics(),
        retained_reference_path=str(retained_path),
    )
    assert result.duration_seconds > source_samples / source_rate
    assert result.duration_delta_seconds < AudioValidator.mp3_duration_tolerance_seconds(
        result.sample_rate
    )


def test_silence_edit_still_rejects_a_lengthened_retained_pcm_timeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = torch.full((1, 48000), 0.1)
    lengthened = torch.full((1, 48001), 0.1)
    decoded = {"source": source, "reference": lengthened, "output": lengthened}
    monkeypatch.setattr(AudioIO, "load", lambda path: (decoded[path], 48000))
    with pytest.raises(AudioValidationError, match="lengthened the retained mix"):
        AudioValidator.validate_delivered_audio(
            "source",
            "output",
            cut_silence=True,
            expect_audible=True,
            metrics=QualityMetrics(),
            retained_reference_path="reference",
        )


def test_delivered_mp3_rejects_missing_source_channel(tmp_path: Path, ffmpeg_path: str) -> None:
    source_path = tmp_path / "stereo.wav"
    output_path = tmp_path / "mono.mp3"
    _write_wav(source_path, _tone(channels=2))
    _encode_mp3(ffmpeg_path, source_path, output_path, channels=1)
    with pytest.raises(AudioValidationError, match="channel count"):
        AudioValidator.validate_delivered_audio(
            str(source_path),
            str(output_path),
            cut_silence=False,
            expect_audible=True,
            metrics=QualityMetrics(),
        )


def test_delivered_mp3_rejects_erased_audible_stereo_channel(
    tmp_path: Path, ffmpeg_path: str
) -> None:
    sample_rate = 48000
    source_path = tmp_path / "stereo-source.wav"
    erased_path = tmp_path / "erased-right.wav"
    output_path = tmp_path / "erased-right.mp3"
    source = _tone(sample_rate=sample_rate, duration_seconds=2.0, amplitude=0.08, channels=2)
    erased = source.copy()
    erased[:, 1] = 0.0
    _write_wav(source_path, source, sample_rate)
    _write_wav(erased_path, erased, sample_rate)
    _encode_mp3(ffmpeg_path, erased_path, output_path)
    with pytest.raises(AudioValidationError, match="audible source channel: channel=2"):
        AudioValidator.validate_delivered_audio(
            str(source_path),
            str(output_path),
            cut_silence=False,
            expect_audible=True,
            metrics=QualityMetrics(),
        )


def test_silent_source_and_delivery_are_valid_and_use_unavailable_sentinels(
    tmp_path: Path, ffmpeg_path: str
) -> None:
    source_path = tmp_path / "silent.wav"
    output_path = tmp_path / "silent.mp3"
    _write_wav(source_path, np.zeros(44100, dtype=np.float32))
    _encode_mp3(ffmpeg_path, source_path, output_path)
    result = AudioValidator.validate_delivered_audio(
        str(source_path),
        str(output_path),
        cut_silence=False,
        expect_audible=True,
        metrics=QualityMetrics(),
    )
    assert result.lufs == -99.0
    assert result.snr_db == 0.0
    assert result.quality_score == 0.0
    assert result.clipping_detected is False


def test_silent_source_cannot_turn_into_audible_delivery(tmp_path: Path, ffmpeg_path: str) -> None:
    source_path = tmp_path / "silent-source.wav"
    audible_path = tmp_path / "audible-output.wav"
    output_path = tmp_path / "audible-output.mp3"
    _write_wav(source_path, np.zeros(44100, dtype=np.float32))
    _write_wav(audible_path, _tone())
    _encode_mp3(ffmpeg_path, audible_path, output_path)
    with pytest.raises(AudioValidationError, match="silent source"):
        AudioValidator.validate_delivered_audio(
            str(source_path),
            str(output_path),
            cut_silence=False,
            expect_audible=True,
            metrics=QualityMetrics(),
        )


@pytest.mark.parametrize("sample_count", [19199, 19200])
def test_sub_r128_window_clip_uses_finite_unavailable_loudness(
    tmp_path: Path, ffmpeg_path: str, sample_count: int
) -> None:
    sample_rate = 48000
    duration_seconds = sample_count / sample_rate
    source_path = tmp_path / f"short-{sample_count}.wav"
    output_path = tmp_path / f"short-{sample_count}.mp3"
    _write_wav(
        source_path, _tone(sample_rate=sample_rate, duration_seconds=duration_seconds), sample_rate
    )
    _encode_mp3(ffmpeg_path, source_path, output_path)
    result = AudioValidator.validate_delivered_audio(
        str(source_path),
        str(output_path),
        cut_silence=False,
        expect_audible=True,
        metrics=QualityMetrics(),
    )
    assert np.isfinite([result.lufs, result.peak_db, result.quality_score]).all()
    if sample_count < 19200:
        assert result.lufs == -99.0


def test_silence_edit_cannot_delete_continuous_program_material(
    tmp_path: Path, ffmpeg_path: str
) -> None:
    source_path = tmp_path / "continuous-source.wav"
    shortened_path = tmp_path / "shortened.wav"
    output_path = tmp_path / "shortened.mp3"
    _write_wav(source_path, _tone(duration_seconds=10.0))
    _write_wav(shortened_path, _tone(duration_seconds=1.0))
    _encode_mp3(ffmpeg_path, shortened_path, output_path)
    with pytest.raises(AudioValidationError, match="removed protected source activity"):
        AudioValidator.validate_delivered_audio(
            str(source_path),
            str(output_path),
            cut_silence=True,
            expect_audible=True,
            metrics=QualityMetrics(),
        )


def test_silence_edit_allows_removing_real_silence(tmp_path: Path, ffmpeg_path: str) -> None:
    source_path = tmp_path / "sparse-source.wav"
    shortened_path = tmp_path / "retained-activity.wav"
    output_path = tmp_path / "retained-activity.mp3"
    source = np.zeros(441000, dtype=np.float32)
    source[176400:220500] = _tone(duration_seconds=1.0)
    _write_wav(source_path, source)
    _write_wav(shortened_path, _tone(duration_seconds=1.0))
    _encode_mp3(ffmpeg_path, shortened_path, output_path)
    result = AudioValidator.validate_delivered_audio(
        str(source_path),
        str(output_path),
        cut_silence=True,
        expect_audible=True,
        metrics=QualityMetrics(),
    )
    assert result.duration_seconds < 2.0


def test_intentionally_zero_delivery_is_allowed_but_unexpected_silence_fails(
    tmp_path: Path, ffmpeg_path: str
) -> None:
    source_path = tmp_path / "audible.wav"
    silent_path = tmp_path / "zero.wav"
    output_path = tmp_path / "zero.mp3"
    _write_wav(source_path, _tone())
    _write_wav(silent_path, np.zeros(44100, dtype=np.float32))
    _encode_mp3(ffmpeg_path, silent_path, output_path)
    result = AudioValidator.validate_delivered_audio(
        str(source_path),
        str(output_path),
        cut_silence=False,
        expect_audible=False,
        metrics=QualityMetrics(),
    )
    assert result.quality_score == 0.0
    with pytest.raises(AudioValidationError, match="audible source"):
        AudioValidator.validate_delivered_audio(
            str(source_path),
            str(output_path),
            cut_silence=False,
            expect_audible=True,
            metrics=QualityMetrics(),
        )


def test_injected_decoded_artifact_rejects_erased_audible_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = torch.full((2, 48000), 0.1)
    delivered = source.clone()
    delivered[1].zero_()
    monkeypatch.setattr(
        AudioIO, "load", lambda path: (source, 48000) if path == "source" else (delivered, 48000)
    )
    with pytest.raises(AudioValidationError, match="audible source channel: channel=2"):
        AudioValidator.validate_delivered_audio(
            "source", "output", cut_silence=False, expect_audible=True, metrics=QualityMetrics()
        )


def test_channel_peak_cannot_hide_near_total_rms_erasure(monkeypatch: pytest.MonkeyPatch) -> None:
    source = torch.full((2, 48000), 0.1)
    delivered = source.clone()
    delivered[1].zero_()
    delivered[1, 0] = 2e-05
    monkeypatch.setattr(
        AudioIO, "load", lambda path: (source, 48000) if path == "source" else (delivered, 48000)
    )
    with pytest.raises(AudioValidationError, match="audible source channel: channel=2"):
        AudioValidator.validate_delivered_audio(
            "source", "output", cut_silence=False, expect_audible=True, metrics=QualityMetrics()
        )


@pytest.mark.parametrize(
    "source",
    [torch.full((1, 48000), 5e-07), torch.nn.functional.pad(torch.tensor([[2e-05]]), (0, 47999))],
    ids=["rms-only", "peak-only"],
)
def test_channel_collapse_gate_accepts_identity_at_single_metric_floor(
    monkeypatch: pytest.MonkeyPatch, source: torch.Tensor
) -> None:
    monkeypatch.setattr(AudioIO, "load", lambda _path: (source, 48000))
    AudioValidator.validate_delivered_audio(
        "source", "output", cut_silence=False, expect_audible=True, metrics=QualityMetrics()
    )


def test_retained_mix_gate_accepts_intentional_source_channel_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = torch.full((2, 48000), 0.1)
    retained = source.clone()
    retained[0].zero_()
    decoded = {"source": source, "reference": retained, "output": retained}
    monkeypatch.setattr(AudioIO, "load", lambda path: (decoded[path], 48000))
    AudioValidator.validate_delivered_audio(
        "source",
        "output",
        cut_silence=False,
        expect_audible=False,
        metrics=QualityMetrics(),
        retained_reference_path="reference",
    )


def test_retained_mix_gate_rejects_erasure_with_source_gate_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = torch.full((2, 48000), 0.1)
    retained = source.clone()
    retained[0].zero_()
    delivered = torch.zeros_like(source)
    decoded = {"source": source, "reference": retained, "output": delivered}
    monkeypatch.setattr(AudioIO, "load", lambda path: (decoded[path], 48000))
    with pytest.raises(AudioValidationError, match="erased the retained mix"):
        AudioValidator.validate_delivered_audio(
            "source",
            "output",
            cut_silence=False,
            expect_audible=False,
            metrics=QualityMetrics(),
            retained_reference_path="reference",
        )


def test_decoded_artifact_rejects_clipped_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    source = torch.full((1, 48000), 0.1)
    delivered = source.clone()
    delivered[0, 123] = 1.0
    monkeypatch.setattr(
        AudioIO, "load", lambda path: (source, 48000) if path == "source" else (delivered, 48000)
    )
    with pytest.raises(AudioValidationError, match="clipped decoded samples"):
        AudioValidator.validate_delivered_audio(
            "source", "output", cut_silence=False, expect_audible=True, metrics=QualityMetrics()
        )


def test_decoded_artifact_rejects_true_peak_above_minus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = torch.full((1, 48000), 0.9)
    delivered = source.clone()
    monkeypatch.setattr(AudioIO, "load", lambda _path: (delivered, 48000))
    with pytest.raises(AudioValidationError, match="true-peak ceiling"):
        AudioValidator.validate_delivered_audio(
            "source", "output", cut_silence=False, expect_audible=True, metrics=QualityMetrics()
        )


def test_decoded_artifact_must_remain_finite(monkeypatch: pytest.MonkeyPatch) -> None:
    source = torch.zeros((1, 48000))
    delivered = source.clone()
    delivered[0, 0] = float("nan")
    monkeypatch.setattr(
        AudioIO, "load", lambda path: (source, 48000) if path == "source" else (delivered, 48000)
    )
    with pytest.raises(AudioValidationError, match="non-finite"):
        AudioValidator.validate_delivered_audio(
            "source", "output", cut_silence=False, expect_audible=False, metrics=QualityMetrics()
        )
