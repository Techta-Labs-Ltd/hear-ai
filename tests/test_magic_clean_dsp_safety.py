from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
import torch
import torchaudio

from hear.services.magic_clean.models import ContentMode
from hear.services.magic_clean.processing.dynamics import DynamicsProcessor
from hear.services.magic_clean.processing.mossformer import (
    MossFormer2Enhancer,
    MossFormerOutputError,
)
from hear.services.magic_clean.processing.noise import NoiseReducer
from hear.services.magic_clean.processing.silence import SilenceProcessor
from hear.services.magic_clean.processing.stems import StemSeparator


def test_compressor_does_not_fade_in_from_zero_state():
    processor = DynamicsProcessor(torch.device("cpu"))
    waveform = torch.full((2, 24_000), 0.1, dtype=torch.float32)

    result = processor.compress(waveform, 48_000, ContentMode.SPEECH)
    gain = result[0] / waveform[0]

    assert gain[0] > 0.5
    assert gain[0].item() == pytest.approx(gain[4_000].item(), abs=1e-6)
    torch.testing.assert_close(result[0], result[1])


def test_lookahead_limiter_enforces_oversampled_true_peak_ceiling():
    processor = DynamicsProcessor(torch.device("cpu"))
    waveform = torch.full((1, 4_096), 0.95)
    waveform[:, 1::2] *= -1

    result = processor.lookahead_limit(waveform, 48_000)
    oversampled = torchaudio.functional.resample(result, 48_000, 192_000)

    assert oversampled.abs().max() <= 10 ** (processor.TRUE_PEAK_DBTP / 20) + 1e-5


@pytest.mark.parametrize("sample_count", [1, 127, 1023, 1024, 1025, 8_193])
def test_disabled_spectral_suppression_is_exact_identity(sample_count):
    generator = torch.Generator().manual_seed(sample_count)
    waveform = torch.randn(2, sample_count, generator=generator)

    result = NoiseReducer().spectral_suppress(waveform, 44_100, strength=0)

    assert result is waveform
    torch.testing.assert_close(result, waveform, rtol=0, atol=0)


def test_mossformer_unloaded_bypass_is_an_immutable_exact_copy():
    waveform = torch.randn(2, 44_101)
    enhancer = MossFormer2Enhancer()

    result = enhancer.enhance(waveform, 44_100)

    assert result.data_ptr() != waveform.data_ptr()
    torch.testing.assert_close(result, waveform, rtol=0, atol=0)


def test_mossformer_rejects_length_drift_instead_of_returning_resampled_audio():
    enhancer = MossFormer2Enhancer()
    enhancer._cv = lambda audio, _unused: np.zeros(audio.shape[-1] - 1, dtype=np.float32)

    with pytest.raises(MossFormerOutputError, match="changed chunk length"):
        enhancer.enhance(torch.ones(1, 44_100), 44_100)


def test_mossformer_rejects_non_finite_model_output():
    enhancer = MossFormer2Enhancer()

    def non_finite(audio, _unused):
        output = np.asarray(audio[0], dtype=np.float32).copy()
        output[len(output) // 2] = np.nan
        return output

    enhancer._cv = non_finite

    with pytest.raises(MossFormerOutputError, match="non-finite"):
        enhancer.enhance(torch.ones(1, 48_000), 48_000)


def test_mossformer_preserves_stereo_rate_length_and_device_across_model_boundary():
    enhancer = MossFormer2Enhancer()
    enhancer._cv = lambda audio, _unused: np.asarray(audio[0], dtype=np.float32)
    waveform = torch.stack(
        (
            torch.linspace(-0.1, 0.1, 44_101),
            torch.linspace(0.2, -0.2, 44_101),
        )
    )

    result = enhancer.enhance(waveform, 44_100)

    assert result.shape == waveform.shape
    assert result.device == waveform.device
    assert result.dtype == waveform.dtype
    assert torch.isfinite(result).all()
    assert not torch.equal(result[0], result[1])


def test_silence_edit_protects_both_channels_and_is_stable_after_first_pass():
    sample_rate = 1_000
    waveform = torch.zeros(2, 6_000)
    waveform[0, 1_000:1_800] = 0.2
    # Speech exists only on the right channel in the second region.
    waveform[1, 4_200:5_000] = 0.2
    processor = SilenceProcessor()

    first = processor.detect_and_strip_silence(waveform, sample_rate)
    second = processor.detect_and_strip_silence(first, sample_rate)

    assert first.shape[1] < waveform.shape[1]
    assert first.shape == second.shape
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first[0].abs().max() == pytest.approx(0.2)
    assert first[1].abs().max() == pytest.approx(0.2)


def test_silence_edit_leaves_continuous_program_material_unchanged():
    timeline = torch.arange(48_000, dtype=torch.float32) / 48_000
    waveform = (0.05 * torch.sin(2 * torch.pi * 220 * timeline)).repeat(2, 1)

    result = SilenceProcessor().detect_and_strip_silence(waveform, 48_000)

    assert result is waveform


def test_silence_edit_retains_a_quiet_word_between_loud_regions():
    sample_rate = 1_000
    waveform = torch.zeros(1, 10_000)
    waveform[:, 500:2_000] = 1.0
    waveform[:, 4_900:5_100] = 0.01
    waveform[:, 8_000:9_500] = 1.0

    result = SilenceProcessor().detect_and_strip_silence(waveform, sample_rate)

    assert result.shape[1] < waveform.shape[1]
    assert int(torch.isclose(result, torch.tensor(0.01)).sum()) >= 200


def test_disk_backed_silence_edit_uses_source_and_enhanced_activity_union(tmp_path):
    sample_rate = 1_000
    source = np.zeros((10_000, 2), dtype=np.float32)
    source[500:2_000, 0] = 0.5
    source[4_900:5_100, 1] = 0.01
    source[8_000:9_500, 0] = 0.5
    enhanced = source.copy()
    enhanced[4_900:5_100, 1] = 0.003
    source_path = tmp_path / "source.wav"
    enhanced_path = tmp_path / "enhanced.wav"
    output_path = tmp_path / "edited.wav"
    sf.write(source_path, source, sample_rate, subtype="FLOAT")
    sf.write(enhanced_path, enhanced, sample_rate, subtype="FLOAT")

    output_samples = SilenceProcessor().detect_and_strip_silence_file(
        str(source_path),
        str(enhanced_path),
        str(output_path),
    )
    output, output_rate = sf.read(output_path, dtype="float32", always_2d=True)

    assert output_rate == sample_rate
    assert sf.info(output_path).format == "RF64"
    assert output.shape == (output_samples, 2)
    assert output_samples < source.shape[0]
    assert int(np.count_nonzero(np.isclose(output[:, 1], 0.003))) >= 200


@pytest.mark.parametrize("activity_seconds", [1, 5, 6])
def test_silence_edit_detects_sparse_activity(activity_seconds):
    sample_rate = 1_000
    waveform = torch.zeros(1, 60 * sample_rate)
    start = 20 * sample_rate
    waveform[:, start : start + activity_seconds * sample_rate] = 0.2

    result = SilenceProcessor().detect_and_strip_silence(waveform, sample_rate)

    assert activity_seconds * sample_rate <= result.shape[1] < waveform.shape[1]


def test_disk_backed_silence_edit_detects_sparse_activity(tmp_path):
    sample_rate = 1_000
    waveform = np.zeros((60 * sample_rate, 1), dtype=np.float32)
    waveform[20_000:21_000] = 0.2
    source_path = tmp_path / "sparse-source.wav"
    enhanced_path = tmp_path / "sparse-enhanced.wav"
    output_path = tmp_path / "sparse-edited.wav"
    sf.write(source_path, waveform, sample_rate, subtype="FLOAT")
    sf.write(enhanced_path, waveform, sample_rate, subtype="FLOAT")

    output_samples = SilenceProcessor().detect_and_strip_silence_file(
        str(source_path),
        str(enhanced_path),
        str(output_path),
    )

    assert 1_000 <= output_samples < waveform.shape[0]


@pytest.mark.parametrize(
    ("sample_rate", "sample_count", "expected_drift"),
    [(96_000, 11, 3), (192_000, 9, 5)],
)
def test_demucs_adapter_accepts_rate_derived_round_trip_drift(
    monkeypatch,
    sample_rate,
    sample_count,
    expected_drift,
):
    class FakeDemucs:
        samplerate = 44_100
        sources = ("vocals", "other")

    separator = StemSeparator(torch.device("cpu"))
    separator._demucs = FakeDemucs()
    waveform = torch.stack(
        (
            torch.linspace(-0.1, 0.1, sample_count),
            torch.linspace(0.1, -0.1, sample_count),
        )
    )
    model_rate = torchaudio.functional.resample(waveform, sample_rate, 44_100)
    round_trip = torchaudio.functional.resample(model_rate, 44_100, sample_rate)
    assert round_trip.shape[-1] - sample_count == expected_drift

    def identity_output(_model, model_input, **kwargs):
        assert kwargs["shifts"] == 0
        return model_input[:, None].repeat(1, 2, 1, 1)

    monkeypatch.setattr(
        "hear.services.magic_clean.processing.stems.apply_model",
        identity_output,
    )

    result = separator.separate(waveform, sample_rate)

    assert set(result) == {"vocals", "other"}
    for stem in result.values():
        assert stem.shape == waveform.shape
        assert torch.isfinite(stem).all()


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("rank", "invalid source tensor"),
        ("batch", "unexpected batch count"),
        ("sources", "unexpected source count"),
        ("channels", "unexpected channel count"),
        ("samples", "stem length unexpectedly"),
        ("non_finite", "non-finite model output"),
    ],
)
def test_demucs_adapter_rejects_invalid_model_output(monkeypatch, defect, message):
    class FakeDemucs:
        samplerate = 48_000
        sources = ("vocals", "other")

    separator = StemSeparator(torch.device("cpu"))
    separator._demucs = FakeDemucs()

    def invalid_output(_model, model_input, **_kwargs):
        output = model_input[:, None].repeat(1, 2, 1, 1)
        if defect == "rank":
            return output[0]
        if defect == "batch":
            return output.repeat(2, 1, 1, 1)
        if defect == "sources":
            return output[:, :1]
        if defect == "channels":
            return output[:, :, :1]
        if defect == "samples":
            return output[..., :-1]
        output[..., 0] = torch.nan
        return output

    monkeypatch.setattr(
        "hear.services.magic_clean.processing.stems.apply_model",
        invalid_output,
    )

    with pytest.raises(RuntimeError, match=message):
        separator.separate(torch.ones(2, 48_000), 48_000)


def test_demucs_adapter_rejects_catastrophic_stem_truncation(monkeypatch):
    class FakeDemucs:
        samplerate = 48_000
        sources = ("vocals", "other")

    separator = StemSeparator(torch.device("cpu"))
    separator._demucs = FakeDemucs()

    def truncated_output(_model, waveform, **kwargs):
        assert kwargs["shifts"] == 0
        channels = waveform.shape[1]
        truncated_samples = waveform.shape[-1] // 2
        return torch.ones(1, 2, channels, truncated_samples)

    monkeypatch.setattr(
        "hear.services.magic_clean.processing.stems.apply_model",
        truncated_output,
    )

    with pytest.raises(RuntimeError, match="stem length unexpectedly"):
        separator.separate(torch.ones(2, 48_000), 48_000)


def test_demucs_load_allowlists_supported_checkpoint_types(monkeypatch, tmp_path):
    observed_safe_globals = set()

    class FakeDemucs:
        def to(self, _device):
            return self

        def eval(self):
            return self

    def fake_get_model(model_name, *, repo):
        assert model_name == "htdemucs"
        assert repo == tmp_path
        observed_safe_globals.update(torch.serialization.get_safe_globals())
        return FakeDemucs()

    monkeypatch.setattr(
        "hear.services.magic_clean.processing.stems.get_model",
        fake_get_model,
    )

    separator = StemSeparator(torch.device("cpu"))
    separator.load("htdemucs", tmp_path)

    safe_names = {getattr(value, "__name__", "") for value in observed_safe_globals}
    serialized_names = {
        value[1]
        for value in observed_safe_globals
        if isinstance(value, tuple) and len(value) == 2
    }
    assert {"HTDemucs", "Fraction", "dtype"} <= safe_names
    assert "numpy.core.multiarray.scalar" in serialized_names
    assert separator.is_loaded
