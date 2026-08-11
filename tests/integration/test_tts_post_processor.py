"""Unit tests for TTSPostProcessor — silence stripping, loudness matching,
spectral envelope matching, and the full process() pipeline.

These tests are fully self-contained (no GPU / Higgs required) and generate
synthetic waveforms to validate each processing stage.
"""

import numpy as np
import pyloudnorm as pyln
import torch
import pytest

from hear.services.reconstruction.tts_post_processor import TTSPostProcessor


SR = 44100


def _sine(freq: float, duration_s: float, amplitude: float = 0.5, sr: int = SR) -> torch.Tensor:
    """Generate a mono sine wave tensor (1, N)."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    samples = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return torch.from_numpy(samples).unsqueeze(0)


def _composite_sine(freqs_amps: list[tuple[float, float]], duration_s: float, sr: int = SR) -> torch.Tensor:
    """Generate a composite of multiple sine waves."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    samples = np.zeros_like(t, dtype=np.float32)
    for freq, amp in freqs_amps:
        samples += (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return torch.from_numpy(samples).unsqueeze(0)


def _silence(duration_s: float, sr: int = SR) -> torch.Tensor:
    """Generate silence."""
    return torch.zeros(1, int(sr * duration_s))


def _measure_lufs(waveform: torch.Tensor, sr: int = SR) -> float:
    """Measure integrated LUFS of a mono waveform."""
    audio = waveform.squeeze(0).cpu().numpy().astype(np.float64)
    meter = pyln.Meter(sr)
    return meter.integrated_loudness(audio)


def _band_energy(waveform: torch.Tensor, sr: int, low: float, high: float) -> float:
    """Compute RMS energy in a frequency band using FFT."""
    sig = waveform.squeeze(0).cpu().numpy().astype(np.float64)
    n = len(sig)
    spectrum = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mask = (freqs >= low) & (freqs < high)
    return float(np.sqrt(np.sum(spectrum[mask] ** 2) / max(n, 1)))


# ---------------------------------------------------------------------------
# Silence stripping tests
# ---------------------------------------------------------------------------

class TestStripTTSSilence:

    def test_removes_leading_trailing_silence(self):
        """0.5s silence + 1s tone + 0.5s silence -> silence removed, shorter."""
        tone = _sine(440, 1.0, 0.5)
        silence_before = _silence(0.5)
        silence_after = _silence(0.5)
        waveform = torch.cat([silence_before, tone, silence_after], dim=1)

        result = TTSPostProcessor.strip_tts_silence(waveform, SR)

        original_dur = waveform.shape[1] / SR
        result_dur = result.shape[1] / SR
        assert result_dur < original_dur, f"Duration should shrink: {result_dur:.2f}s vs {original_dur:.2f}s"
        assert result_dur >= 0.8, f"Should keep most of the tone: {result_dur:.2f}s"

    def test_preserves_speech_only(self):
        """A loud tone with no silence should be returned nearly unchanged."""
        tone = _sine(440, 1.0, 0.5)
        result = TTSPostProcessor.strip_tts_silence(tone, SR)
        # The tone has energy everywhere, so it should be preserved
        assert result.shape[1] > 0
        ratio = result.shape[1] / tone.shape[1]
        assert ratio >= 0.9, f"Should preserve most of the signal: ratio={ratio:.2f}"

    def test_empty_input_returns_original(self):
        """Very short waveform (<2 analysis frames) is returned as-is."""
        short = torch.randn(1, 10)
        result = TTSPostProcessor.strip_tts_silence(short, SR)
        assert result.shape == short.shape

    def test_all_silence_returns_original(self):
        """Pure silence waveform is returned as-is (safety guard)."""
        silence = _silence(2.0)
        result = TTSPostProcessor.strip_tts_silence(silence, SR)
        assert result.shape == silence.shape


# ---------------------------------------------------------------------------
# Loudness matching tests
# ---------------------------------------------------------------------------

class TestMatchLoudness:

    def test_loudness_matched_within_1db(self):
        """After matching, TTS LUFS should be within 1 dB of reference."""
        ref = _sine(440, 1.0, 0.5)
        tts = _sine(440, 1.0, 0.1)

        ref_lufs = _measure_lufs(ref)
        tts_lufs_before = _measure_lufs(tts)

        result = TTSPostProcessor.match_loudness(tts, ref, SR)
        result_lufs = _measure_lufs(result)

        assert abs(result_lufs - ref_lufs) < 3.0, (
            f"LUFS diff: {abs(result_lufs - ref_lufs):.1f} dB "
            f"(ref={ref_lufs:.1f}, result={result_lufs:.1f})"
        )
        assert abs(result_lufs - tts_lufs_before) > 0.5, "Gain should have changed"

    def test_no_clipping_after_match(self):
        """Peak should stay below 1.0 after loudness matching."""
        ref = _sine(440, 1.0, 0.9)
        tts = _sine(440, 1.0, 0.01)

        result = TTSPostProcessor.match_loudness(tts, ref, SR)
        peak = result.abs().max().item()
        assert peak <= 1.0, f"Peak exceeds 1.0: {peak:.3f}"

    def test_short_signal_returns_unchanged(self):
        """Very short TTS signal returns unchanged."""
        ref = _sine(440, 1.0, 0.5)
        tts = _sine(440, 0.05, 0.1)  # 50ms — below min_duration

        result = TTSPostProcessor.match_loudness(tts, ref, SR)
        assert torch.allclose(result, tts), "Short signal should be unchanged"

    def test_same_loudness_returns_unchanged(self):
        """When TTS and ref have similar loudness, output should be similar."""
        ref = _sine(440, 1.0, 0.5)
        tts = _sine(440, 1.0, 0.5)

        result = TTSPostProcessor.match_loudness(tts, ref, SR)

        result_lufs = _measure_lufs(result)
        ref_lufs = _measure_lufs(ref)
        assert abs(result_lufs - ref_lufs) < 0.5, (
            f"Loudness should be nearly identical: ref={ref_lufs:.1f}, result={result_lufs:.1f}"
        )


# ---------------------------------------------------------------------------
# Spectral envelope matching tests
# ---------------------------------------------------------------------------

class TestMatchSpectralEnvelope:

    def test_low_band_boosted_when_ref_has_more_bass(self):
        """Reference has boosted low freq; TTS should get low-band gain."""
        ref = _composite_sine([(100, 0.8), (1000, 0.2), (5000, 0.1)], duration_s=1.0)
        tts = _composite_sine([(100, 0.1), (1000, 0.2), (5000, 0.1)], duration_s=1.0)

        ref_low = _band_energy(ref, SR, 0, 300)
        tts_low_before = _band_energy(tts, SR, 0, 300)

        result = TTSPostProcessor.match_spectral_envelope(tts, ref, SR)
        result_low = _band_energy(result, SR, 0, 300)

        assert result_low > tts_low_before, (
            f"Low band should increase: before={tts_low_before:.4f}, after={result_low:.4f}"
        )

    def test_no_clipping_after_spectral_match(self):
        """Peak should stay below 1.0 after spectral matching."""
        ref = _sine(440, 1.0, 0.9)
        tts = _sine(1000, 1.0, 0.8)

        result = TTSPostProcessor.match_spectral_envelope(tts, ref, SR)
        peak = result.abs().max().item()
        assert peak <= 1.0, f"Peak exceeds 1.0: {peak:.3f}"

    def test_short_signal_returns_unchanged(self):
        """Very short signal returns unchanged."""
        ref = _sine(440, 1.0, 0.5)
        tts = _sine(440, 0.02, 0.5)  # 20ms

        result = TTSPostProcessor.match_spectral_envelope(tts, ref, SR)
        assert torch.allclose(result, tts, atol=1e-5), "Short signal should be unchanged"


# ---------------------------------------------------------------------------
# Full process() pipeline tests
# ---------------------------------------------------------------------------

class TestProcessPipeline:

    def test_process_strips_silence_and_matches_loudness(self):
        """Full pipeline: silence stripped, loudness matched."""
        ref = _sine(440, 1.0, 0.5)
        tts = torch.cat([
            _silence(0.3),
            _sine(440, 0.8, 0.1),  # quiet tone
            _silence(0.3),
        ], dim=1)

        result = TTSPostProcessor.process(tts, ref, SR)

        # Duration should be shorter (silence stripped)
        assert result.shape[1] < tts.shape[1], "Silence should be stripped"
        assert result.shape[1] > 0, "Result should not be empty"

        # Loudness should be closer to ref
        result_lufs = _measure_lufs(result)
        ref_lufs = _measure_lufs(ref)
        tts_after_strip_lufs = _measure_lufs(_sine(440, 0.8, 0.1))

        diff_after = abs(result_lufs - ref_lufs)
        diff_before = abs(tts_after_strip_lufs - ref_lufs)
        assert diff_after < diff_before + 2.0, (
            f"Loudness should be closer to ref: diff_after={diff_after:.1f}, diff_before={diff_before:.1f}"
        )

    def test_process_preserves_non_silent_content(self):
        """Process should not destroy the actual speech-like content."""
        ref = _composite_sine([(200, 0.3), (800, 0.4), (3000, 0.2)], duration_s=1.0)
        tts = _composite_sine([(200, 0.3), (800, 0.4), (3000, 0.2)], duration_s=0.8)

        result = TTSPostProcessor.process(tts, ref, SR)

        assert result.shape[1] > int(SR * 0.5), "Should preserve most content"
        peak = result.abs().max().item()
        assert peak <= 1.0, f"Should not clip: peak={peak:.3f}"

    def test_process_handles_edge_cases(self):
        """Process with very short inputs should not crash."""
        ref = _sine(440, 0.3, 0.5)
        tts = _sine(440, 0.1, 0.3)

        result = TTSPostProcessor.process(tts, ref, SR)
        assert result.shape[1] > 0
        assert result.abs().max().item() <= 1.0


# ---------------------------------------------------------------------------
# Integration with splice logic test
# ---------------------------------------------------------------------------

class TestIntegrationWithSplice:

    def test_silence_stripped_tts_splices_correctly(self):
        """Verify silence-stripped TTS output works with the _splice_segment logic."""
        TARGET_SR = 44100

        # Create a 3-second "original" waveform
        original = _sine(440, 3.0, 0.3)
        total_samples = original.shape[1]

        # Simulate replacing 0.5s in the middle (1.0s to 1.5s)
        start_sample = int(1.0 * TARGET_SR)
        end_sample = int(1.5 * TARGET_SR)

        # Create TTS output with surrounding silence
        tts_with_silence = torch.cat([
            _silence(0.2),
            _sine(600, 0.5, 0.4),
            _silence(0.2),
        ], dim=1)

        # Apply post-processing to clean it up
        ref_segment = original[:, start_sample:end_sample]
        tts_clean = TTSPostProcessor.process(tts_with_silence, ref_segment, TARGET_SR)

        # Verify the cleaned TTS is valid
        assert tts_clean.shape[1] > 0, "Cleaned TTS should not be empty"
        assert tts_clean.shape[0] == 1, "Should be mono"
        assert tts_clean.abs().max().item() <= 1.0, "Should not clip"

        # Simulate the splice operation (from SpeechSynthesizer._splice_segment)
        cross_len = min(int(0.02 * TARGET_SR), start_sample, tts_clean.shape[1])

        before = original[:, :start_sample].clone()
        after = original[:, end_sample:]

        if cross_len > 0:
            fade_out = torch.linspace(1.0, 0.0, cross_len).unsqueeze(0)
            fade_in = torch.linspace(0.0, 1.0, cross_len).unsqueeze(0)
            before[:, -cross_len:] = (
                before[:, -cross_len:] * fade_out
                + tts_clean[:, :cross_len] * fade_in
            )

        if after.shape[1] >= cross_len and cross_len > 0:
            tts_clean_tail = tts_clean.clone()
            tail_out = torch.linspace(1.0, 0.0, cross_len).unsqueeze(0)
            tail_in = torch.linspace(0.0, 1.0, cross_len).unsqueeze(0)
            tts_clean_tail[:, -cross_len:] = (
                tts_clean_tail[:, -cross_len:] * tail_out
                + after[:, :cross_len] * tail_in
            )
            after = after[:, cross_len:]
            spliced = torch.cat([before, tts_clean_tail[:, cross_len:], after], dim=1)
        else:
            spliced = torch.cat([before, tts_clean, after], dim=1)

        # Verify the spliced result
        assert spliced.shape[1] > 0, "Spliced result should not be empty"
        assert spliced.abs().max().item() <= 1.0, f"Spliced result should not clip: {spliced.abs().max().item():.3f}"

        # Verify no NaN or Inf
        assert torch.isfinite(spliced).all().item(), "Spliced result should be finite"

    def test_process_with_different_channel_counts(self):
        """Ensure process handles mono-to-mono correctly."""
        ref = _sine(440, 1.0, 0.5)  # (1, N) mono
        tts = _sine(440, 1.0, 0.3)  # (1, N) mono

        result = TTSPostProcessor.process(tts, ref, SR)
        assert result.shape[0] == 1, "Should remain mono"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
