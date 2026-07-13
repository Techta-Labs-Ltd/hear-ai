import logging

import numpy as np
import pyloudnorm as pyln
import torch
import torchaudio

from .silence_processor import SilenceProcessor
from app.services.enhancer.quality.dnsmos_scorer import DNSMOSScorer

logger = logging.getLogger(__name__)


class TTSPostProcessor:
    """Post-processing pipeline for TTS output before splicing into the original audio.

    Cleans up reconstructed segments by stripping silence, matching loudness,
    and applying gentle spectral envelope matching so the TTS output blends
    naturally with the surrounding original audio.
    """


    TTS_PRE_SPEECH_PAD_MS = 50
    TTS_POST_SPEECH_PAD_MS = 50
    TTS_MERGE_GAP_MS = 100
    TTS_MIN_SEGMENT_MS = 50


    LOW_CUTOFF = 300
    HIGH_CUTOFF = 3000


    MAX_LOUDNESS_GAIN_DB = 6.0
    MAX_BAND_GAIN_DB = 3.0

    @staticmethod
    def strip_tts_silence(waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Strip leading/trailing silence from TTS output using adaptive energy detection.

        Uses tighter padding and merge thresholds than the general-purpose
        SilenceProcessor since TTS silence patterns are more predictable.
        """
        try:
            sp = SilenceProcessor()
            original_params = (
                sp.PRE_SPEECH_PAD_MS,
                sp.POST_SPEECH_PAD_MS,
                sp.SEGMENT_MERGE_GAP_MS,
                sp.MIN_SPEECH_SEGMENT_MS,
            )

            sp.PRE_SPEECH_PAD_MS = TTSPostProcessor.TTS_PRE_SPEECH_PAD_MS
            sp.POST_SPEECH_PAD_MS = TTSPostProcessor.TTS_POST_SPEECH_PAD_MS
            sp.SEGMENT_MERGE_GAP_MS = TTSPostProcessor.TTS_MERGE_GAP_MS
            sp.MIN_SPEECH_SEGMENT_MS = TTSPostProcessor.TTS_MIN_SEGMENT_MS

            try:
                result = sp.detect_and_strip_silence(waveform, sr)
            finally:
                (
                    sp.PRE_SPEECH_PAD_MS,
                    sp.POST_SPEECH_PAD_MS,
                    sp.SEGMENT_MERGE_GAP_MS,
                    sp.MIN_SPEECH_SEGMENT_MS,
                ) = original_params

            if result.shape[1] == 0:
                logger.warning("TTS silence strip produced empty output, returning original")
                return waveform

            return result
        except Exception as e:
            logger.warning("TTS silence strip failed: %s", e)
            return waveform

    @staticmethod
    def match_loudness(
        tts_waveform: torch.Tensor,
        ref_waveform: torch.Tensor,
        sr: int,
    ) -> torch.Tensor:
        """Match the integrated loudness (LUFS) of TTS output to the reference segment."""
        try:
            min_duration = 0.2
            if tts_waveform.shape[1] < int(sr * min_duration):
                return tts_waveform
            if ref_waveform.shape[1] < int(sr * min_duration):
                return tts_waveform

            meter = pyln.Meter(sr)

            tts_np = tts_waveform.squeeze(0).cpu().numpy().astype(np.float64)
            ref_np = ref_waveform.squeeze(0).cpu().numpy().astype(np.float64)

            tts_loudness = meter.integrated_loudness(tts_np)
            ref_loudness = meter.integrated_loudness(ref_np)

            if not (np.isfinite(tts_loudness) and np.isfinite(ref_loudness)):
                return tts_waveform
            if tts_loudness < -70.0 or ref_loudness < -70.0:
                return tts_waveform

            diff_db = ref_loudness - tts_loudness
            diff_db = max(-TTSPostProcessor.MAX_LOUDNESS_GAIN_DB, min(diff_db, TTSPostProcessor.MAX_LOUDNESS_GAIN_DB))

            gain = 10.0 ** (diff_db / 20.0)
            result = tts_waveform * gain

            peak = result.abs().max().item()
            if peak > 0.99:
                result = result * (0.99 / peak)

            logger.debug(
                "Loudness match: TTS %.1f LUFS -> ref %.1f LUFS (gain %.1f dB)",
                tts_loudness, ref_loudness, diff_db,
            )
            return result
        except Exception as e:
            logger.warning("Loudness match failed: %s", e)
            return tts_waveform

    @staticmethod
    def match_spectral_envelope(
        tts_waveform: torch.Tensor,
        ref_waveform: torch.Tensor,
        sr: int,
    ) -> torch.Tensor:
        """Apply 3-band EQ matching to nudge TTS timbre toward the reference.

        Bands: low (<300 Hz), mid (300-3000 Hz), high (>3000 Hz).
        Measures RMS energy per band in both signals and applies gentle gain
        corrections to the TTS output, clamped to MAX_BAND_GAIN_DB.
        """
        try:
            tts_np = tts_waveform.squeeze(0).cpu().numpy().astype(np.float64)
            ref_np = ref_waveform.squeeze(0).cpu().numpy().astype(np.float64)

            min_samples = int(sr * 0.05)
            if len(tts_np) < min_samples or len(ref_np) < min_samples:
                return tts_waveform

            low_cutoff = TTSPostProcessor.LOW_CUTOFF
            high_cutoff = TTSPostProcessor.HIGH_CUTOFF

            tts_rms = TTSPostProcessor._band_rms(tts_np, sr, low_cutoff, high_cutoff)
            ref_rms = TTSPostProcessor._band_rms(ref_np, sr, low_cutoff, high_cutoff)

            gain_db = np.zeros(3)
            max_gain = TTSPostProcessor.MAX_BAND_GAIN_DB
            for i in range(3):
                if tts_rms[i] < 1e-10 or ref_rms[i] < 1e-10:
                    gain_db[i] = 0.0
                    continue
                ratio = ref_rms[i] / tts_rms[i]
                ratio_db = 20.0 * np.log10(ratio)
                gain_db[i] = max(-max_gain, min(ratio_db, max_gain))

            result = TTSPostProcessor._apply_band_gains(tts_np, sr, low_cutoff, high_cutoff, gain_db)
            result_tensor = torch.from_numpy(result.astype(np.float32)).unsqueeze(0).to(tts_waveform.device)

            peak = result_tensor.abs().max().item()
            if peak > 0.99:
                result_tensor = result_tensor * (0.99 / peak)

            return result_tensor
        except Exception as e:
            logger.warning("Spectral envelope match failed: %s", e)
            return tts_waveform

    @staticmethod
    def _trim_digital_silence(waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Trim near-zero samples from start/end only."""
        try:
            sig = waveform.squeeze(0)
            abs_sig = sig.abs()
            peak = abs_sig.max().item()
            if peak < 1e-10:
                return waveform
            threshold = peak * 1e-4
            nonzero = torch.where(abs_sig > threshold)[0]
            if len(nonzero) == 0:
                return waveform
            first = nonzero[0].item()
            last = nonzero[-1].item() + 1
            pad = int(sr * 0.05)
            first = max(0, first - pad)
            last = min(waveform.shape[1], last + pad)
            trimmed = waveform[:, first:last]
            if trimmed.shape[1] < int(sr * 0.05):
                return waveform
            return trimmed
        except Exception as e:
            logger.warning("Digital silence trim failed: %s", e)
            return waveform

    @staticmethod
    def _compress_internal_silence(waveform: torch.Tensor, sr: int, max_gap_ms: float = 80.0) -> torch.Tensor:
        """Compress internal silence gaps to max_gap_ms so speech flows naturally."""
        try:
            sig = waveform.squeeze(0)
            frame_size = int(sr * 0.01)
            n_frames = sig.shape[0] // frame_size
            if n_frames < 4:
                return waveform

            peak = sig.abs().max().item()
            if peak < 1e-10:
                return waveform
            silence_thresh = peak * 0.01


            is_silent = []
            for i in range(n_frames):
                s = i * frame_size
                e = min(s + frame_size, sig.shape[0])
                is_silent.append(sig[s:e].abs().mean().item() < silence_thresh)


            sil_regions = []
            i = 0
            while i < n_frames:
                if is_silent[i]:
                    start_f = i
                    while i < n_frames and is_silent[i]:
                        i += 1
                    sil_regions.append((start_f * frame_size, i * frame_size))
                else:
                    i += 1

            if not sil_regions:
                return waveform


            max_sil_samples = int(sr * max_gap_ms / 1000.0)
            result_parts = []
            prev_end = 0
            for sil_s, sil_e in sil_regions:

                is_leading = sil_s == 0
                is_trailing = sil_e >= sig.shape[0] - frame_size
                if is_leading or is_trailing:

                    keep = min(int(sr * 0.05), sil_e - sil_s)
                    if is_leading:
                        result_parts.append(sig[sil_e - keep:sil_e].unsqueeze(0))
                    else:
                        result_parts.append(sig[sil_s:sil_s + keep].unsqueeze(0))
                    prev_end = sil_e
                    continue


                if sil_s > prev_end:
                    result_parts.append(sig[prev_end:sil_s].unsqueeze(0))


                sil_len = sil_e - sil_s
                if sil_len > max_sil_samples:

                    keep = max_sil_samples
                    half = keep // 2
                    compressed = torch.cat([sig[sil_s:sil_s+half], sig[sil_e-half:sil_e]]).unsqueeze(0)
                    result_parts.append(compressed)
                else:
                    result_parts.append(sig[sil_s:sil_e].unsqueeze(0))
                prev_end = sil_e


            if prev_end < sig.shape[0]:
                result_parts.append(sig[prev_end:].unsqueeze(0))

            if not result_parts:
                return waveform

            result = torch.cat(result_parts, dim=1)
            orig_dur = waveform.shape[1] / sr
            new_dur = result.shape[1] / sr
            if new_dur < orig_dur * 0.5:

                return waveform
            logger.info("Silence compressed: %.2fs -> %.2fs (%.0f%%)", orig_dur, new_dur, new_dur/orig_dur*100)
            return result
        except Exception as e:
            logger.warning("Internal silence compression failed: %s", e)
            return waveform

    @staticmethod
    def _apply_edge_fades(waveform: torch.Tensor, sr: int, fade_ms: float = 30.0) -> torch.Tensor:
        """Apply short cosine fade-in at start and fade-out at end for smooth splicing."""
        fade_len = int(sr * fade_ms / 1000.0)
        if waveform.shape[1] < fade_len * 2:
            return waveform
        result = waveform.clone()
        fade_in = 0.5 * (1.0 - torch.cos(torch.linspace(0, torch.pi, fade_len)))
        fade_out = fade_in.flip(0)
        result[0, :fade_len] *= fade_in
        result[0, -fade_len:] *= fade_out
        return result

    @staticmethod
    def _score_dnsmos(waveform: torch.Tensor, sr: int) -> float:
        try:
            scorer = DNSMOSScorer()
            if not scorer.load():
                return 0.0
            from app.services.enhancer.models import AudioBuffer
            audio_16k = torchaudio.functional.resample(waveform, sr, 16000) if sr != 16000 else waveform
            buf = AudioBuffer(data=audio_16k, sample_rate=16000)
            return scorer.score(buf)
        except Exception as e:
            logger.warning("DNSMOS scoring failed: %s", e)
            return 0.0

    @staticmethod
    def process(
        tts_waveform: torch.Tensor,
        ref_waveform: torch.Tensor,
        sr: int,
    ) -> torch.Tensor:
        """Run the full TTS post-processing pipeline with quality gating.

        Order: trim digital silence -> compress silence -> edge fades ->
        loudness match -> spectral envelope match.

        If DNSMOS drops more than 0.1 after processing, the raw TTS output
        is used instead to prevent quality degradation from over-processing.

        Args:
            tts_waveform: The raw TTS output waveform (1, N).
            ref_waveform: The original segment being replaced (1, M) used as
                reference for loudness and spectral matching.
            sr: Sample rate.

        Returns:
            Cleaned and matched TTS waveform ready for splicing.
        """
        tts_dur = tts_waveform.shape[1] / sr
        ref_dur = ref_waveform.shape[1] / sr
        tts_peak = tts_waveform.abs().max().item()
        ref_peak = ref_waveform.abs().max().item()

        raw_dnsmos = TTSPostProcessor._score_dnsmos(tts_waveform, sr)

        result = TTSPostProcessor._trim_digital_silence(tts_waveform, sr)
        result = TTSPostProcessor._compress_internal_silence(result, sr)
        result = TTSPostProcessor._apply_edge_fades(result, sr)

        result = TTSPostProcessor.match_loudness(result, ref_waveform, sr)

        result_peak_after_loud = result.abs().max().item()
        if result_peak_after_loud < 0.01 and tts_peak > 0.01:
            logger.warning(
                "Loudness match crushed TTS output (peak %.4f -> %.4f), reverting. ref_peak=%.4f ref_dur=%.2fs tts_dur=%.2fs",
                tts_peak, result_peak_after_loud, ref_peak, ref_dur, tts_dur,
            )
            result = tts_waveform

        result = TTSPostProcessor.match_spectral_envelope(result, ref_waveform, sr)

        final_peak = result.abs().max().item()
        if final_peak < 0.01 and tts_peak > 0.01:
            logger.warning(
                "Post-processing made TTS near-silent (peak %.4f -> %.4f), using raw TTS output",
                tts_peak, final_peak,
            )
            result = tts_waveform

        final_dnsmos = TTSPostProcessor._score_dnsmos(result, sr)
        if raw_dnsmos > 0.0 and final_dnsmos > 0.0 and raw_dnsmos - final_dnsmos > 0.1:
            logger.warning(
                "Post-processing degraded DNSMOS (%.2f -> %.2f, delta=%.2f), reverting to cleaned raw",
                raw_dnsmos, final_dnsmos, raw_dnsmos - final_dnsmos,
            )
            cleaned = TTSPostProcessor._trim_digital_silence(tts_waveform, sr)
            cleaned = TTSPostProcessor._apply_edge_fades(cleaned, sr)
            result = cleaned

        logger.info(
            "TTSPostProcessor: tts_dur=%.2fs ref_dur=%.2fs tts_peak=%.4f final_peak=%.4f final_dur=%.2fs dnsmos_raw=%.2f dnsmos_final=%.2f",
            tts_dur, ref_dur, tts_peak, result.abs().max().item(), result.shape[1]/sr,
            raw_dnsmos, final_dnsmos,
        )

        return result

    @staticmethod
    def _band_rms(
        signal: np.ndarray,
        sr: int,
        low_cutoff: int,
        high_cutoff: int,
    ) -> list[float]:
        """Compute RMS energy in three frequency bands using FFT."""
        n = len(signal)
        spectrum = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)

        low_mask = freqs < low_cutoff
        mid_mask = (freqs >= low_cutoff) & (freqs < high_cutoff)
        high_mask = freqs >= high_cutoff

        def band_rms(mask):
            energy = np.sum(spectrum[mask] ** 2)
            return float(np.sqrt(energy / max(n, 1)))

        return [band_rms(low_mask), band_rms(mid_mask), band_rms(high_mask)]

    @staticmethod
    def _apply_band_gains(
        signal: np.ndarray,
        sr: int,
        low_cutoff: int,
        high_cutoff: int,
        gain_db: np.ndarray,
    ) -> np.ndarray:
        """Apply per-band gain adjustments in the frequency domain."""
        n = len(signal)
        spectrum = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)

        low_mask = freqs < low_cutoff
        mid_mask = (freqs >= low_cutoff) & (freqs < high_cutoff)
        high_mask = freqs >= high_cutoff

        gain_lin = 10.0 ** (gain_db / 20.0)

        spectrum[low_mask] *= gain_lin[0]
        spectrum[mid_mask] *= gain_lin[1]
        spectrum[high_mask] *= gain_lin[2]

        return np.fft.irfft(spectrum, n=n)
