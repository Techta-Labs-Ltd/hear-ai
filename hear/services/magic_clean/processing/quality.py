import numpy as np
import pyloudnorm as pyln
import torch
import torchaudio

from hear.utils.audio_dsp import match_length

from .audio_io import AudioIO


class QualityMetrics:
    TARGET_LUFS = -16.0

    @staticmethod
    def compute_snr(raw: torch.Tensor, enhanced: torch.Tensor) -> float:
        raw, enhanced = match_length(raw.cpu(), enhanced.cpu())
        sig_p = enhanced.pow(2).mean().item()
        noi_p = (raw - enhanced).pow(2).mean().item() + 1e-10
        return 10 * np.log10(max(sig_p, 1e-10) / noi_p)

    @staticmethod
    def compute_lufs(w: torch.Tensor) -> float:
        try:
            meter = pyln.Meter(AudioIO.TARGET_SR)
            loudness = meter.integrated_loudness(w.cpu().squeeze(0).numpy().astype(np.float64))
            return loudness if np.isfinite(loudness) else -99.0
        except Exception:
            rms = w.pow(2).mean().sqrt().item()
            return float(20 * np.log10(rms + 1e-8))

    @staticmethod
    def compute_true_peak_db(w: torch.Tensor, sr: int) -> float:
        """Measure a conservative 4x oversampled true-peak estimate."""
        if w.numel() == 0:
            return -99.0
        try:
            oversampled = torchaudio.functional.resample(w.float(), sr, sr * 4)
            peak = oversampled.abs().max().item()
        except Exception:
            peak = w.abs().max().item()
        return float(20 * np.log10(peak + 1e-8))

    @staticmethod
    def compute_snr_estimate(w: torch.Tensor, sr: int) -> float:
        """Estimate speech-to-background level from robust short-time energy.

        This is intentionally a no-reference estimate. A value of ``0`` is the
        existing wire-compatible unavailable sentinel when the material has no
        useful energy contrast.
        """
        signal = w.detach().float().mean(dim=0).cpu()
        frame_samples = max(1, round(sr * 0.02))
        frame_count = signal.numel() // frame_samples
        if frame_count < 10:
            return 0.0
        frames = signal[: frame_count * frame_samples].reshape(frame_count, frame_samples)
        powers = frames.square().mean(dim=1).numpy()
        noise_power = float(np.percentile(powers, 20))
        signal_power = float(np.percentile(powers, 80))
        if signal_power <= 1e-10 or signal_power <= noise_power * 1.05:
            return 0.0
        return float(10 * np.log10(signal_power / max(noise_power, 1e-10)))

    @staticmethod
    def compute_quality_score(
        snr_db: float,
        clipping: bool,
        lufs: float,
        *,
        snr_available: bool = True,
    ) -> float:


        snr_score = min(1.0, max(0.0, (snr_db + 5) / 40)) if snr_available else 0.0
        lufs_score = 1.0 - min(1.0, abs(lufs - QualityMetrics.TARGET_LUFS) / 20)
        clip_pen = 0.3 if clipping else 0.0
        return round(max(0.0, snr_score * 0.6 + lufs_score * 0.4 - clip_pen), 3)
