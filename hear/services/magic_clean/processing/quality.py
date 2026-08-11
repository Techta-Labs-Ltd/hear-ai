import numpy as np
import pyloudnorm as pyln
import torch
from .helpers import match_length
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
            meter    = pyln.Meter(AudioIO.TARGET_SR)
            loudness = meter.integrated_loudness(w.cpu().squeeze(0).numpy().astype(np.float64))
            return loudness if np.isfinite(loudness) else -99.0
        except Exception:
            rms = w.pow(2).mean().sqrt().item()
            return float(20 * np.log10(rms + 1e-8))

    @staticmethod
    def compute_quality_score(snr_db: float, clipping: bool, lufs: float) -> float:
        snr_score  = min(1.0, max(0.0, (snr_db + 5) / 40))
        lufs_score = 1.0 - min(1.0, abs(lufs - QualityMetrics.TARGET_LUFS) / 20)
        clip_pen   = 0.3 if clipping else 0.0
        return round(max(0.0, snr_score * 0.6 + lufs_score * 0.4 - clip_pen), 3)
