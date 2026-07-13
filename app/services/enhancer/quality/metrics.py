import numpy as np
import pyloudnorm as pyln
import torch

from app.services.enhancer.models import AudioBuffer


class QualityMetrics:
    TARGET_LUFS = -16.0

    @staticmethod
    def compute_snr(raw: AudioBuffer, enhanced: AudioBuffer) -> float:
        r = raw.data.cpu()
        e = enhanced.data.cpu()
        n = min(r.shape[-1], e.shape[-1])
        r, e = r[..., :n], e[..., :n]
        sig_p = e.pow(2).mean().item()
        noi_p = (r - e).pow(2).mean().item() + 1e-10
        return 10 * np.log10(max(sig_p, 1e-10) / noi_p)

    @staticmethod
    def compute_lufs(buf: AudioBuffer) -> float:
        try:
            meter = pyln.Meter(buf.sample_rate)
            loudness = meter.integrated_loudness(buf.data.cpu().squeeze(0).numpy().astype(np.float64))
            return loudness if np.isfinite(loudness) else -99.0
        except Exception:
            rms = buf.data.pow(2).mean().sqrt().item()
            return float(20 * np.log10(rms + 1e-8))

    @staticmethod
    def compute_quality_score(snr_db: float, clipping: bool, lufs: float) -> float:
        snr_score = min(1.0, max(0.0, (snr_db + 5) / 40))
        lufs_score = 1.0 - min(1.0, abs(lufs - QualityMetrics.TARGET_LUFS) / 20)
        clip_pen = 0.3 if clipping else 0.0
        return round(max(0.0, snr_score * 0.6 + lufs_score * 0.4 - clip_pen), 3)
