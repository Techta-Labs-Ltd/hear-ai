import numpy as np
import torch
from numba import njit
from scipy.ndimage import maximum_filter1d

from app.services.enhancer.base import ProcessingStage
from app.services.enhancer.models import ProcessingContext


@njit(cache=True)
def _compute_gain(peak_lookahead: np.ndarray, ceiling: float, release_coef: float) -> np.ndarray:
    n = len(peak_lookahead)
    gain = np.ones(n, dtype=np.float64)
    prev_g = 1.0
    for i in range(n):
        g_needed = ceiling / (peak_lookahead[i] + 1e-10)
        g_needed = min(g_needed, 1.0)
        if g_needed < prev_g:
            prev_g = g_needed
        else:
            prev_g = release_coef * prev_g + (1.0 - release_coef) * g_needed
        gain[i] = prev_g
    return gain


def _soft_clip(x: torch.Tensor, ceiling: float, threshold_ratio: float = 0.9) -> torch.Tensor:
    result = x.clone()
    threshold = threshold_ratio * ceiling
    above = x.abs() > threshold
    if above.any():
        above_vals = x[above]
        t = ((above_vals.abs() - threshold) / (ceiling - threshold + 1e-10)).clamp(0.0, 1.0)
        soft = threshold + (ceiling - threshold) * (3.0 * t * t - 2.0 * t * t * t)
        result[above] = soft * above_vals.sign()
    return result


class LookaheadLimiter(ProcessingStage):
    name = "lookahead_limiter"

    def __init__(self, config):
        self._c = config

    async def process(self, ctx: ProcessingContext) -> ProcessingContext:
        try:
            sr = ctx.audio.sample_rate
            w = ctx.audio.data
            ceiling = 10 ** (self._c.limiter_ceiling_db / 20)
            threshold_ratio = getattr(self._c, "limiter_soft_clip_threshold", 0.9)
            lookahead_samples = int(sr * self._c.limiter_lookahead_ms / 1000)
            release_coef = np.exp(-1.0 / (sr * self._c.limiter_release_ms / 1000))
            sig_np = w.squeeze(0).cpu().numpy().astype(np.float64)

            abs_sig = np.abs(sig_np)

            if lookahead_samples < 1:
                peak_env = abs_sig
            else:
                padded = np.pad(abs_sig, (0, lookahead_samples), mode="edge")
                from numpy.lib.stride_tricks import sliding_window_view
                windows = sliding_window_view(padded, lookahead_samples + 1)
                peak_env = windows.max(axis=-1)

            gain = _compute_gain(peak_env, ceiling, release_coef)
            gain_tensor = torch.from_numpy(gain.astype(np.float32)).unsqueeze(0).to(w.device)

            x = w * gain_tensor
            ctx.audio.data = _soft_clip(x, ceiling, threshold_ratio)
        except Exception:
            pass
        return ctx