import numpy as np
import torch
from numba import njit

from app.services.enhancer.base import ProcessingStage
from app.services.enhancer.models import ProcessingContext


@njit(cache=True)
def _envelope_follower(sig: np.ndarray, atk: float, rel: float) -> np.ndarray:
    env = np.empty_like(sig)
    prev = 0.0
    for i in range(len(sig)):
        c = atk if np.abs(sig[i]) > prev else rel
        prev = c * prev + (1.0 - c) * np.abs(sig[i])
        env[i] = prev
    return env


@njit(cache=True)
def _compute_gain(is_click: np.ndarray, click_coef: float, recover_coef: float) -> np.ndarray:
    gain = np.ones(len(is_click))
    suppress = 1.0
    for i in range(len(is_click)):
        if is_click[i]:
            c = click_coef
            target = 0.3
        else:
            c = recover_coef
            target = 1.0
        suppress = c * suppress + (1.0 - c) * target
        gain[i] = suppress
        if is_click[i]:
            gain[i] = min(gain[i], 0.3)
    return gain.astype(np.float32)


class TransientSuppressor(ProcessingStage):
    name = "transient_suppress"

    def __init__(self, config):
        self._c = config

    async def process(self, ctx: ProcessingContext) -> ProcessingContext:
        try:
            w = ctx.audio.data
            sr = ctx.audio.sample_rate
            sig = w.squeeze(0).cpu().numpy().astype(np.float64)

            atk = np.exp(-1.0 / (sr * 0.001))
            rel = np.exp(-1.0 / (sr * 0.050))
            env = _envelope_follower(sig, atk, rel)

            frame = int(sr * 0.050)
            kernel = np.ones(frame) / frame
            sig2 = sig ** 2
            local_rms = np.sqrt(np.convolve(sig2, kernel, mode='same'))

            ratio = np.abs(sig) / (local_rms + 1e-10)
            is_click = ratio > 4.0

            if not is_click.any():
                return ctx

            click_coef = np.exp(-1.0 / (sr * 0.0005))
            recover_coef = np.exp(-1.0 / (sr * 0.010))
            gain = _compute_gain(is_click, click_coef, recover_coef)

            gain_t = torch.from_numpy(gain).unsqueeze(0).to(w.device)
            ctx.audio.data = w * gain_t
        except Exception:
            pass
        return ctx
