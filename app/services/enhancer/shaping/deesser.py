import numpy as np
import torch
from numba import njit
from scipy import signal as scipy_signal

from app.services.enhancer.base import ProcessingStage
from app.services.enhancer.models import ProcessingContext


@njit(cache=True)
def _sibilance_envelope(sibilance: np.ndarray, attack_c: float, release_c: float) -> np.ndarray:
    env = np.empty_like(sibilance)
    prev = 0.0
    for i in range(len(sibilance)):
        c = attack_c if sibilance[i] > prev else release_c
        prev = c * prev + (1.0 - c) * sibilance[i]
        env[i] = prev
    return env


class DeEsser(ProcessingStage):
    name = "deesser"

    def __init__(self, config):
        self._c = config

    async def process(self, ctx: ProcessingContext) -> ProcessingContext:
        try:
            sr = ctx.audio.sample_rate
            w = ctx.audio.data
            s_freq = self._c.deesser_freq_hz
            s_q = 2.0
            theta = 2.0 * np.pi * s_freq / sr
            alpha_d = np.sin(theta) / (2.0 * s_q)
            a0 = 1.0 + alpha_d
            b = np.array([1.0 / a0, -2.0 * np.cos(theta) / a0, 1.0 / a0])
            a = np.array([1.0, -2.0 * np.cos(theta) / a0, (1.0 - alpha_d) / a0])
            sig = w.squeeze(0).cpu().numpy().astype(np.float64)

            sos_b, sos_a = scipy_signal.tf2sos(b, a)
            sibilance_sidechain = scipy_signal.sosfilt(sos_b, sos_a, sig)

            sibilance = np.abs(sibilance_sidechain)
            attack_c = np.exp(-1.0 / (sr * 1 / 1000.0))
            release_c = np.exp(-1.0 / (sr * 50 / 1000.0))
            threshold_lin = 10.0 ** (self._c.deesser_threshold_db / 20.0)
            reduction_lin = 10.0 ** (self._c.deesser_reduction_db / 20.0)

            env = _sibilance_envelope(sibilance, attack_c, release_c)

            gain = np.ones_like(env)
            over = env > threshold_lin
            if over.any():
                overshoot = (env[over] - threshold_lin) / (env[over] + 1e-10)
                gain[over] = 1.0 - overshoot * (1.0 - reduction_lin)
                gain = np.clip(gain, reduction_lin, 1.0)

            ctx.audio.data = torch.from_numpy((sig * gain).astype(np.float32)).unsqueeze(0).to(w.device)
        except Exception:
            pass
        return ctx
