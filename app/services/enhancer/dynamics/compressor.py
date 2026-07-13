import numpy as np
import torch

from app.services.enhancer.base import ProcessingStage
from app.services.enhancer.models import ProcessingContext


class Compressor(ProcessingStage):
    name = "compressor"

    def __init__(self, config):
        self._c = config

    async def process(self, ctx: ProcessingContext) -> ProcessingContext:
        try:
            sr = ctx.audio.sample_rate
            w = ctx.audio.data
            threshold_lin = 10.0 ** (self._c.compressor_threshold_db / 20.0)
            attack_coef = np.exp(-1.0 / (sr * self._c.compressor_attack_ms / 1000.0))
            release_coef = np.exp(-1.0 / (sr * self._c.compressor_release_ms / 1000.0))
            makeup_lin = 10.0 ** (self._c.compressor_makeup_db / 20.0)
            ratio = self._c.compressor_ratio

            sig_np = w.squeeze(0).cpu().numpy().astype(np.float64)
            abs_sig = np.abs(sig_np)

            env = np.empty_like(abs_sig)
            prev = abs_sig[0]
            env[0] = prev
            for i in range(1, len(abs_sig)):
                c = attack_coef if abs_sig[i] > prev else release_coef
                prev = c * prev + (1.0 - c) * abs_sig[i]
                env[i] = prev

            gain_np = np.ones_like(env)
            over = env > threshold_lin
            gain_np[over] = (
                (threshold_lin + (env[over] - threshold_lin) / ratio)
                / (env[over] + 1e-12)
            )
            gain_np = np.clip(gain_np, 0.0, 1.0)

            gain = torch.from_numpy(gain_np.astype(np.float32)).unsqueeze(0).to(w.device)
            out = w * gain * makeup_lin

            peak = out.abs().max().item()
            if peak > 0.99:
                out = out * (0.99 / peak)

            ctx.audio.data = out
        except Exception:
            pass
        return ctx
