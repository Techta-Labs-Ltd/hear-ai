import warnings

import numpy as np
import pyloudnorm as pyln
import torch

from app.services.enhancer.base import ProcessingStage
from app.services.enhancer.models import ProcessingContext


class PeakNormalizer(ProcessingStage):
    name = "peak_normalize"

    def __init__(self, config):
        self._c = config

    async def process(self, ctx: ProcessingContext) -> ProcessingContext:
        w = ctx.audio.data
        peak = w.abs().max().item()
        if peak > 1e-8:
            target = 10 ** (self._c.limiter_ceiling_db / 20)
            ctx.audio.data = w * (target / peak)
        return ctx


class LoudnessNormalizer(ProcessingStage):
    name = "lufs_normalize"

    def __init__(self, config):
        self._c = config

    async def process(self, ctx: ProcessingContext) -> ProcessingContext:
        try:
            w = ctx.audio.data
            sr = ctx.audio.sample_rate
            original_device = w.device
            if w.shape[1] < int(sr * 0.5):
                ctx.audio.data = self._peak_normalise(w)
                return ctx
            meter = pyln.Meter(sr)
            audio_np = w.cpu().squeeze(0).numpy().astype(np.float64)
            loudness = meter.integrated_loudness(audio_np)
            if not np.isfinite(loudness) or loudness < -70.0 or loudness > 0.0:
                ctx.audio.data = self._peak_normalise(w)
                return ctx
            if abs(loudness - self._c.lufs_target) <= 0.5:
                return ctx
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                normalised = pyln.normalize.loudness(audio_np, loudness, self._c.lufs_target)
            if not np.isfinite(normalised).all():
                ctx.audio.data = self._peak_normalise(w)
                return ctx
            result = torch.from_numpy(normalised.astype(np.float32)).unsqueeze(0).to(original_device)
            ctx.audio.data = result
        except Exception:
            pass
        return ctx

    def _peak_normalise(self, w: torch.Tensor) -> torch.Tensor:
        original_device = w.device
        peak = w.abs().max().item()
        if peak < 1e-8:
            return w
        ceiling = 10 ** (self._c.limiter_ceiling_db / 20)
        result = w * (ceiling / peak)
        return result.to(original_device)