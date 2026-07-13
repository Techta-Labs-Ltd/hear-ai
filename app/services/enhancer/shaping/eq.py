import logging

import torch
import torchaudio.functional as F

from app.services.enhancer.base import ProcessingStage
from app.services.enhancer.models import ProcessingContext

logger = logging.getLogger(__name__)


class DCOffsetRemover(ProcessingStage):
    name = "dc_offset_remover"

    async def process(self, ctx: ProcessingContext) -> ProcessingContext:
        try:
            ctx.audio.data = F.highpass_biquad(ctx.audio.data, ctx.audio.sample_rate, 20.0)
        except Exception as e:
            logger.warning("DCOffsetRemover failed: %s", e)
        return ctx


class SpeechEQ(ProcessingStage):
    name = "speech_eq"

    def __init__(self, config):
        self._c = config

    async def process(self, ctx: ProcessingContext) -> ProcessingContext:
        try:
            w = ctx.audio.data
            sr = ctx.audio.sample_rate
            w = F.highpass_biquad(w, sr, cutoff_freq=self._c.eq_highpass_hz)
            w = F.bass_biquad(w, sr, gain=self._c.eq_bass_cut_db, central_freq=self._c.eq_bass_cut_hz)
            w = F.treble_biquad(w, sr, gain=self._c.eq_treble_boost_db, central_freq=self._c.eq_treble_boost_hz)
            w = F.lowpass_biquad(w, sr, cutoff_freq=16000.0)
            ctx.audio.data = w
        except Exception as e:
            logger.warning("SpeechEQ failed: %s", e)
        return ctx