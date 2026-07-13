import logging
import warnings

import numpy as np
import torch
import torchaudio.transforms as T

from app.services.enhancer.base import ProcessingStage
from app.services.enhancer.models import ProcessingContext

logger = logging.getLogger(__name__)


class VADGate(ProcessingStage):
    name = "vad_gate"

    def __init__(self, config):
        self._c = config
        self._vad_model = None
        self._vad_utils = None
        self._resamplers = {}

    def load(self):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._vad_model, self._vad_utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    trust_repo=True,
                    verbose=False,
                )
            self._ready = True
        except Exception as e:
            logger.warning("Silero VAD load failed: %s", e)
            self._ready = False

    async def process(self, ctx: ProcessingContext) -> ProcessingContext:
        if not self._ready or not self._vad_model or not self._vad_utils:
            return ctx
        try:
            w = ctx.audio.data
            sr = ctx.audio.sample_rate
            original_device = w.device
            get_speech_timestamps = self._vad_utils[0]

            w_cpu = w.cpu()
            w_16k = w_cpu
            if sr != 16000:
                if sr not in self._resamplers:
                    self._resamplers[sr] = T.Resample(sr, 16000)
                w_16k = self._resamplers[sr](w_cpu)

            wav_16k_mono = w_16k.mean(dim=0, keepdim=True).squeeze(0)

            speech_ts = get_speech_timestamps(
                wav_16k_mono,
                self._vad_model,
                sampling_rate=16000,
                threshold=self._c.vad_threshold,
                min_speech_duration_ms=100,
                min_silence_duration_ms=250,
            )

            n_samples = w.shape[1]
            ratio = sr / 16000.0
            fade_samples = int(sr * self._c.vad_fade_ms / 1000)

            gain = torch.full((n_samples,), self._c.vad_silence_gain, device=original_device, dtype=torch.float32)

            if speech_ts:
                for ts in speech_ts:
                    start_idx = max(0, int(ts['start'] * ratio) - fade_samples)
                    end_idx = min(n_samples, int(ts['end'] * ratio) + fade_samples)
                    gain[start_idx:end_idx] = self._c.vad_speech_gain

                for ts in speech_ts:
                    s_idx = max(0, int(ts['start'] * ratio) - fade_samples)
                    e_idx = min(n_samples, int(ts['end'] * ratio) + fade_samples)
                    if fade_samples > 0:
                        fade_in = torch.linspace(0, 1, min(fade_samples, n_samples - s_idx), device=original_device)
                        fade_out = torch.linspace(1, 0, min(fade_samples, e_idx), device=original_device)
                        if s_idx + len(fade_in) <= n_samples:
                            gain[s_idx:s_idx + len(fade_in)] = torch.max(
                                gain[s_idx:s_idx + len(fade_in)],
                                self._c.vad_uncertain_gain + (self._c.vad_speech_gain - self._c.vad_uncertain_gain) * fade_in,
                            )
                        if e_idx - len(fade_out) >= 0:
                            gain[e_idx - len(fade_out):e_idx] = torch.max(
                                gain[e_idx - len(fade_out):e_idx],
                                self._c.vad_uncertain_gain + (self._c.vad_speech_gain - self._c.vad_uncertain_gain) * fade_out.flip(0),
                            )
            else:
                gain[:] = self._c.vad_silence_gain

            if ctx.event_mask is not None:
                event_gain = 1.0 - (1.0 - self._c.vad_silence_gain) * (1.0 - ctx.event_mask)
                gain = torch.min(gain, event_gain.squeeze(0))

            gain = gain.clamp(self._c.vad_silence_gain, self._c.vad_speech_gain)
            ctx.audio.data = w * gain.unsqueeze(0)
        except Exception as e:
            logger.warning("VAD gate failed: %s", e)
        return ctx
