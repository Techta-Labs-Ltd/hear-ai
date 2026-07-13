import logging
import threading

import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model
from torchaudio.functional import resample

from app.services.enhancer.base import ProcessingStage
from app.services.enhancer.models import ProcessingContext

logger = logging.getLogger(__name__)


class DemucsSeparator(ProcessingStage):
    name = "demucs"

    def __init__(self, config):
        self._c = config
        self._demucs = None
        self._lock = threading.Lock()

    def load(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._demucs = get_model("htdemucs")
            self._demucs.to(device)
            self._demucs.eval()
            self._ready = True
        except Exception as e:
            logger.warning("Demucs load failed: %s", e)
            self._ready = False

    async def process(self, ctx: ProcessingContext) -> ProcessingContext:
        if not self._ready or self._demucs is None:
            return ctx
        try:
            w = ctx.audio.data
            sr = ctx.audio.sample_rate
            original_len = w.shape[1]
            orig_energy = w.pow(2).mean().item()

            stereo = w.repeat(2, 1) if w.shape[0] == 1 else w
            resampled = resample(stereo, sr, self._demucs.samplerate)

            with self._lock:
                with torch.no_grad():
                    sources = apply_model(self._demucs, resampled[None], progress=False, device=resampled.device)[0]

            for i, name in enumerate(self._demucs.sources):
                if name == "vocals":
                    stem = resample(sources[i], self._demucs.samplerate, sr)
                    stem = stem.mean(dim=0, keepdim=True) if stem.shape[0] > 1 else stem
                    if stem.shape[1] < original_len:
                        pad = torch.zeros((1, original_len - stem.shape[1]), device=stem.device)
                        stem = torch.cat([stem, pad], dim=1)
                    stem = stem[:, :original_len]
                    vocal_energy = stem.pow(2).mean().item()
                    if orig_energy > 1e-8 and vocal_energy < 0.05 * orig_energy:
                        logger.warning(
                            "Demucs vocals energy (%.6f) is < 5%% of original (%.6f), skipping",
                            vocal_energy, orig_energy,
                        )
                        return ctx
                    ctx.audio = ctx.audio.clone()
                    ctx.audio.data = stem
                    break
        except Exception as e:
            logger.warning("Demucs process failed: %s", e)
        return ctx
