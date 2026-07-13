import logging
import threading

import numpy as np
import torch
import torchaudio

from app.services.enhancer.base import ProcessingStage
from app.services.enhancer.models import ProcessingContext

logger = logging.getLogger(__name__)


class DeepFilterEnhancer(ProcessingStage):
    name = "deepfilter"

    SR = 48000
    CHUNK_SECONDS = 30
    OVERLAP_SECONDS = 3.0
    WET_DRY = 0.85

    def __init__(self, config):
        self._c = config
        self._dfn = None
        self._state = None
        self._lock = threading.Lock()

    def load(self):
        try:
            from df import init_df
            import inspect as _inspect
            model, self._state, _ = init_df()
            self._dfn_mod = _inspect.getmodule(init_df)
            self._dfn = model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._ready = True
        except Exception as e:
            logger.warning("DeepFilterNet load failed: %s", e)
            self._ready = False

    def _process_chunk(self, w_chunk: torch.Tensor, sr: int, original_len: int) -> torch.Tensor:
        if sr != self.SR:
            w_48k = torchaudio.functional.resample(w_chunk, sr, self.SR)
        else:
            w_48k = w_chunk

        w_cpu = w_48k.cpu()
        torch.cuda.empty_cache()
        use_gpu = torch.cuda.is_available()

        from df import enhance as df_enhance

        try:
            with self._lock:
                if use_gpu:
                    model = self._dfn.to("cuda")
                else:
                    model = self._dfn
                with torch.no_grad():
                    atten_lim = getattr(self._c, "deepfilter_atten_lim_db", 20.0)
                    result = df_enhance(model, self._state, w_cpu, atten_lim_db=atten_lim)
                if use_gpu:
                    self._dfn = model
        except RuntimeError as e:
            if use_gpu and "out of memory" in str(e).lower():
                logger.warning("DeepFilterNet GPU OOM, falling back to CPU")
                torch.cuda.empty_cache()
                self._dfn = self._dfn.to("cpu")
                original_get_device = getattr(self._dfn_mod, "get_device", None)
                self._dfn_mod.get_device = lambda: "cpu"
                try:
                    with self._lock:
                        with torch.no_grad():
                            atten_lim = getattr(self._c, "deepfilter_atten_lim_db", 20.0)
                            result = df_enhance(self._dfn, self._state, w_cpu, atten_lim_db=atten_lim)
                finally:
                    if original_get_device:
                        self._dfn_mod.get_device = original_get_device
            else:
                raise

        if isinstance(result, np.ndarray):
            result = torch.from_numpy(result.copy()).float()
        if result.dim() == 1:
            result = result.unsqueeze(0)

        if sr != self.SR:
            result = torchaudio.functional.resample(result, self.SR, sr)

        result = result[:, :original_len]
        if result.shape[1] < original_len:
            pad = torch.zeros(1, original_len - result.shape[1])
            result = torch.cat([result, pad], dim=1)

        return result

    async def process(self, ctx: ProcessingContext) -> ProcessingContext:
        if not self._ready or self._dfn is None:
            return ctx

        w = ctx.audio.data
        sr = ctx.audio.sample_rate
        n = w.shape[1]
        chunk_samples = int(self.CHUNK_SECONDS * sr)
        overlap_samples = int(self.OVERLAP_SECONDS * sr)

        if n <= chunk_samples:
            result = self._process_chunk(w, sr, n)
            # Wet/dry blend to preserve natural voice characteristics
            blended = self.WET_DRY * result.to(w.device) + (1.0 - self.WET_DRY) * w
            ctx.audio.data = blended
            return ctx

        chunks = []
        pos = 0
        while pos < n:
            end = min(pos + chunk_samples, n)
            chunk = w[:, pos:end]
            processed = self._process_chunk(chunk, sr, chunk.shape[1])
            chunks.append(processed.to(w.device))
            pos += chunk_samples - overlap_samples

        if len(chunks) == 1:
            ctx.audio.data = chunks[0]
        else:
            blended = []
            fade = torch.linspace(0, 1, overlap_samples, device=w.device).view(1, -1)
            for i, chunk in enumerate(chunks):
                if i == 0:
                    blended.append(chunk)
                else:
                    prev = blended[-1]
                    overlap_prev = prev[:, -overlap_samples:]
                    overlap_cur = chunk[:, :overlap_samples]
                    blended[-1] = prev[:, :-overlap_samples]
                    blended.append(overlap_prev * (1 - fade) + overlap_cur * fade)
                    blended.append(chunk[:, overlap_samples:])
            blended_data = torch.cat(blended, dim=1)[:, :n]
            ctx.audio.data = self.WET_DRY * blended_data + (1.0 - self.WET_DRY) * w

        return ctx
