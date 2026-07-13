import logging
import threading

import numpy as np
import torch
import torchaudio

from app.services.enhancer.base import ProcessingStage
from app.services.enhancer.models import ProcessingContext

logger = logging.getLogger(__name__)


class MPSENetEnhancer(ProcessingStage):
    name = "mpsenet"

    SR = 16000

    def __init__(self, config):
        self._c = config
        self._model = None
        self._lock = threading.Lock()

    def _try_torchhub(self) -> bool:
        hub_repos = [
            "lx-ljl/MPSENET",
            "unilight/MP-SENet",
        ]
        for repo in hub_repos:
            try:
                self._model = torch.hub.load(repo, "model", trust_repo=True)
                self._model.eval()
                if torch.cuda.is_available():
                    self._model = self._model.to("cuda")
                logger.info("MP-SENet loaded from torch.hub/%s", repo)
                return True
            except Exception:
                continue
        return False

    def _try_onnx(self) -> bool:
        try:
            import onnxruntime as ort
            import os
            search_paths = [
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "models", "mpsenet", "model.onnx"),
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "checkpoints", "mpsenet", "model.onnx"),
            ]
            for model_path in search_paths:
                if os.path.exists(model_path):
                    self._onnx_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                    self._ready = True
                    logger.info("MP-SENet loaded from ONNX: %s", model_path)
                    return True
        except Exception:
            pass
        return False

    def load(self):
        if self._try_torchhub():
            self._ready = True
            return
        if self._try_onnx():
            self._ready = True
            return
        logger.warning("MP-SENet all load methods failed — residual cleanup disabled")

    async def process(self, ctx: ProcessingContext) -> ProcessingContext:
        if not self._ready:
            return ctx

        w = ctx.audio.data
        sr = ctx.audio.sample_rate
        original_len = w.shape[1]
        original_device = w.device

        if sr != self.SR:
            w_16k = torchaudio.functional.resample(w, sr, self.SR)
        else:
            w_16k = w

        try:
            if hasattr(self, "_onnx_session"):
                audio_np = w_16k.squeeze(0).cpu().numpy().astype(np.float32)
                audio_np = audio_np.reshape(1, 1, -1)
                ort_inputs = {self._onnx_session.get_inputs()[0].name: audio_np}
                ort_outs = self._onnx_session.run(None, ort_inputs)
                result = torch.from_numpy(ort_outs[0].squeeze(0)).float().unsqueeze(0)
            else:
                with self._lock:
                    with torch.no_grad():
                        result = self._model(w_16k.unsqueeze(0))
                if isinstance(result, (list, tuple)):
                    result = result[0]
                if result.dim() == 3:
                    result = result.squeeze(0)

            if sr != self.SR:
                result = torchaudio.functional.resample(result, self.SR, sr)

            result = result[:, :original_len]
            if result.shape[1] < original_len:
                pad = torch.zeros(1, original_len - result.shape[1])
                result = torch.cat([result, pad], dim=1)

            ctx.audio.data = result.to(original_device)
        except Exception as e:
            logger.warning("MP-SENet process failed: %s", e)

        return ctx
