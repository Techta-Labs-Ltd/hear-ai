import logging
import threading

import numpy as np
import torch
import torchaudio

from app.services.enhancer.base import ProcessingStage
from app.services.enhancer.models import ProcessingContext, SoundEvent

logger = logging.getLogger(__name__)


EVENT_CLASSES = {
    "cough": [289],
    "throat_clearing": [291],
    "click": [312, 313],
    "pop": [302],
    "applause": [396, 397],
}


class YAMNetClassifier(ProcessingStage):
    name = "yamnet"

    SR = 16000
    HOP_SECONDS = 0.48
    FRAME_SECONDS = 0.96

    def __init__(self, config):
        self._c = config
        self._model = None
        self._lock = threading.Lock()

    def _try_torchhub(self) -> bool:
        hub_repos = [
            "turian/yamnet",
            "thelou1s/yamnet",
        ]
        for repo in hub_repos:
            try:
                self._model = torch.hub.load(repo, "yamnet", trust_repo=True)
                self._model.eval()
                logger.info("YAMNet loaded from torch.hub/%s", repo)
                return True
            except Exception:
                continue
        return False

    def _try_onnx(self) -> bool:
        try:
            import os
            import onnxruntime as ort
            search_paths = [
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "models", "yamnet", "yamnet.onnx"),
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "checkpoints", "yamnet", "yamnet.onnx"),
            ]
            for model_path in search_paths:
                if os.path.exists(model_path):
                    self._onnx = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
                    self._onnx_input_name = self._onnx.get_inputs()[0].name
                    logger.info("YAMNet loaded from ONNX: %s", model_path)
                    return True
                logger.debug("YAMNet ONNX not found at %s", model_path)
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
        logger.info("YAMNet not available — event classification disabled")

    async def process(self, ctx: ProcessingContext) -> ProcessingContext:
        if not self._ready:
            return ctx

        w = ctx.audio.data
        sr = ctx.audio.sample_rate
        n = w.shape[1]
        device = w.device

        if sr != self.SR:
            w_16k = torchaudio.functional.resample(w, sr, self.SR)
        else:
            w_16k = w

        hop = int(self.HOP_SECONDS * self.SR)
        frame_len = int(self.FRAME_SECONDS * self.SR)
        n_frames = max(1, (w_16k.shape[1] - frame_len) // hop + 1)

        try:
            if hasattr(self, "_onnx"):
                audio_np = w_16k.squeeze(0).cpu().numpy().astype(np.float32)
                ort_inputs = {self._onnx_input_name: audio_np}
                ort_outs = self._onnx.run(None, ort_inputs)
                scores = ort_outs[0]
                if scores.ndim == 2 and scores.shape[0] < n_frames:
                    pad_width = ((0, n_frames - scores.shape[0]), (0, 0))
                    scores = np.pad(scores, pad_width, mode="edge")
                elif scores.ndim == 2 and scores.shape[0] > n_frames:
                    scores = scores[:n_frames]
            else:
                with self._lock:
                    with torch.no_grad():
                        scores_np, _ = self._model(w_16k.unsqueeze(0))
                scores = scores_np.squeeze(0).cpu().numpy()
                if scores.shape[0] > n_frames:
                    scores = scores[:n_frames]

            event_mask = torch.ones(n, dtype=torch.float32, device=device)
            events = []
            target_classes = set()
            for cls_ids in EVENT_CLASSES.values():
                target_classes.update(cls_ids)

            for frame_idx in range(scores.shape[0]):
                frame_scores = scores[frame_idx]
                for event_name, cls_ids in EVENT_CLASSES.items():
                    max_score = max(frame_scores[cid] for cid in cls_ids if cid < len(frame_scores))
                    if max_score > 0.3:
                        start_s = frame_idx * hop
                        end_s = start_s + hop
                        sr_ratio = sr / self.SR
                        s = int(start_s * sr_ratio)
                        e = min(int(end_s * sr_ratio), n)
                        event_mask[s:e] = 0.05

            fade = int(0.01 * sr)
            fade_in = torch.linspace(0, 1, fade, device=device)
            fade_out = torch.linspace(1, 0, fade, device=device)
            suppressed = (event_mask < 0.5).nonzero(as_tuple=True)[0]
            if len(suppressed) > 0:
                s_groups = torch.where(torch.diff(suppressed) > 1)[0] + 1
                regions = torch.tensor_split(suppressed, s_groups.tolist())
                for reg in regions:
                    if len(reg) < int(0.02 * sr):
                        event_mask[reg[0]:reg[-1] + 1] = 1.0
                        continue
                    s = max(0, int(reg[0]) - fade)
                    e = min(n, int(reg[-1]) + fade + 1)
                    event_mask[s:reg[0]] = fade_in[:reg[0] - s] if reg[0] > s else fade_in[:0]
                    event_mask[reg[-1] + 1:e] = fade_out[:e - reg[-1] - 1] if e > reg[-1] + 1 else fade_out[:0]
                    events.append(SoundEvent(
                        start_sample=int(reg[0]),
                        end_sample=int(reg[-1]) + 1,
                        confidence=0.7,
                    ))

            ctx.event_mask = event_mask.unsqueeze(0)
            ctx.events = events
        except Exception as e:
            logger.warning("YAMNet process failed: %s", e)

        return ctx
