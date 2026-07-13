import logging
import os
import threading

import numpy as np
import onnxruntime as ort
import torch
import torchaudio

from app.services.enhancer.models import AudioBuffer

logger = logging.getLogger(__name__)


class DNSMOSScorer:
    SR = 16000
    WINDOW_SAMPLES = 144160
    HOP_SECONDS = 4.5

    def __init__(self):
        self._session = None
        self._lock = threading.Lock()

    def load(self) -> bool:
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..",
            "models", "dnsmos", "sig_bak_ovr.onnx"
        )
        if not os.path.exists(model_path):
            model_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "..",
                "models", "dnsmos", "model_v8.onnx"
            )
        if not os.path.exists(model_path):
            alt = os.path.join(
                os.path.dirname(__file__), "..", "..", "..",
                "models", "dnsmos", "sig_bak_ovr.onnx"
            )
            if os.path.exists(alt):
                model_path = alt
        if not os.path.exists(model_path):
            logger.warning("DNSMOS model not found at %s", model_path)
            return False
        try:
            self._session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
            logger.info("DNSMOS loaded from %s", model_path)
            return True
        except Exception as e:
            logger.warning("DNSMOS load failed: %s", e)
            return False

    def score(self, audio: AudioBuffer) -> float:
        if self._session is None:
            return 0.0
        try:
            w = audio.data.float()
            sr = audio.sample_rate
            if sr != self.SR:
                w_16k = torchaudio.functional.resample(w, sr, self.SR)
            else:
                w_16k = w
            sig = w_16k.squeeze(0).cpu().numpy().astype(np.float32)
            window = self.WINDOW_SAMPLES
            hop = int(self.HOP_SECONDS * self.SR)
            scores = []
            for start in range(0, len(sig), hop):
                end = start + window
                if end > len(sig):
                    break
                seg = sig[start:end].copy().astype(np.float32)
                with self._lock:
                    ort_inputs = {self._session.get_inputs()[0].name: seg.reshape(1, -1)}
                    ort_outs = self._session.run(None, ort_inputs)
                if len(ort_outs) >= 3:
                    sig_mos = float(ort_outs[0][0][0])
                    bak_mos = float(ort_outs[1][0][0])
                    ovr_mos = float(ort_outs[2][0][0])
                    scores.append(ovr_mos)
                elif len(ort_outs) >= 1:
                    scores.append(float(ort_outs[0][0][0]))
            return float(np.mean(scores)) if scores else 0.0
        except Exception as e:
            logger.warning("DNSMOS scoring failed: %s", e)
            return 0.0
