import logging
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torchaudio

from .audio_buffer import AudioBuffer

logger = logging.getLogger(__name__)


class DNSMOSScorer:
    SR = 16000
    WINDOW_SAMPLES = 144160
    HOP_SECONDS = 4.5

    def __init__(self):
        self._session = None
        self._lock = threading.Lock()

    def load(self) -> bool:
        model_directory = Path(__file__).resolve().parents[3] / "models" / "dnsmos"
        model_path = model_directory / "sig_bak_ovr.onnx"
        if not model_path.exists():
            model_path = model_directory / "model_v8.onnx"
        if not model_path.exists():
            logger.warning("DNSMOS model not found at %s", model_path)
            return False
        try:
            self._session = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            logger.info("DNSMOS loaded from %s", model_path)
            return True
        except Exception as exc:
            logger.warning("DNSMOS load failed: %s", exc)
            return False

    def score(self, audio: AudioBuffer) -> float:
        if self._session is None:
            return 0.0
        try:
            waveform = audio.data.float()
            if audio.sample_rate != self.SR:
                waveform = torchaudio.functional.resample(
                    waveform, audio.sample_rate, self.SR
                )
            signal = waveform.squeeze(0).cpu().numpy().astype(np.float32)
            hop = int(self.HOP_SECONDS * self.SR)
            scores = []
            for start in range(0, len(signal), hop):
                segment = signal[start:start + self.WINDOW_SAMPLES]
                if len(segment) < self.WINDOW_SAMPLES:
                    break
                with self._lock:
                    input_name = self._session.get_inputs()[0].name
                    outputs = self._session.run(
                        None, {input_name: segment.copy().reshape(1, -1)}
                    )
                output_index = 2 if len(outputs) >= 3 else 0
                scores.append(float(outputs[output_index][0][0]))
            return float(np.mean(scores)) if scores else 0.0
        except Exception as exc:
            logger.warning("DNSMOS scoring failed: %s", exc)
            return 0.0
