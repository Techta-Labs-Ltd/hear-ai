import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model
from .audio_io import AudioIO

import threading

class StemSeparator:
    def __init__(self, device: torch.device):
        self._demucs = None
        self._device = device
        self._lock = threading.Lock()

    def load(self, model_name: str):
        self._demucs = get_model(model_name)
        self._demucs.to(self._device)
        self._demucs.eval()

    @property
    def is_loaded(self) -> bool:
        return self._demucs is not None

    def separate(self, waveform: torch.Tensor, sr: int) -> dict[str, torch.Tensor]:
        original_len = waveform.shape[1]
        stereo       = waveform.repeat(2, 1) if waveform.shape[0] == 1 else waveform
        resampled    = AudioIO.resample(stereo, sr, self._demucs.samplerate)

        with self._lock:
            with torch.inference_mode():
                sources = apply_model(
                    self._demucs, resampled[None], progress=False,
                )[0]

        result = {}
        for i, name in enumerate(self._demucs.sources):
            stem = AudioIO.resample(sources[i], self._demucs.samplerate, AudioIO.TARGET_SR)
            stem = AudioIO.to_mono(stem)
            if stem.shape[1] < original_len:
                pad  = torch.zeros((1, original_len - stem.shape[1]), device=stem.device)
                stem = torch.cat([stem, pad], dim=1)
            result[name] = stem[:, :original_len]
        return result
