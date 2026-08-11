import os
import torch
import torchaudio
import torchaudio.functional as F

class AudioIO:
    TARGET_SR = 44100

    @staticmethod
    def load(path: str) -> tuple[torch.Tensor, int]:
        if os.path.getsize(path) == 0:
            raise ValueError(f"Audio file is empty: {path}")
        for backend in ("soundfile", "ffmpeg", None):
            try:
                kwargs = {"backend": backend} if backend else {}
                waveform, sr = torchaudio.load(path, **kwargs)
                return waveform, sr
            except Exception:
                continue
        raise RuntimeError(f"Could not load audio from {path}.")

    @staticmethod
    def to_mono(w: torch.Tensor) -> torch.Tensor:
        return w.mean(dim=0, keepdim=True) if w.shape[0] > 1 else w

    @staticmethod
    def resample(w: torch.Tensor, orig: int, target: int) -> torch.Tensor:
        return F.resample(w, orig, target) if orig != target else w

    @staticmethod
    def detect_clipping(w: torch.Tensor) -> bool:
        return (w.abs() > 0.99).float().mean().item() > 0.001
