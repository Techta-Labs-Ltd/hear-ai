import os

import torch
import torchaudio
import torchaudio.functional as F

from app.services.enhancer.models import AudioBuffer


class AudioIO:

    @staticmethod
    def load(path: str) -> AudioBuffer:
        if os.path.getsize(path) == 0:
            raise ValueError("Audio file is empty")
        for backend in ("soundfile", "ffmpeg", None):
            try:
                kwargs = {"backend": backend} if backend else {}
                waveform, sr = torchaudio.load(path, **kwargs)
                return AudioBuffer(data=waveform, sample_rate=sr)
            except Exception:
                continue
        raise RuntimeError("Could not load audio file")

    @staticmethod
    def to_mono(buf: AudioBuffer) -> AudioBuffer:
        if buf.data.shape[0] > 1:
            return AudioBuffer(data=buf.data.mean(dim=0, keepdim=True), sample_rate=buf.sample_rate)
        return buf

    @staticmethod
    def resample(buf: AudioBuffer, target_sr: int) -> AudioBuffer:
        if buf.sample_rate != target_sr:
            return AudioBuffer(
                data=F.resample(buf.data, buf.sample_rate, target_sr),
                sample_rate=target_sr,
            )
        return buf

    @staticmethod
    def detect_clipping(buf: AudioBuffer, threshold: float = 0.99) -> bool:
        return (buf.data.abs() > threshold).float().mean().item() > 0.001

    @staticmethod
    def save(wav: torch.Tensor, sr: int, path: str):
        torchaudio.save(path, wav.cpu(), sr)
