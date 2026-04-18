import asyncio
import os
import tempfile
import threading
import warnings
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pyloudnorm as pyln
import scipy.signal as ss
import torch
import torchaudio
import torchaudio.functional as F
from demucs.apply import apply_model
from demucs.pretrained import get_model
from df.enhance import enhance as df_enhance, init_df
from silero_vad import get_speech_timestamps, load_silero_vad

from app.config import settings
from app.core.storage import storage
from pyrnnoise import RNNoise as _RNNoise

warnings.filterwarnings("ignore")


class ContentMode(str, Enum):
    SPEECH = "speech"
    MUSIC = "music"
    AUTO = "auto"


@dataclass
class EnhancementResult:
    b2_key: str
    enhanced_url: str
    local_path: str
    quality_score: float
    snr_db: float
    peak_db: float
    lufs: float
    clipping_detected: bool
    mode_used: str


def _match(a, b):
    n = min(a.shape[-1], b.shape[-1])
    return a[..., :n], b[..., :n]


class AudioEnhancer:
    TARGET_SR = 44100
    DFN_SR = 48000
    VAD_SR = 16000
    TARGET_LUFS = -16.0
    TRUE_PEAK_DBTP = -1.0

    def __init__(self):
        self._demucs_model = None
        self._dfn_model = None
        self._vad_model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._vad_lock = threading.Lock()
        self._dfn_lock = threading.Lock()
        self._rnn = _RNNoise(sample_rate=48000)
        self._rnnoise_lock = threading.Lock()

    def load(self):
        self._demucs_model = get_model(settings.DEMUCS_MODEL)
        self._demucs_model.to(self._device).eval()
        self._dfn_model, _, _ = init_df()
        self._vad_model = load_silero_vad()

    def _load_audio(self, path):
        waveform, sr = torchaudio.load(path)
        return waveform, sr

    def _mono(self, w):
        return w.mean(dim=0, keepdim=True) if w.shape[0] > 1 else w

    def _resample(self, w, a, b):
        return F.resample(w, a, b) if a != b else w

    def _deepfilter(self, w):
        w48 = self._resample(w.cpu(), self.TARGET_SR, self.DFN_SR)
        with self._dfn_lock:
            _, state, _ = init_df()
            with torch.no_grad():
                out = df_enhance(self._dfn_model, state, w48)
        if isinstance(out, np.ndarray):
            out = torch.from_numpy(out)
        if out.dim() == 1:
            out = out.unsqueeze(0)
        return self._resample(out.float(), self.DFN_SR, self.TARGET_SR).to(self._device)

    def _demucs_vocals(self, w):
        stereo = w.repeat(2, 1)
        res = self._resample(stereo, self.TARGET_SR, self._demucs_model.samplerate)
        with torch.no_grad():
            s = apply_model(self._demucs_model, res[None])[0]
        idx = self._demucs_model.sources.index("vocals")
        v = self._resample(s[idx], self._demucs_model.samplerate, self.TARGET_SR)
        return self._mono(v)

    def _rnnoise_denoise(self, w: torch.Tensor) -> torch.Tensor:
        try:
            target_sr = 48000
            x48 = self._resample(w.cpu(), self.TARGET_SR, target_sr).squeeze().numpy()
            x16 = (np.clip(x48, -1.0, 1.0) * 32767).astype(np.int16)

            out_chunks = []
            with self._rnnoise_lock:
                for _, denoised in self._rnn.denoise_chunk(x16[np.newaxis, :]):
                    out_chunks.append(denoised)

            if not out_chunks:
                return w

            out16 = np.concatenate(out_chunks, axis=-1).squeeze()
            out_f = out16.astype(np.float32) / 32767.0

            pad = len(x48) - len(out_f)
            if pad > 0:
                out_f = np.pad(out_f, (0, pad))
            else:
                out_f = out_f[:len(x48)]

            y = torch.from_numpy(out_f).unsqueeze(0).to(self._device)
            return self._resample(y, target_sr, self.TARGET_SR)
        except Exception:
            return w

    def _transient_suppress(self, w):
        x = w.squeeze(0).cpu().numpy()
        env = np.abs(x)
        avg = ss.uniform_filter1d(env, size=1024)
        spikes = env > avg * 4.0
        mask = (~spikes).astype(np.float32)
        mask = ss.gaussian_filter1d(mask, sigma=20)
        return torch.from_numpy(x * mask).unsqueeze(0).to(w.device)

    def _vad_mask(self, w):
        w16 = self._resample(w.cpu(), self.TARGET_SR, self.VAD_SR).squeeze()
        with self._vad_lock:
            ts = get_speech_timestamps(w16, self._vad_model, sampling_rate=self.VAD_SR)
        mask = torch.zeros((1, w.shape[1]), device=self._device)
        scale = self.TARGET_SR / self.VAD_SR
        for t in ts:
            s = int(t["start"] * scale)
            e = int(t["end"] * scale)
            mask[:, s:e] = 1.0
        return mask

    def _noise_gate(self, w):
        thr = 10 ** (-60 / 20)
        env = w.abs()
        smoothed = torch.nn.functional.avg_pool1d(env.unsqueeze(1), 1024, stride=1, padding=512).squeeze(1)
        mask = smoothed > thr
        return w * mask.float()

    def _eq(self, w):
        w = F.highpass_biquad(w, self.TARGET_SR, 100)
        w = F.equalizer_biquad(w, self.TARGET_SR, 3000, gain=2.0, Q=1.8)
        return w

    def _compress(self, w):
        thr = 10 ** (-18 / 20)
        ratio = 2.5
        env = w.abs()
        gain = torch.ones_like(w)
        over = env > thr
        gain[over] = (thr + (env[over] - thr) / ratio) / (env[over] + 1e-6)
        return w * gain

    def _lufs(self, w):
        meter = pyln.Meter(self.TARGET_SR)
        x = w.cpu().squeeze().numpy().astype(np.float64)
        if len(x) < self.TARGET_SR // 2:
            return w
        l = meter.integrated_loudness(x)
        y = pyln.normalize.loudness(x, l, self.TARGET_LUFS)
        return torch.from_numpy(y.astype(np.float32)).unsqueeze(0).to(self._device)

    def _limit(self, w):
        peak = w.abs().max()
        ceiling = 10 ** (self.TRUE_PEAK_DBTP / 20)
        if peak > ceiling:
            w = w * (ceiling / peak)
        return w

    def _snr(self, raw, enhanced):
        raw, enhanced = _match(raw.cpu(), enhanced.cpu())
        s = enhanced.pow(2).mean().item()
        n = (raw - enhanced).pow(2).mean().item() + 1e-10
        return 10 * np.log10(s / n)

    def _loudness(self, w):
        meter = pyln.Meter(self.TARGET_SR)
        return meter.integrated_loudness(w.cpu().squeeze().numpy().astype(np.float64))

    async def enhance(self, path, rid, jid, mode=ContentMode.SPEECH):
        loop = asyncio.get_running_loop()

        w, sr = self._load_audio(path)
        raw = w.clone()

        w = self._mono(w)
        w = self._resample(w, sr, self.TARGET_SR).to(self._device)

        vocals = await loop.run_in_executor(None, self._demucs_vocals, w)
        w, vocals = _match(w, vocals)
        w = vocals

        clean = await loop.run_in_executor(None, self._deepfilter, w)
        w, clean = _match(w, clean)
        w = clean

        w = await loop.run_in_executor(None, self._rnnoise_denoise, w)
        w = await loop.run_in_executor(None, self._transient_suppress, w)

        mask = await loop.run_in_executor(None, self._vad_mask, w)
        w, mask = _match(w, mask)
        w = w * (0.1 + 0.9 * mask)

        w = await loop.run_in_executor(None, self._noise_gate, w)
        w = await loop.run_in_executor(None, self._eq, w)
        w = await loop.run_in_executor(None, self._compress, w)

        w = await loop.run_in_executor(None, self._lufs, w)
        w = await loop.run_in_executor(None, self._limit, w)

        raw_ref = self._resample(self._mono(raw), sr, self.TARGET_SR)

        snr = self._snr(raw_ref, w)
        lufs = self._loudness(w)
        peak = 20 * np.log10(w.abs().max().item() + 1e-8)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out = tmp.name

        torchaudio.save(out, w.cpu(), self.TARGET_SR)

        key = f"{settings.B2_ENHANCED_PREFIX}{rid}/{jid}.wav"
        url = await loop.run_in_executor(None, storage.upload_file, out, key)

        return EnhancementResult(
            b2_key=key,
            enhanced_url=url,
            local_path=out,
            quality_score=round(max(0.0, min(1.0, (snr + 5) / 40)), 3),
            snr_db=round(snr, 2),
            peak_db=round(peak, 2),
            lufs=round(lufs, 2),
            clipping_detected=(w.abs() > 0.99).float().mean().item() > 0.001,
            mode_used=mode.value,
        )