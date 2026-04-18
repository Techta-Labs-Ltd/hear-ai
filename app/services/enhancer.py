import asyncio
import os
import tempfile
import warnings
from dataclasses import dataclass

import numpy as np
import pyloudnorm as pyln
import torch
import torch.nn.functional as F_nn
import torchaudio
import torchaudio.functional as F
from demucs.apply import apply_model
from demucs.pretrained import get_model
from df.enhance import enhance as df_enhance, init_df
from silero_vad import get_speech_timestamps, load_silero_vad

warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

from app.config import settings
from app.core.storage import storage


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


class AudioEnhancer:
    TARGET_SR      = 44100
    DFN_SR         = 48_000
    DNS_SR         = 16_000
    TARGET_LUFS    = -16.0
    TRUE_PEAK_DBTP = -1.0


    _COMP_THRESHOLD_DB = -15.0
    _COMP_RATIO        = 1.5
    _COMP_MAKEUP_DB    = 2.0

    def __init__(self):
        self._demucs    = None
        self._dfn_model = None
        self._dfn_state = None
        self._vad_model = None
        self._device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        self._demucs = get_model(settings.DEMUCS_MODEL)
        self._demucs.to(self._device)
        self._demucs.eval()

        self._dfn_model, self._dfn_state, _ = init_df()
        self._dfn_model = self._dfn_model.to(self._device)

        self._vad_model = load_silero_vad()

    @property
    def is_loaded(self) -> bool:
        return self._demucs is not None

    def _load_audio(self, path: str) -> tuple[torch.Tensor, int]:
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

    def _to_mono(self, w: torch.Tensor) -> torch.Tensor:
        return w.mean(dim=0, keepdim=True) if w.shape[0] > 1 else w

    def _resample(self, w: torch.Tensor, orig: int, target: int) -> torch.Tensor:
        return F.resample(w, orig, target) if orig != target else w

    def _detect_clipping(self, w: torch.Tensor) -> bool:
        return (w.abs() > 0.99).float().mean().item() > 0.001

    def _compute_snr(self, raw: torch.Tensor, enhanced: torch.Tensor) -> float:
        raw = raw.cpu()
        enhanced = enhanced.cpu()
        n = min(raw.shape[1], enhanced.shape[1])
        sig_p = enhanced[:, :n].pow(2).mean().item()
        noi_p = (raw[:, :n] - enhanced[:, :n]).pow(2).mean().item() + 1e-10
        return 10 * np.log10(sig_p / noi_p)

    def _compute_lufs(self, w: torch.Tensor) -> float:
        try:
            meter = pyln.Meter(self.TARGET_SR)
            loudness = meter.integrated_loudness(w.cpu().squeeze(0).numpy().astype(np.float64))
            return loudness if np.isfinite(loudness) else -99.0
        except Exception:
            rms = w.pow(2).mean().sqrt().item()
            return float(20 * np.log10(rms + 1e-8))

    def _compute_quality_score(self, snr_db: float, clipping: bool, lufs: float) -> float:
        snr_score  = min(1.0, max(0.0, (snr_db + 5) / 40))
        lufs_score = 1.0 - min(1.0, abs(lufs - self.TARGET_LUFS) / 20)
        clip_pen   = 0.3 if clipping else 0.0
        return round(max(0.0, snr_score * 0.6 + lufs_score * 0.4 - clip_pen), 3)


    def _deepfilter_denoise(self, w: torch.Tensor) -> torch.Tensor:
        try:
            w_48k = self._resample(w.to(self._device), self.TARGET_SR, self.DFN_SR)
            with torch.no_grad():
                clean = df_enhance(self._dfn_model, self._dfn_state, w_48k)
            if isinstance(clean, np.ndarray):
                clean = torch.from_numpy(clean)
            if clean.dim() == 1:
                clean = clean.unsqueeze(0)
            return self._resample(clean.to(self._device), self.DFN_SR, self.TARGET_SR)
        except Exception:
            return w

    def _demucs_extract_vocals(self, waveform: torch.Tensor, sr: int, blend: float = 1.0) -> torch.Tensor:
        orig_mono = self._to_mono(waveform)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        resampled = self._resample(waveform, sr, self._demucs.samplerate)
        with torch.no_grad():
            sources = apply_model(self._demucs, resampled[None], progress=False)[0]
        idx    = self._demucs.sources.index("vocals")
        vocals = self._resample(sources[idx], self._demucs.samplerate, self.TARGET_SR)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        vocals_mono = self._to_mono(vocals)
        
        if blend < 1.0:
            # Parallel blend slightly hides robotic Demucs artifacts while reducing dogs/claps
            min_len = min(vocals_mono.shape[1], orig_mono.shape[1])
            return (vocals_mono[:, :min_len] * blend) + (orig_mono[:, :min_len] * (1.0 - blend))
            
        return vocals_mono

    def _apply_eq(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            w = F.highpass_biquad(w, sr, cutoff_freq=80.0)     # remove rumble
            w = F.lowpass_biquad(w, sr, cutoff_freq=10000.0)   # remove hiss
            w = F.equalizer_biquad(w, sr, center_freq=2500.0, gain=1.5, Q=2.0)
            return w
        except Exception:
            return w

    def _noise_gate(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            threshold = 10 ** (-45 / 20)  # aggressive floor
            mask = w.abs() > threshold
            return w * mask
        except Exception:
            return w

    def _compress(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            threshold_lin = 10 ** (self._COMP_THRESHOLD_DB / 20)
            makeup_lin    = 10 ** (self._COMP_MAKEUP_DB / 20)
            
            kernel_size = int(sr * 0.01)
            if kernel_size % 2 == 0: kernel_size += 1
            
            env = F_nn.avg_pool1d(w.abs().unsqueeze(1), kernel_size=kernel_size, stride=1, padding=kernel_size//2).squeeze(1)
            mask = env > threshold_lin
            
            gain = torch.ones_like(w)
            excess = env[mask] - threshold_lin
            compressed_env = threshold_lin + (excess / self._COMP_RATIO)
            gain[mask] = compressed_env / env[mask]
            
            smooth_kernel = int(sr * 0.05)
            if smooth_kernel % 2 == 0: smooth_kernel += 1
            gain = F_nn.avg_pool1d(gain.unsqueeze(1), kernel_size=smooth_kernel, stride=1, padding=smooth_kernel//2).squeeze(1)
            
            return w * gain * makeup_lin
        except Exception:
            return w

    def _strip_silence(self, w: torch.Tensor, timestamps: list[dict], sr: int) -> torch.Tensor:
        try:
            if not timestamps:
                return w
            start = max(0, timestamps[0]['start'])
            end = min(w.shape[1], timestamps[-1]['end'])
            trimmed = w[:, start:end].clone()
            
            fade_len = int(sr * 0.01)
            if trimmed.shape[1] > fade_len * 2:
                fade_in = torch.linspace(0.0, 1.0, fade_len, device=w.device)
                fade_out = torch.linspace(1.0, 0.0, fade_len, device=w.device)
                trimmed[0, :fade_len] *= fade_in
                trimmed[0, -fade_len:] *= fade_out
            return trimmed
        except Exception:
            return w

    def _remove_internal_silence(self, w: torch.Tensor, sr: int, max_gap_ms: int = 500) -> torch.Tensor:
        # Disabled as manual audio splicing creates phase-mismatch pops/cracks
        return w

    def _normalise_lufs(self, w: torch.Tensor) -> torch.Tensor:
        try:
            meter = pyln.Meter(self.TARGET_SR)
            audio_np = w.cpu().squeeze(0).numpy().astype(np.float64)
            loudness = meter.integrated_loudness(audio_np)
            if not np.isfinite(loudness) or loudness < -60:
                return self._peak_normalise(w)
            if abs(loudness - self.TARGET_LUFS) <= 2.0:
                return w
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                normalised = pyln.normalize.loudness(audio_np, loudness, self.TARGET_LUFS)
            result = torch.from_numpy(normalised.astype(np.float32)).unsqueeze(0).to(self._device)
            peak = result.abs().max().item()
            if peak > 0.99:
                result = result * (0.99 / peak)
            return result
        except Exception:
            return self._peak_normalise(w)

    def _peak_normalise(self, w: torch.Tensor) -> torch.Tensor:
        peak = w.abs().max().item()
        if peak < 1e-8:
            return w
        return w * (10 ** (-1.0 / 20) / peak)

    def _true_peak_limit(self, w: torch.Tensor) -> torch.Tensor:
        ceiling = 10 ** (self.TRUE_PEAK_DBTP / 20)
        tp = w.abs().max().item()
        if tp <= ceiling:
            return w
        w_up = self._resample(w, self.TARGET_SR, self.TARGET_SR * 4)
        tp_up = w_up.abs().max().item()
        if tp_up > ceiling:
            w = w * (ceiling / tp_up)
        return w

    def _generate_vad_mask(self, w: torch.Tensor, sr: int) -> tuple[torch.Tensor, list[dict]]:
        w_16k = self._resample(w.cpu(), sr, 16000)
        if w_16k.dim() > 1:
            w_16k = w_16k.squeeze(0)
            
        with torch.no_grad():
            timestamps = get_speech_timestamps(w_16k, self._vad_model, sampling_rate=16000, return_seconds=False)
            
        mask = torch.zeros((1, w.shape[1]), device=self._device)
        scale = sr / 16000.0
        
        pad_16k = int(16000 * 0.08)
        min_len_16k = int(16000 * 0.12)
        max_idx_16k = w_16k.shape[0]
        
        scaled_timestamps = []
        for ts in timestamps:
            if (ts['end'] - ts['start']) < min_len_16k:
                continue
            start_16k = max(0, ts['start'] - pad_16k)
            end_16k = min(max_idx_16k, ts['end'] + pad_16k)
            
            start_idx = int(start_16k * scale)
            end_idx = int(end_16k * scale)
            mask[0, start_idx:end_idx] = 1.0
            scaled_timestamps.append({'start': start_idx, 'end': end_idx})
            
        kernel_size = int(sr * 0.02)
        if kernel_size % 2 == 0: kernel_size += 1
        pad = kernel_size // 2
        smoothed = F_nn.avg_pool1d(mask, kernel_size, stride=1, padding=pad)
        
        return torch.clamp(smoothed, 0.0, 1.0), scaled_timestamps

    def _dynamic_denoise(self, raw: torch.Tensor, cleaned: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        blend = 1.0 - (mask * 0.3)
        min_len = min(raw.shape[1], cleaned.shape[1], mask.shape[1])
        return (cleaned[:, :min_len] * blend[:, :min_len]) + (raw[:, :min_len] * (1.0 - blend[:, :min_len]))

    async def enhance(self, input_path: str, recording_id: str, job_id: str) -> EnhancementResult:
        loop = asyncio.get_event_loop()
        waveform, sr = self._load_audio(input_path)
        raw_clone = waveform.clone()
        clipping_input = self._detect_clipping(waveform)

        mono = self._to_mono(waveform)
        enhanced = self._resample(mono, sr, self.TARGET_SR).to(self._device)

        # VAD
        vad_mask, speech_timestamps = await loop.run_in_executor(
            None, self._generate_vad_mask, enhanced, self.TARGET_SR
        )

        # DeepFilterNet
        dfn_cleaned = await loop.run_in_executor(None, self._deepfilter_denoise, enhanced)

        # Dynamic denoise (stronger)
        enhanced = await loop.run_in_executor(
            None, self._dynamic_denoise, enhanced, dfn_cleaned, vad_mask
        )

        # Noise energy check
        min_len = min(enhanced.shape[1], vad_mask.shape[1])
        noise_energy = (enhanced[:, :min_len] * (1.0 - vad_mask[:, :min_len])).pow(2).mean().sqrt().item()

        # Demucs (stronger)
        if noise_energy > 0.08:
            enhanced = await loop.run_in_executor(
                None, self._demucs_extract_vocals, enhanced, self.TARGET_SR, 0.3
            )

        # EQ
        enhanced = await loop.run_in_executor(None, self._apply_eq, enhanced, self.TARGET_SR)

        # Compression
        enhanced = await loop.run_in_executor(None, self._compress, enhanced, self.TARGET_SR)

        # Noise gate (critical)
        enhanced = await loop.run_in_executor(None, self._noise_gate, enhanced, self.TARGET_SR)
        
        # Soft saturation removes micro-noise
        enhanced = torch.tanh(enhanced * 1.2)

        # Trim silence
        enhanced = await loop.run_in_executor(
            None, self._strip_silence, enhanced, speech_timestamps, self.TARGET_SR
        )

        # Normalize + limit
        enhanced = await loop.run_in_executor(None, self._normalise_lufs, enhanced)
        enhanced = await loop.run_in_executor(None, self._true_peak_limit, enhanced)

        raw_at_target = self._resample(self._to_mono(raw_clone), sr, self.TARGET_SR)
        snr = self._compute_snr(raw_at_target, enhanced)
        lufs = self._compute_lufs(enhanced)
        peak_db = 20 * np.log10(enhanced.abs().max().item() + 1e-8)
        quality_score = self._compute_quality_score(snr, clipping_input, lufs)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
        torchaudio.save(out_path, enhanced.cpu(), self.TARGET_SR)

        b2_key = f"{settings.B2_ENHANCED_PREFIX}{recording_id}/{job_id}.wav"
        enhanced_url = await loop.run_in_executor(None, storage.upload_file, out_path, b2_key)

        return EnhancementResult(
            b2_key=b2_key,
            enhanced_url=enhanced_url,
            local_path=out_path,
            quality_score=quality_score,
            snr_db=round(snr, 2),
            peak_db=round(peak_db, 2),
            lufs=round(lufs, 2),
            clipping_detected=clipping_input,
        )
