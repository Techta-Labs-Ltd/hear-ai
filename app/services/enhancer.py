import asyncio
import os
import tempfile
import warnings
from dataclasses import dataclass

import numpy as np
import pyloudnorm as pyln
import torch
import torchaudio
import torchaudio.functional as F
from demucs.apply import apply_model
from demucs.pretrained import get_model
from df.enhance import enhance as df_enhance, init_df

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
    TARGET_LUFS    = -12.0
    TRUE_PEAK_DBTP = -1.0

    _HP_FREQ       = 80.0
    _LP_FREQ       = 14_000.0
    _EQ_CUT_FREQ   = 200.0
    _EQ_CUT_GAIN   = -2.0
    _EQ_CUT_Q      = 1.4
    _EQ_BOOST_FREQ = 3_000.0
    _EQ_BOOST_GAIN = 2.0
    _EQ_BOOST_Q    = 2.0

    _DESS_FREQ1 = 7_000.0
    _DESS_GAIN1 = -3.0
    _DESS_Q1    = 1.5
    _DESS_FREQ2 = 9_000.0
    _DESS_GAIN2 = -2.0
    _DESS_Q2    = 2.0

    _COMP_THRESHOLD_DB = -18.0
    _COMP_RATIO        = 3.5
    _COMP_MAKEUP_DB    = 6.0

    _GATE_THRESHOLD_DB = -35.0

    def __init__(self):
        self._demucs    = None
        self._dfn_model = None
        self._device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        self._demucs = get_model(settings.DEMUCS_MODEL)
        self._demucs.to(self._device)
        self._demucs.eval()

        self._dfn_model, _, _ = init_df()
        self._dfn_model = self._dfn_model.to(self._device)
        self._dfn_model.eval()

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
            w_48k = self._resample(w, self.TARGET_SR, self.DFN_SR).cpu()
            _, dfn_state, _ = init_df()
            with torch.no_grad():
                clean = df_enhance(self._dfn_model, dfn_state, w_48k)
            if isinstance(clean, np.ndarray):
                clean = torch.from_numpy(clean)
            if clean.dim() == 1:
                clean = clean.unsqueeze(0)
            return self._resample(clean.to(self._device), self.DFN_SR, self.TARGET_SR)
        except Exception:
            return w

    def _demucs_extract_vocals(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        resampled = self._resample(waveform, sr, self._demucs.samplerate)
        with torch.no_grad():
            sources = apply_model(self._demucs, resampled[None], progress=False)[0]
        idx    = self._demucs.sources.index("vocals")
        vocals = self._resample(sources[idx], self._demucs.samplerate, self.TARGET_SR)
        return self._to_mono(vocals)

    def _noise_gate(self, w: torch.Tensor) -> torch.Tensor:
        kernel_size = int(self.TARGET_SR * 0.02)
        if kernel_size % 2 == 0: kernel_size += 1
        
        env = torch.nn.functional.avg_pool1d(
            w.abs().unsqueeze(1),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        ).squeeze(1)
        
        threshold_lin = 10 ** (self._GATE_THRESHOLD_DB / 20)
        raw_gain = torch.clamp(env / threshold_lin, max=1.0)
        
        smooth_kernel = int(self.TARGET_SR * 0.05)
        if smooth_kernel % 2 == 0: smooth_kernel += 1
        
        gain = torch.nn.functional.avg_pool1d(
            raw_gain.unsqueeze(1),
            kernel_size=smooth_kernel,
            stride=1,
            padding=smooth_kernel // 2
        ).squeeze(1)
        
        return w * gain

    def _apply_eq(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            w = F.highpass_biquad(w, sr, cutoff_freq=self._HP_FREQ)
            w = F.lowpass_biquad(w, sr, cutoff_freq=self._LP_FREQ)
            w = F.equalizer_biquad(w, sr, center_freq=self._EQ_CUT_FREQ, gain=self._EQ_CUT_GAIN, Q=self._EQ_CUT_Q)
            w = F.equalizer_biquad(w, sr, center_freq=self._EQ_BOOST_FREQ, gain=self._EQ_BOOST_GAIN, Q=self._EQ_BOOST_Q)
            peak = w.abs().max().clamp(min=1e-8)
            if peak > 0.5:
                w = w * (0.9 / peak)
            return w
        except Exception:
            return w

    def _de_ess(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            w = F.equalizer_biquad(w, sr, center_freq=self._DESS_FREQ1, gain=self._DESS_GAIN1, Q=self._DESS_Q1)
            w = F.equalizer_biquad(w, sr, center_freq=self._DESS_FREQ2, gain=self._DESS_GAIN2, Q=self._DESS_Q2)
            return w
        except Exception:
            return w

    def _compress(self, w: torch.Tensor) -> torch.Tensor:
        kernel_size = int(self.TARGET_SR * 0.01)
        if kernel_size % 2 == 0: kernel_size += 1
        
        env = torch.nn.functional.avg_pool1d(
            w.abs().unsqueeze(1), 
            kernel_size=kernel_size, 
            stride=1, 
            padding=kernel_size // 2
        ).squeeze(1)
        
        threshold_lin = 10 ** (self._COMP_THRESHOLD_DB / 20)
        makeup_lin = 10 ** (self._COMP_MAKEUP_DB / 20)
        
        above = torch.clamp(env - threshold_lin, min=0.0)
        
        output_env = torch.where(
            env > threshold_lin,
            threshold_lin + (env - threshold_lin) / self._COMP_RATIO,
            env
        )
        
        # Calculate raw gain securely avoiding divide-by-zero
        raw_gain = torch.where(env > 1e-8, output_env / env, torch.ones_like(env))
        
        smooth_kernel = int(self.TARGET_SR * 0.05)
        if smooth_kernel % 2 == 0: smooth_kernel += 1
        
        gain = torch.nn.functional.avg_pool1d(
            raw_gain.unsqueeze(1),
            kernel_size=smooth_kernel,
            stride=1,
            padding=smooth_kernel // 2
        ).squeeze(1)
        
        return w * gain * makeup_lin

    def _strip_silence(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            frame_size = int(sr * 0.05)
            n_frames   = w.shape[1] // frame_size
            if n_frames < 2:
                return w
            energies = w.abs().unfold(1, frame_size, frame_size).mean(dim=2).squeeze(0)
            voiced   = (energies > 0.001).nonzero(as_tuple=True)[0]
            if len(voiced) < 2:
                return w
            start   = int(voiced[0].item()) * frame_size
            end     = min(w.shape[1], (int(voiced[-1].item()) + 1) * frame_size)
            trimmed = w[:, start:end]
            
            fade_len = int(sr * 0.02)
            if trimmed.shape[1] > fade_len * 2:
                fade_in = torch.linspace(0.0, 1.0, fade_len, device=w.device)
                fade_out = torch.linspace(1.0, 0.0, fade_len, device=w.device)
                trimmed = trimmed.clone()
                trimmed[0, :fade_len] *= fade_in
                trimmed[0, -fade_len:] *= fade_out
                
            return trimmed
        except Exception:
            return w

    def _remove_internal_silence(self, w: torch.Tensor, sr: int, max_gap_ms: int = 500) -> torch.Tensor:
        try:
            frame_ms = 20
            frame_size = int(sr * (frame_ms / 1000.0))
            max_gap_frames = int(max_gap_ms / frame_ms)
            
            energies = w.pow(2).unfold(1, frame_size, frame_size).mean(dim=2).sqrt().squeeze(0)
            voiced = energies > 0.001
            
            smoothed_voiced = voiced.clone()
            silence_run = 0
            for i in range(len(voiced)):
                if not voiced[i]:
                    silence_run += 1
                else:
                    if silence_run > 0 and silence_run <= max_gap_frames:
                        smoothed_voiced[i-silence_run:i] = True
                    silence_run = 0
                    
            segments = []
            current_seg = []
            for i, v in enumerate(smoothed_voiced):
                if v:
                    current_seg.append(i)
                else:
                    if current_seg:
                        segments.append((current_seg[0], current_seg[-1]))
                        current_seg = []
            if current_seg:
                segments.append((current_seg[0], current_seg[-1]))
                
            if not segments:
                return w
                
            fade_len = int(sr * 0.015)
            fade_in = torch.linspace(0.0, 1.0, fade_len, device=w.device)
            fade_out = torch.linspace(1.0, 0.0, fade_len, device=w.device)
            
            out_tensors = []
            for i, (sf, ef) in enumerate(segments):
                start_samp = sf * frame_size
                end_samp = min(w.shape[1], (ef + 1) * frame_size)
                
                chunk = w[:, start_samp:end_samp].clone()
                if chunk.shape[1] > fade_len * 2:
                    if i > 0:
                        chunk[0, :fade_len] *= fade_in
                    if i < len(segments) - 1:
                        chunk[0, -fade_len:] *= fade_out
                out_tensors.append(chunk)
                
            return torch.cat(out_tensors, dim=1) if out_tensors else w
        except Exception:
            return w

    def _normalise_lufs(self, w: torch.Tensor) -> torch.Tensor:
        try:
            meter = pyln.Meter(self.TARGET_SR)
            audio_np = w.cpu().squeeze(0).numpy().astype(np.float64)
            loudness = meter.integrated_loudness(audio_np)
            if not np.isfinite(loudness) or loudness < -60:
                return self._peak_normalise(w)
            import warnings
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
        w_up    = self._resample(w, self.TARGET_SR, self.TARGET_SR * 4)
        tp      = w_up.abs().max().item()
        if tp > ceiling:
            w = w * (ceiling / tp)
        return w

    async def enhance(self, input_path: str, recording_id: str, job_id: str) -> EnhancementResult:
        loop = asyncio.get_event_loop()
        waveform, sr = self._load_audio(input_path)
        raw_clone = waveform.clone()
        clipping_input = self._detect_clipping(waveform)

        mono = self._to_mono(waveform)
        enhanced = self._resample(mono, sr, self.TARGET_SR).to(self._device)

        enhanced = await loop.run_in_executor(None, self._deepfilter_denoise, enhanced)
        enhanced = await loop.run_in_executor(None, self._demucs_extract_vocals, enhanced, self.TARGET_SR)
        enhanced = await loop.run_in_executor(None, self._apply_eq, enhanced, self.TARGET_SR)
        enhanced = await loop.run_in_executor(None, self._de_ess, enhanced, self.TARGET_SR)
        enhanced = await loop.run_in_executor(None, self._noise_gate, enhanced)
        enhanced = await loop.run_in_executor(None, self._compress, enhanced)
        enhanced = await loop.run_in_executor(None, self._strip_silence, enhanced, self.TARGET_SR)
        enhanced = await loop.run_in_executor(None, self._remove_internal_silence, enhanced, self.TARGET_SR)
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
