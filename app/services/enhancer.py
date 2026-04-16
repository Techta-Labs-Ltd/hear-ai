import asyncio
import os
import tempfile
from dataclasses import dataclass

import numpy as np
import noisereduce as nr
import pyloudnorm as pyln
import torch
import torchaudio
import torchaudio.functional as F
from demucs.apply import apply_model
from demucs.pretrained import get_model
from denoiser import pretrained as dns_pretrained
from df.enhance import enhance as df_enhance, init_df
from nara_wpe.wpe import wpe_v8
from nara_wpe.utils import stft as wpe_stft, istft as wpe_istft

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
    TARGET_SR = 44100
    DFN_SR    = 48_000
    DNS_SR    = 16_000

    # Loudness targets
    TARGET_LUFS      = -12.0   # Loud, commercial-ready (Spotify -14, YouTube -14, broadcast -23)
    TRUE_PEAK_DBTP   = -1.0    # Broadcast ceiling

    # EQ
    _HP_FREQ        = 80.0
    _LP_FREQ        = 14_000.0
    _EQ_CUT_FREQ    = 200.0
    _EQ_CUT_GAIN    = -2.0
    _EQ_CUT_Q       = 1.4
    _EQ_BOOST_FREQ  = 3_000.0
    _EQ_BOOST_GAIN  = 2.0
    _EQ_BOOST_Q     = 2.0

    # De-essing
    _DESS_FREQ1 = 7_000.0
    _DESS_GAIN1 = -3.0
    _DESS_Q1    = 1.5
    _DESS_FREQ2 = 9_000.0
    _DESS_GAIN2 = -2.0
    _DESS_Q2    = 2.0

    # Compressor
    _COMP_THRESHOLD_DB = -18.0
    _COMP_RATIO        = 3.5
    _COMP_ATTACK_MS    = 10.0
    _COMP_RELEASE_MS   = 120.0
    _COMP_MAKEUP_DB    = 5.0

    # Noise gate
    _GATE_THRESHOLD_DB = -45.0
    _GATE_ATTACK_MS    = 5.0
    _GATE_RELEASE_MS   = 60.0

    def __init__(self):
        self._demucs    = None
        self._dns       = None
        self._dfn_model = None
        self._dfn_state = None
        self._device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        self._demucs = get_model(settings.DEMUCS_MODEL)
        self._demucs.to(self._device)
        self._demucs.eval()
        print("[ENHANCER] Demucs loaded")

        self._dfn_model, self._dfn_state, _ = init_df()
        self._dfn_model = self._dfn_model.to(self._device)
        self._dfn_model.eval()
        print(f"[ENHANCER] DeepFilterNet loaded on {self._device}")

        self._dns = dns_pretrained.dns64()
        self._dns.to(self._device)
        self._dns.eval()
        print("[ENHANCER] Facebook DNS denoiser loaded (dns64 fallback)")

        print("[ENHANCER] nara_wpe | pyloudnorm | noisereduce ready")

    @property
    def is_loaded(self) -> bool:
        return self._demucs is not None

    # ------------------------------------------------------------------ #
    #  I/O helpers
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    #  Signal analysis
    # ------------------------------------------------------------------ #

    def _detect_clipping(self, w: torch.Tensor) -> bool:
        return (w.abs() > 0.99).float().mean().item() > 0.001

    def _compute_snr(self, raw: torch.Tensor, enhanced: torch.Tensor) -> float:
        n = min(raw.shape[1], enhanced.shape[1])
        noise = raw[:, :n] - enhanced[:, :n]
        sig_p = enhanced[:, :n].pow(2).mean().item()
        noi_p = noise.pow(2).mean().item() + 1e-10
        return 10 * np.log10(sig_p / noi_p)

    def _compute_lufs(self, w: torch.Tensor) -> float:
        try:
            meter = pyln.Meter(self.TARGET_SR)
            audio_np = w.squeeze(0).numpy().astype(np.float64)
            return meter.integrated_loudness(audio_np)
        except Exception:
            rms = w.pow(2).mean().sqrt().item()
            return 20 * np.log10(rms + 1e-8)

    def _compute_quality_score(self, snr_db: float, clipping: bool, lufs: float) -> float:
        snr_score  = min(1.0, max(0.0, (snr_db + 5) / 40))
        lufs_score = 1.0 - min(1.0, abs(lufs - self.TARGET_LUFS) / 20)
        clip_pen   = 0.3 if clipping else 0.0
        return round(max(0.0, snr_score * 0.6 + lufs_score * 0.4 - clip_pen), 3)

    # ------------------------------------------------------------------ #
    #  Processing pipeline
    # ------------------------------------------------------------------ #

    def _dereverberate(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        """nara_wpe blind de-reverberation — removes room echo and reverb."""
        try:
            audio_np = w.squeeze(0).numpy().astype(np.float64)
            Y = wpe_stft(audio_np, size=512, shift=128).T[None]
            Z = wpe_v8(Y, taps=10, delay=3, iterations=3)
            out = wpe_istft(Z[0].T, size=512, shift=128)[: audio_np.shape[0]]
            return torch.from_numpy(out.astype(np.float32)).unsqueeze(0)
        except Exception:
            return w

    def _deepfilter_denoise(self, w: torch.Tensor) -> torch.Tensor:
        """DeepFilterNet SOTA speech enhancement @ 48 kHz on GPU."""
        try:
            print("[ENHANCER] DeepFilterNet running...")
            w_48k = self._resample(w, self.TARGET_SR, self.DFN_SR).to(self._device)
            clean = df_enhance(self._dfn_model, self._dfn_state, w_48k)
            if isinstance(clean, np.ndarray):
                clean = torch.from_numpy(clean)
            if clean.dim() == 1:
                clean = clean.unsqueeze(0)
            clean = clean.cpu()
            print("[ENHANCER] DeepFilterNet done")
            return self._resample(clean, self.DFN_SR, self.TARGET_SR)
        except Exception as e:
            print(f"[ENHANCER] DeepFilterNet error: {e} — falling back to DNS")
            return self._dns_denoise(w)



    def _dns_denoise(self, w: torch.Tensor) -> torch.Tensor:
        """Facebook dns64 fallback denoiser @ 16 kHz."""
        try:
            w_16k = self._resample(w, self.TARGET_SR, self.DNS_SR).to(self._device)
            with torch.no_grad():
                clean = self._dns(w_16k[None])[0].cpu()
            return self._resample(clean, self.DNS_SR, self.TARGET_SR)
        except Exception as e:
            print(f"[ENHANCER] DNS error: {e} — falling back to noisereduce")
            return self._nr_denoise(w)

    def _nr_denoise(self, w: torch.Tensor) -> torch.Tensor:
        """noisereduce spectral subtraction — final fallback."""
        try:
            reduced = nr.reduce_noise(
                y=w.squeeze(0).numpy(), sr=self.TARGET_SR,
                stationary=False, prop_decrease=0.85,
            )
            return torch.from_numpy(reduced).unsqueeze(0)
        except Exception:
            return w

    def _residual_cleanup(self, w: torch.Tensor) -> torch.Tensor:
        """
        Moderate noisereduce pass applied AFTER Demucs.
        Removes residual noise that bled through Demucs' vocals stem
        without eating the speech signal.
        """
        try:
            audio_np = w.squeeze(0).numpy()
            # Non-stationary pass: targets transient sounds (barking, clicks)
            stage1 = nr.reduce_noise(
                y=audio_np,
                sr=self.TARGET_SR,
                stationary=False,
                prop_decrease=0.70,
            )
            # Stationary pass: removes constant residual hum/hiss
            stage2 = nr.reduce_noise(
                y=stage1,
                sr=self.TARGET_SR,
                stationary=True,
                prop_decrease=0.60,
            )
            return torch.from_numpy(stage2.astype(np.float32)).unsqueeze(0)
        except Exception:
            return w

    def _demucs_extract_vocals(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """Demucs vocals stem extraction — strips background music."""
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        resampled = self._resample(waveform, sr, self._demucs.samplerate)
        with torch.no_grad():
            sources = apply_model(
                self._demucs, resampled[None].to(self._device), progress=False,
            )[0]
        idx    = self._demucs.sources.index("vocals")
        vocals = self._resample(sources[idx].cpu(), self._demucs.samplerate, self.TARGET_SR)
        return self._to_mono(vocals)

    def _noise_gate(self, w: torch.Tensor) -> torch.Tensor:
        """
        Adaptive noise gate — silences frames below threshold.
        Eliminates residual noise between sentences for a clean, studio-like feel.
        """
        sr = self.TARGET_SR
        x  = w.squeeze(0).numpy().astype(np.float32)

        threshold_lin  = 10 ** (self._GATE_THRESHOLD_DB / 20)
        attack_coef    = np.exp(-1.0 / (sr * self._GATE_ATTACK_MS  / 1000.0))
        release_coef   = np.exp(-1.0 / (sr * self._GATE_RELEASE_MS / 1000.0))

        envelope = np.zeros_like(x)
        level    = 0.0
        for i, s in enumerate(np.abs(x)):
            coef     = attack_coef if s > level else release_coef
            level    = coef * level + (1.0 - coef) * s
            envelope[i] = level

        gain = np.where(envelope > threshold_lin, 1.0, envelope / (threshold_lin + 1e-8))
        return torch.from_numpy((x * gain).astype(np.float32)).unsqueeze(0)

    def _declick(self, w: torch.Tensor) -> torch.Tensor:
        """
        De-clicking — detects and interpolates over transient spikes
        (plosives, mic bumps, electrical clicks).
        """
        x      = w.squeeze(0).numpy().astype(np.float32)
        diff   = np.abs(np.diff(x, prepend=x[0]))
        median = np.median(diff)
        clicks = diff > (median * 20)

        for i in np.where(clicks)[0]:
            left  = max(0, i - 3)
            right = min(len(x) - 1, i + 3)
            x[i]  = np.interp(i, [left, right], [x[left], x[right]])

        return torch.from_numpy(x).unsqueeze(0)

    def _apply_eq(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        """High-pass → low-pass → mud cut → presence boost."""
        try:
            w = F.highpass_biquad(w, sr, cutoff_freq=self._HP_FREQ)
            w = F.lowpass_biquad(w, sr, cutoff_freq=self._LP_FREQ)
            w = F.equalizer_biquad(w, sr, center_freq=self._EQ_CUT_FREQ,
                                   gain=self._EQ_CUT_GAIN, Q=self._EQ_CUT_Q)
            w = F.equalizer_biquad(w, sr, center_freq=self._EQ_BOOST_FREQ,
                                   gain=self._EQ_BOOST_GAIN, Q=self._EQ_BOOST_Q)
            peak = w.abs().max().clamp(min=1e-8)
            if peak > 0.5:
                w = w * (0.9 / peak)
            return w
        except Exception:
            return w

    def _de_ess(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        """Attenuate harsh sibilance in the 7–9 kHz range."""
        try:
            w = F.equalizer_biquad(w, sr, center_freq=self._DESS_FREQ1,
                                   gain=self._DESS_GAIN1, Q=self._DESS_Q1)
            w = F.equalizer_biquad(w, sr, center_freq=self._DESS_FREQ2,
                                   gain=self._DESS_GAIN2, Q=self._DESS_Q2)
            return w
        except Exception:
            return w

    def _compress(self, w: torch.Tensor) -> torch.Tensor:
        """
        Feed-forward dynamic range compressor.
        Ratio 3.5:1 — evens out loud and quiet passages for consistent
        podcast loudness without pumping or squashing.
        """
        sr = self.TARGET_SR
        x  = w.squeeze(0).numpy().astype(np.float32)

        threshold_lin = 10 ** (self._COMP_THRESHOLD_DB / 20)
        attack_coef   = np.exp(-1.0 / (sr * self._COMP_ATTACK_MS  / 1000.0))
        release_coef  = np.exp(-1.0 / (sr * self._COMP_RELEASE_MS / 1000.0))
        makeup_lin    = 10 ** (self._COMP_MAKEUP_DB / 20)

        envelope  = np.zeros_like(x)
        level     = 0.0
        for i, s in enumerate(np.abs(x)):
            coef     = attack_coef if s > level else release_coef
            level    = coef * level + (1.0 - coef) * s
            envelope[i] = level

        above       = np.maximum(envelope - threshold_lin, 0.0)
        gain_reduce = threshold_lin + above / self._COMP_RATIO
        gain        = np.where(envelope > 1e-8, gain_reduce / envelope, 1.0)
        compressed  = x * gain * makeup_lin

        return torch.from_numpy(compressed.astype(np.float32)).unsqueeze(0)

    def _normalise_lufs(self, w: torch.Tensor) -> torch.Tensor:
        """
        ITU-R BS.1770 integrated loudness normalisation via pyloudnorm.
        Falls back to peak-based normalisation if loudness measurement fails
        (can happen when noise gate creates many silent frames that confuse
        the ITU-R gating algorithm).
        """
        try:
            meter    = pyln.Meter(self.TARGET_SR)
            audio_np = w.squeeze(0).numpy().astype(np.float64)
            loudness = meter.integrated_loudness(audio_np)
            print(f"[ENHANCER] Measured loudness: {loudness:.1f} LUFS → target {self.TARGET_LUFS} LUFS")

            if not np.isfinite(loudness) or loudness < -60:
                print("[ENHANCER] Loudness too low for pyloudnorm — using peak normalisation")
                return self._peak_normalise(w)

            normalised = pyln.normalize.loudness(audio_np, loudness, self.TARGET_LUFS)
            return torch.from_numpy(normalised.astype(np.float32)).unsqueeze(0)
        except Exception as e:
            print(f"[ENHANCER] pyloudnorm error: {e} — using peak normalisation")
            return self._peak_normalise(w)

    def _peak_normalise(self, w: torch.Tensor) -> torch.Tensor:
        """Peak-based normalisation: scale so peak sits at -1 dBFS."""
        peak = w.abs().max().item()
        if peak < 1e-8:
            return w
        target_peak = 10 ** (-1.0 / 20)   # -1 dBFS
        return w * (target_peak / peak)

    def _rms_normalise(self, w: torch.Tensor) -> torch.Tensor:
        """RMS normalisation fallback."""
        rms = w.pow(2).mean().sqrt().item()
        if rms < 1e-8:
            return w
        target_rms = 10 ** (self.TARGET_LUFS / 20)
        w = w * (target_rms / rms)
        peak = w.abs().max().item()
        if peak > 0.99:
            w = w * (0.99 / peak)
        return w

    def _true_peak_limit(self, w: torch.Tensor) -> torch.Tensor:
        """
        True peak limiter using 4x oversampling.
        Ensures inter-sample peaks never exceed -1 dBTP —
        critical for MP3/AAC encoding without distortion.
        """
        ceiling = 10 ** (self.TRUE_PEAK_DBTP / 20)
        w_up    = self._resample(w, self.TARGET_SR, self.TARGET_SR * 4)
        tp      = w_up.abs().max().item()
        if tp > ceiling:
            w = w * (ceiling / tp)
        return w

    def _strip_silence(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            frame_size = int(sr * 0.02)
            n_frames   = w.shape[1] // frame_size
            if n_frames < 2:
                return w
            energies = torch.tensor([
                w[:, i * frame_size:(i + 1) * frame_size].abs().mean().item()
                for i in range(n_frames)
            ])
            voiced = (energies > 0.001).nonzero(as_tuple=True)[0]
            if len(voiced) < 2:
                return w
            start   = int(voiced[0].item()) * frame_size
            end     = min(w.shape[1], (int(voiced[-1].item()) + 1) * frame_size)
            trimmed = w[:, start:end]
            return trimmed if trimmed.shape[1] >= sr * 0.1 else w
        except Exception:
            return w

    def _remove_internal_silence(self, w: torch.Tensor, sr: int, max_gap_ms: int = 500) -> torch.Tensor:
        frame_size     = int(sr * 0.02)
        threshold      = 0.005
        max_gap_frames = int(max_gap_ms / 20)
        n_frames       = w.shape[1] // frame_size

        energy = torch.tensor([
            w[:, i * frame_size:(i + 1) * frame_size].pow(2).mean().sqrt().item()
            for i in range(n_frames)
        ])
        voiced    = energy > threshold
        segments: list[torch.Tensor] = []
        silence_run = 0

        for i in range(n_frames):
            if voiced[i]:
                if 0 < silence_run <= max_gap_frames:
                    start = (i - silence_run) * frame_size
                    segments.append(w[:, start:i * frame_size])
                segments.append(w[:, i * frame_size:(i + 1) * frame_size])
                silence_run = 0
            else:
                silence_run += 1

        return torch.cat(segments, dim=1) if segments else w

    # ------------------------------------------------------------------ #
    #  Public entry point
    # ------------------------------------------------------------------ #

    async def enhance(self, input_path: str, recording_id: str, job_id: str) -> EnhancementResult:
        loop = asyncio.get_event_loop()

        waveform, sr   = self._load_audio(input_path)
        raw_clone      = waveform.clone()
        clipping_input = self._detect_clipping(waveform)

        mono = self._to_mono(waveform)
        mono = self._resample(mono, sr, self.TARGET_SR)

        # Stage 1 — De-reverberation (room echo / reverb removal)
        enhanced = await loop.run_in_executor(None, self._dereverberate, mono, self.TARGET_SR)

        # Stage 2 — DeepFilterNet pass 1 (primary noise removal)
        enhanced = await loop.run_in_executor(None, self._deepfilter_denoise, enhanced)

        # Stage 2b — DeepFilterNet pass 2 (second pass catches what pass 1 missed;
        # after pass 1 lowers the noise floor, the model can suppress more on pass 2)
        enhanced = await loop.run_in_executor(None, self._deepfilter_denoise, enhanced)

        # Stage 3 — Demucs vocals extraction (background music / instruments)
        enhanced = await loop.run_in_executor(
            None, self._demucs_extract_vocals, enhanced.repeat(2, 1), self.TARGET_SR
        )

        # Stage 3b — Residual cleanup: double-pass noisereduce on post-Demucs signal
        # Bark and other sounds that bled through Demucs' vocals stem are eliminated here
        enhanced = await loop.run_in_executor(None, self._residual_cleanup, enhanced)

        # Stage 4 — Noise gate (silence non-speech frames between sentences)
        enhanced = await loop.run_in_executor(None, self._noise_gate, enhanced)

        # Stage 5 — De-click (remove plosives, mic bumps, electrical clicks)
        enhanced = await loop.run_in_executor(None, self._declick, enhanced)

        # Stage 6 — EQ (high-pass, low-pass, mud cut, presence boost)
        enhanced = self._apply_eq(enhanced, self.TARGET_SR)

        # Stage 7 — De-essing (control sibilance 7–9 kHz)
        enhanced = self._de_ess(enhanced, self.TARGET_SR)

        # Stage 8 — Dynamic range compression (even out loud/quiet passages)
        enhanced = await loop.run_in_executor(None, self._compress, enhanced)

        # Stage 9 — Silence trim & internal gap cleanup
        enhanced = self._strip_silence(enhanced, self.TARGET_SR)
        enhanced = self._remove_internal_silence(enhanced, self.TARGET_SR)

        # Stage 10 — ITU-R BS.1770 LUFS normalisation (-16 LUFS target)
        enhanced = await loop.run_in_executor(None, self._normalise_lufs, enhanced)

        # Stage 11 — True peak limiter (-1 dBTP — broadcast safe)
        enhanced = self._true_peak_limit(enhanced)

        raw_mono      = self._to_mono(raw_clone)
        raw_at_target = self._resample(raw_mono, sr, self.TARGET_SR)
        snr           = self._compute_snr(raw_at_target, enhanced)
        lufs          = self._compute_lufs(enhanced)
        peak_db       = 20 * np.log10(enhanced.abs().max().item() + 1e-8)
        quality_score = self._compute_quality_score(snr, clipping_input, lufs)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
        torchaudio.save(out_path, enhanced, self.TARGET_SR)

        b2_key       = f"{settings.B2_ENHANCED_PREFIX}{recording_id}/{job_id}.wav"
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
