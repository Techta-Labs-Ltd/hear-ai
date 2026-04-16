import asyncio
import os
import tempfile
from dataclasses import dataclass

import numpy as np
import noisereduce as nr
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
    DFN_SR = 48_000
    DNS_SR = 16_000

    TARGET_LUFS = -16.0

    _HP_FREQ = 80.0
    _LP_FREQ = 14_000.0
    _EQ_CUT_FREQ = 200.0
    _EQ_CUT_GAIN = -2.0
    _EQ_CUT_Q = 1.4
    _EQ_BOOST_FREQ = 3_000.0
    _EQ_BOOST_GAIN = 2.0
    _EQ_BOOST_Q = 2.0

    _DESS_FREQ1 = 7_000.0
    _DESS_GAIN1 = -3.0
    _DESS_Q1 = 1.5
    _DESS_FREQ2 = 9_000.0
    _DESS_GAIN2 = -2.0
    _DESS_Q2 = 2.0

    def __init__(self):
        self._demucs = None
        self._dns = None
        self._dfn_model = None
        self._dfn_state = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self):
        self._demucs = get_model(settings.DEMUCS_MODEL)
        self._demucs.to(self._device)
        self._demucs.eval()

        self._dfn_model, self._dfn_state, _ = init_df()
        print("[ENHANCER] DeepFilterNet loaded")

        self._dns = dns_pretrained.dns64()
        self._dns.to(self._device)
        self._dns.eval()
        print("[ENHANCER] Facebook DNS denoiser loaded (dns64)")

        print("[ENHANCER] nara_wpe de-reverberation ready")

    @property
    def is_loaded(self) -> bool:
        return self._demucs is not None

    # ------------------------------------------------------------------ #
    #  I/O helpers
    # ------------------------------------------------------------------ #

    def _load_audio(self, path: str) -> tuple[torch.Tensor, int]:
        file_size = os.path.getsize(path)
        if file_size == 0:
            raise ValueError(f"Audio file is empty: {path}")

        for backend in ("soundfile", "ffmpeg", None):
            try:
                kwargs = {"backend": backend} if backend else {}
                waveform, sr = torchaudio.load(path, **kwargs)
                return waveform, sr
            except Exception:
                continue

        raise RuntimeError(
            f"Could not load audio from {path} ({file_size} bytes). "
            "Install soundfile or ffmpeg for torchaudio backend support."
        )

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
        min_len = min(raw.shape[1], enhanced.shape[1])
        noise = raw[:, :min_len] - enhanced[:, :min_len]
        signal_power = enhanced[:, :min_len].pow(2).mean().item()
        noise_power = noise.pow(2).mean().item() + 1e-10
        return 10 * np.log10(signal_power / noise_power)

    def _compute_lufs(self, w: torch.Tensor) -> float:
        rms = w.pow(2).mean().sqrt().item()
        return 20 * np.log10(rms + 1e-8)

    def _compute_quality_score(self, snr_db: float, clipping: bool, lufs: float) -> float:
        snr_score = min(1.0, max(0.0, (snr_db + 5) / 40))
        lufs_score = 1.0 - min(1.0, abs(lufs - self.TARGET_LUFS) / 20)
        clip_penalty = 0.3 if clipping else 0.0
        return round(max(0.0, snr_score * 0.6 + lufs_score * 0.4 - clip_penalty), 3)

    # ------------------------------------------------------------------ #
    #  Processing pipeline
    # ------------------------------------------------------------------ #

    def _dereverberate(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            audio_np = w.squeeze(0).numpy().astype(np.float64)
            Y = wpe_stft(audio_np, size=512, shift=128).T[None]
            Z = wpe_v8(Y, taps=10, delay=3, iterations=3)
            dereverbed = wpe_istft(Z[0].T, size=512, shift=128)
            dereverbed = dereverbed[: audio_np.shape[0]]
            return torch.from_numpy(dereverbed.astype(np.float32)).unsqueeze(0)
        except Exception:
            return w

    def _deepfilter_denoise(self, w: torch.Tensor) -> torch.Tensor:
        try:
            w_48k = self._resample(w, self.TARGET_SR, self.DFN_SR)
            audio_np = w_48k.squeeze(0).numpy()
            clean_np = df_enhance(self._dfn_model, self._dfn_state, audio_np)
            clean = torch.from_numpy(clean_np).unsqueeze(0)
            return self._resample(clean, self.DFN_SR, self.TARGET_SR)
        except Exception as e:
            print(f"[ENHANCER] DeepFilterNet error: {e} — falling back to DNS")
            return self._dns_denoise(w)

    def _dns_denoise(self, w: torch.Tensor) -> torch.Tensor:
        try:
            w_16k = self._resample(w, self.TARGET_SR, self.DNS_SR).to(self._device)
            with torch.no_grad():
                clean = self._dns(w_16k[None])[0].cpu()
            return self._resample(clean, self.DNS_SR, self.TARGET_SR)
        except Exception as e:
            print(f"[ENHANCER] DNS error: {e} — falling back to noisereduce")
            return self._nr_denoise(w)

    def _nr_denoise(self, w: torch.Tensor) -> torch.Tensor:
        try:
            audio_np = w.squeeze(0).numpy()
            reduced = nr.reduce_noise(
                y=audio_np, sr=self.TARGET_SR,
                stationary=False, prop_decrease=0.85,
            )
            return torch.from_numpy(reduced).unsqueeze(0)
        except Exception:
            return w

    def _demucs_extract_vocals(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if waveform.dim() == 2 and waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        resampled = self._resample(waveform, sr, self._demucs.samplerate)

        with torch.no_grad():
            sources = apply_model(
                self._demucs,
                resampled[None].to(self._device),
                progress=False,
            )[0]

        vocals_idx = self._demucs.sources.index("vocals")
        vocals = sources[vocals_idx].cpu()
        vocals = self._resample(vocals, self._demucs.samplerate, self.TARGET_SR)
        return self._to_mono(vocals)

    def _apply_eq(self, w: torch.Tensor, sr: int) -> torch.Tensor:
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
        try:
            w = F.equalizer_biquad(w, sr, center_freq=self._DESS_FREQ1,
                                   gain=self._DESS_GAIN1, Q=self._DESS_Q1)
            w = F.equalizer_biquad(w, sr, center_freq=self._DESS_FREQ2,
                                   gain=self._DESS_GAIN2, Q=self._DESS_Q2)
            return w
        except Exception:
            return w

    def _strip_silence(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            frame_size = int(sr * 0.02)
            n_frames = w.shape[1] // frame_size
            if n_frames < 2:
                return w

            energies = torch.tensor([
                w[:, i * frame_size:(i + 1) * frame_size].abs().mean().item()
                for i in range(n_frames)
            ])
            voiced = (energies > 0.001).nonzero(as_tuple=True)[0]
            if len(voiced) < 2:
                return w

            start = int(voiced[0].item()) * frame_size
            end = min(w.shape[1], (int(voiced[-1].item()) + 1) * frame_size)
            trimmed = w[:, start:end]
            return trimmed if trimmed.shape[1] >= sr * 0.1 else w
        except Exception:
            return w

    def _remove_internal_silence(self, w: torch.Tensor, sr: int, max_gap_ms: int = 500) -> torch.Tensor:
        frame_size = int(sr * 0.02)
        threshold = 0.005
        max_gap_frames = int(max_gap_ms / 20)

        n_frames = w.shape[1] // frame_size
        energy = torch.tensor([
            w[:, i * frame_size:(i + 1) * frame_size].pow(2).mean().sqrt().item()
            for i in range(n_frames)
        ])

        voiced = energy > threshold
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

    def _normalise_loudness(self, w: torch.Tensor, target_lufs: float = -16.0) -> torch.Tensor:
        rms = w.pow(2).mean().sqrt().item()
        if rms < 1e-8:
            return w
        target_rms = 10 ** (target_lufs / 20)
        w = w * (target_rms / rms)
        peak = w.abs().max().item()
        if peak > 0.99:
            w = w * (0.99 / peak)
        return w

    # ------------------------------------------------------------------ #
    #  Public entry point
    # ------------------------------------------------------------------ #

    async def enhance(self, input_path: str, recording_id: str, job_id: str) -> EnhancementResult:
        loop = asyncio.get_event_loop()

        waveform, sr = self._load_audio(input_path)
        raw_clone = waveform.clone()
        clipping_input = self._detect_clipping(waveform)

        mono = self._to_mono(waveform)
        mono = self._resample(mono, sr, self.TARGET_SR)

        # Stage 1: De-reverberation — remove room echo/reverb
        enhanced = await loop.run_in_executor(None, self._dereverberate, mono, self.TARGET_SR)

        # Stage 2: DeepFilterNet — SOTA speech enhancement, removes all noise types
        enhanced = await loop.run_in_executor(None, self._deepfilter_denoise, enhanced)

        # Stage 3: Demucs — isolate vocals stem, strip background music
        enhanced = await loop.run_in_executor(
            None, self._demucs_extract_vocals, enhanced.repeat(2, 1), self.TARGET_SR
        )

        # Stage 4: EQ, de-essing, silence cleanup, loudness normalisation
        enhanced = self._apply_eq(enhanced, self.TARGET_SR)
        enhanced = self._de_ess(enhanced, self.TARGET_SR)
        enhanced = self._strip_silence(enhanced, self.TARGET_SR)
        enhanced = self._remove_internal_silence(enhanced, self.TARGET_SR)
        enhanced = self._normalise_loudness(enhanced, self.TARGET_LUFS)

        raw_mono = self._to_mono(raw_clone)
        raw_at_target = self._resample(raw_mono, sr, self.TARGET_SR)
        snr = self._compute_snr(raw_at_target, enhanced)
        lufs = self._compute_lufs(enhanced)
        peak_db = 20 * np.log10(enhanced.abs().max().item() + 1e-8)
        quality_score = self._compute_quality_score(snr, clipping_input, lufs)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
        torchaudio.save(out_path, enhanced, self.TARGET_SR)

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
