import asyncio
import os
import tempfile
import threading
import warnings
from dataclasses import dataclass
from enum import Enum

import logging
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

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

from app.config import settings
from app.core.storage import storage


class ContentMode(str, Enum):
    SPEECH = "speech"
    MUSIC  = "music"
    AUTO   = "auto"


@dataclass
class EnhancementResult:
    b2_key:            str
    enhanced_url:      str
    local_path:        str
    quality_score:     float
    snr_db:            float
    peak_db:           float
    lufs:              float
    clipping_detected: bool
    mode_used:         str


class AudioEnhancer:
    TARGET_SR      = 44100
    DFN_SR         = 48_000
    VAD_SR         = 16_000
    TARGET_LUFS    = -16.0
    TRUE_PEAK_DBTP = -1.0

    _SPEECH_COMP_THRESHOLD_DB = -18.0
    _SPEECH_COMP_RATIO        = 2.5
    _SPEECH_COMP_MAKEUP_DB    = 1.5
    _SPEECH_COMP_ATTACK_MS    = 5
    _SPEECH_COMP_RELEASE_MS   = 80

    _MUSIC_COMP_THRESHOLD_DB  = -12.0
    _MUSIC_COMP_RATIO         = 1.8
    _MUSIC_COMP_MAKEUP_DB     = 0.5

    _VAD_SPEECH_THRESHOLD     = 0.4
    _VAD_MIN_SPEECH_MS        = 120
    _VAD_MIN_SILENCE_MS       = 300
    _VAD_PAD_MS               = 80
    _VAD_SUPPRESS_FLOOR       = 0.02

    _GATE_THRESHOLD_DB        = -50.0
    _GATE_ATTACK_MS           = 2
    _GATE_RELEASE_MS          = 100

    _BREATH_ENERGY_THRESHOLD  = 0.004
    _BREATH_MAX_DURATION_MS   = 220

    _MUSIC_ENERGY_RATIO       = 0.65

    def __init__(self):
        self._demucs    = None
        self._dfn_model = None
        self._dfn_state = None
        self._vad_model = None
        self._device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._vad_lock  = threading.Lock()
        self._dfn_lock  = threading.Lock()

    def load(self):
        self._demucs = get_model(settings.DEMUCS_MODEL)
        self._demucs.to(self._device)
        self._demucs.eval()

        self._dfn_model, self._dfn_state, _ = init_df()

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

    def _detect_content_mode(self, w: torch.Tensor, sr: int) -> ContentMode:
        n_fft    = 2048
        hop      = 512
        window   = torch.hann_window(n_fft, device=w.device)
        stft     = torch.stft(w.squeeze(0), n_fft=n_fft, hop_length=hop, win_length=n_fft,
                               window=window, return_complex=True)
        mag      = stft.abs()
        freqs    = torch.linspace(0, sr / 2, mag.shape[0], device=w.device)

        speech_band = ((freqs >= 100) & (freqs <= 4000))
        music_band  = ((freqs > 4000) & (freqs <= 16000))

        speech_energy = mag[speech_band].pow(2).mean().item()
        music_energy  = mag[music_band].pow(2).mean().item()
        total_energy  = speech_energy + music_energy + 1e-10

        if music_energy / total_energy > self._MUSIC_ENERGY_RATIO:
            return ContentMode.MUSIC
        return ContentMode.SPEECH

    def _compute_snr(self, raw: torch.Tensor, enhanced: torch.Tensor) -> float:
        raw      = raw.cpu()
        enhanced = enhanced.cpu()
        n        = min(raw.shape[1], enhanced.shape[1])
        sig_p    = enhanced[:, :n].pow(2).mean().item()
        noi_p    = (raw[:, :n] - enhanced[:, :n]).pow(2).mean().item() + 1e-10
        return 10 * np.log10(sig_p / noi_p)

    def _compute_lufs(self, w: torch.Tensor) -> float:
        try:
            meter    = pyln.Meter(self.TARGET_SR)
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
            w_cpu = w.cpu()
            w_48k = self._resample(w_cpu, self.TARGET_SR, self.DFN_SR)
            with self._dfn_lock:
                _, fresh_state, _ = init_df()
                with torch.no_grad():
                    clean = df_enhance(self._dfn_model, fresh_state, w_48k)
            if isinstance(clean, np.ndarray):
                clean = torch.from_numpy(clean)
            if clean.dim() == 1:
                clean = clean.unsqueeze(0)
            return self._resample(clean.float(), self.DFN_SR, self.TARGET_SR).to(self._device)
        except Exception as e:
            logger.error("DeepFilterNet failed: %s", e)
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

    def _demucs_separate_music(self, waveform: torch.Tensor, sr: int) -> dict[str, torch.Tensor]:
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        resampled = self._resample(waveform, sr, self._demucs.samplerate)
        with torch.no_grad():
            sources = apply_model(self._demucs, resampled[None], progress=False)[0]
        return {
            name: self._resample(sources[i], self._demucs.samplerate, self.TARGET_SR)
            for i, name in enumerate(self._demucs.sources)
        }

    def _apply_eq_speech(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            w = F.highpass_biquad(w, sr, cutoff_freq=100.0)
            w = F.equalizer_biquad(w, sr, center_freq=300.0,  gain=-2.0, Q=1.5)
            w = F.equalizer_biquad(w, sr, center_freq=1000.0, gain=1.0,  Q=2.0)
            w = F.equalizer_biquad(w, sr, center_freq=3000.0, gain=2.5,  Q=1.8)
            w = F.equalizer_biquad(w, sr, center_freq=8000.0, gain=1.5,  Q=2.0)
            w = F.lowpass_biquad(w, sr, cutoff_freq=12000.0)
            return w
        except Exception:
            return w

    def _apply_eq_music(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            w = F.highpass_biquad(w, sr, cutoff_freq=30.0)
            w = F.equalizer_biquad(w, sr, center_freq=100.0,  gain=1.0, Q=1.2)
            w = F.equalizer_biquad(w, sr, center_freq=3000.0, gain=-1.0, Q=2.0)
            w = F.equalizer_biquad(w, sr, center_freq=8000.0, gain=2.0,  Q=1.5)
            return w
        except Exception:
            return w

    def _noise_gate(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            threshold_lin  = 10 ** (self._GATE_THRESHOLD_DB / 20)
            attack_samples = max(1, int(sr * self._GATE_ATTACK_MS / 1000))
            release_samples = max(1, int(sr * self._GATE_RELEASE_MS / 1000))

            env = F_nn.avg_pool1d(
                w.abs(), kernel_size=attack_samples, stride=1,
                padding=attack_samples // 2
            )
            gate_open  = (env > threshold_lin).float()
            smoothed   = F_nn.avg_pool1d(
                gate_open, kernel_size=release_samples, stride=1,
                padding=release_samples // 2
            )
            gate       = torch.clamp(smoothed / (threshold_lin + 1e-9) * env, 0.0, 1.0)
            smooth_k   = max(3, int(sr * 0.005))
            if smooth_k % 2 == 0:
                smooth_k += 1
            gate       = F_nn.avg_pool1d(gate, smooth_k, stride=1, padding=smooth_k // 2)
            return w * gate
        except Exception:
            return w

    def _generate_vad_mask(self, w: torch.Tensor, sr: int) -> tuple[torch.Tensor, list[dict]]:
        w_16k = self._resample(w.cpu(), sr, self.VAD_SR)
        if w_16k.dim() > 1:
            w_16k = w_16k.squeeze(0)

        with self._vad_lock:
            with torch.no_grad():
                timestamps = get_speech_timestamps(
                    w_16k,
                    self._vad_model,
                    sampling_rate        = self.VAD_SR,
                    threshold            = self._VAD_SPEECH_THRESHOLD,
                    min_speech_duration_ms = self._VAD_MIN_SPEECH_MS,
                    min_silence_duration_ms = self._VAD_MIN_SILENCE_MS,
                    return_seconds       = False,
                )

        total_samples_44k = w.shape[1]
        mask              = torch.zeros((1, total_samples_44k), device=self._device)
        pad_samples_16k   = int(self.VAD_SR * self._VAD_PAD_MS / 1000)
        min_len_16k       = int(self.VAD_SR * self._VAD_MIN_SPEECH_MS / 1000)
        max_16k           = w_16k.shape[0]
        scale             = sr / self.VAD_SR

        scaled_timestamps = []
        for ts in timestamps:
            if (ts["end"] - ts["start"]) < min_len_16k:
                continue
            s16 = max(0, ts["start"] - pad_samples_16k)
            e16 = min(max_16k, ts["end"] + pad_samples_16k)
            s44 = min(int(s16 * scale), total_samples_44k)
            e44 = min(int(e16 * scale), total_samples_44k)
            mask[0, s44:e44] = 1.0
            scaled_timestamps.append({"start": s44, "end": e44})

        smooth_k = max(3, int(sr * 0.02))
        if smooth_k % 2 == 0:
            smooth_k += 1
        mask = F_nn.avg_pool1d(mask, smooth_k, stride=1, padding=smooth_k // 2)
        return torch.clamp(mask, 0.0, 1.0), scaled_timestamps

    def _vad_suppress(self, w: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        min_len    = min(w.shape[1], mask.shape[1])
        floor      = self._VAD_SUPPRESS_FLOOR
        suppression = floor + (mask[:, :min_len] * (1.0 - floor))
        return w[:, :min_len] * suppression

    def _remove_breaths_and_smacks(self, w: torch.Tensor, sr: int, vad_mask: torch.Tensor) -> torch.Tensor:
        try:
            min_len      = min(w.shape[1], vad_mask.shape[1])
            w            = w[:, :min_len]
            mask         = vad_mask[:, :min_len]

            non_speech   = (mask < 0.3).squeeze(0)
            window_ms    = 10
            window_n     = max(1, int(sr * window_ms / 1000))
            energy       = F_nn.avg_pool1d(
                w.abs(), kernel_size=window_n, stride=window_n, padding=0
            ).squeeze(0)

            max_breath_n = int(sr * self._BREATH_MAX_DURATION_MS / 1000 / window_n)
            threshold    = self._BREATH_ENERGY_THRESHOLD
            event_mask   = (energy > threshold).squeeze(0) if energy.dim() > 1 else (energy > threshold)

            scale_factor = window_n
            result       = w.clone()

            i = 0
            while i < len(event_mask):
                if event_mask[i]:
                    j = i
                    while j < len(event_mask) and event_mask[j]:
                        j += 1
                    duration_windows = j - i
                    if duration_windows <= max_breath_n:
                        s = i * scale_factor
                        e = min(j * scale_factor, result.shape[1])
                        if non_speech[s] if s < len(non_speech) else True:
                            fade = torch.linspace(1.0, 0.0, e - s, device=w.device)
                            result[0, s:e] = result[0, s:e] * fade * 0.05
                    i = j
                else:
                    i += 1

            return result
        except Exception:
            return w

    def _strip_silence(self, w: torch.Tensor, timestamps: list[dict], sr: int) -> torch.Tensor:
        try:
            if not timestamps:
                return w
            start    = max(0, timestamps[0]["start"])
            end      = min(w.shape[1], timestamps[-1]["end"])
            trimmed  = w[:, start:end].clone()
            fade_len = max(1, int(sr * 0.015))
            if trimmed.shape[1] > fade_len * 2:
                fade_in  = torch.linspace(0.0, 1.0, fade_len, device=w.device)
                fade_out = torch.linspace(1.0, 0.0, fade_len, device=w.device)
                trimmed[0, :fade_len]  *= fade_in
                trimmed[0, -fade_len:] *= fade_out
            return trimmed
        except Exception:
            return w

    def _compress(self, w: torch.Tensor, sr: int, mode: ContentMode) -> torch.Tensor:
        try:
            if mode == ContentMode.MUSIC:
                threshold_db  = self._MUSIC_COMP_THRESHOLD_DB
                ratio         = self._MUSIC_COMP_RATIO
                makeup_db     = self._MUSIC_COMP_MAKEUP_DB
                attack_ms     = 20
                release_ms    = 200
            else:
                threshold_db  = self._SPEECH_COMP_THRESHOLD_DB
                ratio         = self._SPEECH_COMP_RATIO
                makeup_db     = self._SPEECH_COMP_MAKEUP_DB
                attack_ms     = self._SPEECH_COMP_ATTACK_MS
                release_ms    = self._SPEECH_COMP_RELEASE_MS

            threshold_lin  = 10 ** (threshold_db / 20)
            attack_n       = max(1, int(sr * attack_ms / 1000))
            release_n      = max(1, int(sr * release_ms / 1000))
            if attack_n % 2 == 0:
                attack_n += 1
            if release_n % 2 == 0:
                release_n += 1

            env  = F_nn.avg_pool1d(
                w.abs().unsqueeze(1), kernel_size=attack_n, stride=1,
                padding=attack_n // 2
            ).squeeze(1)

            gain = torch.ones_like(w)
            over = env > threshold_lin
            gain[over] = (threshold_lin + (env[over] - threshold_lin) / ratio) / (env[over] + 1e-9)

            gain = F_nn.avg_pool1d(
                gain.unsqueeze(1), kernel_size=release_n, stride=1,
                padding=release_n // 2
            ).squeeze(1)

            return w * gain * (10 ** (makeup_db / 20))
        except Exception:
            return w

    def _normalise_lufs(self, w: torch.Tensor) -> torch.Tensor:
        try:
            meter    = pyln.Meter(self.TARGET_SR)
            audio_np = w.cpu().squeeze(0).numpy().astype(np.float64)
            loudness = meter.integrated_loudness(audio_np)
            if not np.isfinite(loudness) or loudness < -70:
                return self._peak_normalise(w)
            if abs(loudness - self.TARGET_LUFS) <= 0.5:
                return w
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                normalised = pyln.normalize.loudness(audio_np, loudness, self.TARGET_LUFS)
            result = torch.from_numpy(normalised.astype(np.float32)).unsqueeze(0).to(self._device)
            peak   = result.abs().max().item()
            if peak > 0.99:
                result = result * (0.99 / peak)
            return result
        except Exception:
            return self._peak_normalise(w)

    def _peak_normalise(self, w: torch.Tensor) -> torch.Tensor:
        peak = w.abs().max().item()
        if peak < 1e-8:
            return w
        return w * (10 ** (self.TRUE_PEAK_DBTP / 20) / peak)

    def _true_peak_limit(self, w: torch.Tensor) -> torch.Tensor:
        ceiling = 10 ** (self.TRUE_PEAK_DBTP / 20)
        w_up    = self._resample(w, self.TARGET_SR, self.TARGET_SR * 4)
        tp_up   = w_up.abs().max().item()
        if tp_up > ceiling:
            w = w * (ceiling / tp_up)
        return w

    async def enhance(
        self,
        input_path:  str,
        recording_id: str,
        job_id:       str,
        mode:         ContentMode = ContentMode.AUTO,
    ) -> EnhancementResult:
        loop = asyncio.get_event_loop()

        waveform, sr   = self._load_audio(input_path)
        raw_clone      = waveform.clone()
        clipping_input = self._detect_clipping(waveform)

        mono     = self._to_mono(waveform)
        enhanced = self._resample(mono, sr, self.TARGET_SR).to(self._device)

        if mode == ContentMode.AUTO:
            mode = await loop.run_in_executor(
                None, self._detect_content_mode, enhanced, self.TARGET_SR
            )

        vad_mask, speech_timestamps = await loop.run_in_executor(
            None, self._generate_vad_mask, enhanced, self.TARGET_SR
        )

        dfn_cleaned = await loop.run_in_executor(None, self._deepfilter_denoise, enhanced)

        if mode == ContentMode.SPEECH:
            _n = min(enhanced.shape[1], dfn_cleaned.shape[1])
            noise_energy = (enhanced[:, :_n] - dfn_cleaned[:, :_n]).pow(2).mean().sqrt().item()
            if noise_energy > 0.005:
                dfn_cleaned = await loop.run_in_executor(
                    None, self._demucs_extract_vocals, dfn_cleaned, self.TARGET_SR
                )
            enhanced = dfn_cleaned
            enhanced = await loop.run_in_executor(
                None, self._vad_suppress, enhanced, vad_mask
            )
            enhanced = await loop.run_in_executor(
                None, self._remove_breaths_and_smacks, enhanced, self.TARGET_SR, vad_mask
            )
            enhanced = await loop.run_in_executor(
                None, self._strip_silence, enhanced, speech_timestamps, self.TARGET_SR
            )
            enhanced = await loop.run_in_executor(
                None, self._apply_eq_speech, enhanced, self.TARGET_SR
            )
            enhanced = await loop.run_in_executor(
                None, self._noise_gate, enhanced, self.TARGET_SR
            )
            enhanced = await loop.run_in_executor(
                None, self._compress, enhanced, self.TARGET_SR, mode
            )

        else:
            separated = await loop.run_in_executor(
                None, self._demucs_separate_music, enhanced, self.TARGET_SR
            )
            stems     = [separated[s] for s in self._demucs.sources]
            enhanced  = sum(self._to_mono(s) for s in stems)
            enhanced  = dfn_cleaned if dfn_cleaned.shape[1] == enhanced.shape[1] else enhanced
            enhanced  = await loop.run_in_executor(
                None, self._apply_eq_music, enhanced, self.TARGET_SR
            )
            enhanced  = await loop.run_in_executor(
                None, self._compress, enhanced, self.TARGET_SR, mode
            )

        enhanced = await loop.run_in_executor(None, self._normalise_lufs, enhanced)
        enhanced = await loop.run_in_executor(None, self._true_peak_limit, enhanced)

        raw_at_target = self._resample(self._to_mono(raw_clone), sr, self.TARGET_SR)
        snr           = self._compute_snr(raw_at_target, enhanced)
        lufs          = self._compute_lufs(enhanced)
        peak_db       = 20 * np.log10(enhanced.abs().max().item() + 1e-8)
        quality_score = self._compute_quality_score(snr, clipping_input, lufs)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
        torchaudio.save(out_path, enhanced.cpu(), self.TARGET_SR)

        b2_key       = f"{settings.B2_ENHANCED_PREFIX}{recording_id}/{job_id}.wav"
        enhanced_url = await loop.run_in_executor(None, storage.upload_file, out_path, b2_key)

        return EnhancementResult(
            b2_key            = b2_key,
            enhanced_url      = enhanced_url,
            local_path        = out_path,
            quality_score     = quality_score,
            snr_db            = round(snr, 2),
            peak_db           = round(peak_db, 2),
            lufs              = round(lufs, 2),
            clipping_detected = clipping_input,
            mode_used         = mode.value,
        )