import asyncio
import logging
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

warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

logger = logging.getLogger(__name__)


class ContentMode(str, Enum):
    SPEECH = "speech"
    MUSIC = "music"
    PODCAST = "podcast"
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


def _cosine_fade(n: int, device: torch.device) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, n, device=device)
    return (1.0 - torch.cos(t * torch.pi)) * 0.5


def _match_length(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(a.shape[-1], b.shape[-1])
    return a[..., :n], b[..., :n]


def _iir_envelope(sig: np.ndarray, attack_coef: float, release_coef: float) -> np.ndarray:
    env = np.zeros_like(sig)
    prev = 0.0
    for i, x in enumerate(sig):
        c = attack_coef if x > prev else release_coef
        prev = c * prev + (1.0 - c) * x
        env[i] = prev
    return env


class AudioEnhancer:
    TARGET_SR = 44100
    DFN_SR = 48_000
    VAD_SR = 16_000
    TARGET_LUFS = -16.0
    TRUE_PEAK_DBTP = -1.0

    _SPEECH_COMP_THRESHOLD_DB = -18.0
    _SPEECH_COMP_RATIO = 2.5
    _SPEECH_COMP_MAKEUP_DB = 1.5
    _SPEECH_COMP_ATTACK_MS = 5
    _SPEECH_COMP_RELEASE_MS = 80

    _MUSIC_COMP_THRESHOLD_DB = -12.0
    _MUSIC_COMP_RATIO = 1.8
    _MUSIC_COMP_MAKEUP_DB = 0.5
    _MUSIC_COMP_ATTACK_MS = 20
    _MUSIC_COMP_RELEASE_MS = 200

    _PODCAST_COMP_THRESHOLD_DB = -16.0
    _PODCAST_COMP_RATIO = 2.0
    _PODCAST_COMP_MAKEUP_DB = 1.0
    _PODCAST_COMP_ATTACK_MS = 10
    _PODCAST_COMP_RELEASE_MS = 120

    _GATE_THRESHOLD_DB = -65.0
    _GATE_ATTACK_MS = 1
    _GATE_RELEASE_MS = 150
    _GATE_HOLD_MS = 50

    _VAD_SPEECH_THRESHOLD = 0.35
    _VAD_MIN_SPEECH_MS = 100
    _VAD_MIN_SILENCE_MS = 400
    _VAD_PAD_MS = 120
    _VAD_CROSSFADE_MS = 20
    _VAD_SUPPRESS_FLOOR = 0.08

    _STRIP_FADE_MS = 25
    _CLASSIFY_WINDOW_S = 30
    _PODCAST_STEM_ATTENUATION_DB = -12.0
    _MUSIC_BED_MIN_RMS = 1e-4
    _MUSIC_LUFS = -14.0

    def __init__(self):
        self._demucs = None
        self._dfn_model = None
        self._dfn_state = None
        self._vad_model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._vad_lock = threading.Lock()
        self._dfn_lock = threading.Lock()

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

    def _detect_mode_from_stems(self, stems: dict[str, torch.Tensor]) -> ContentMode:
        drums_rms = stems["drums"].pow(2).mean().sqrt().item()
        bass_rms = stems["bass"].pow(2).mean().sqrt().item()
        other_rms = stems["other"].pow(2).mean().sqrt().item()
        vocals_rms = stems["vocals"].pow(2).mean().sqrt().item()

        music_power = drums_rms + bass_rms + other_rms
        total_power = music_power + vocals_rms + 1e-10
        rhythm_power = drums_rms + bass_rms

        # No vocals at all → pure music/instrumental
        if vocals_rms < 1e-5:
            return ContentMode.MUSIC

        # Strong rhythm/bass and sparse vocals → music
        if rhythm_power / total_power > 0.4 and vocals_rms / total_power < 0.2:
            return ContentMode.MUSIC

        # Significant voice + some music → podcast / mixed content
        if vocals_rms / total_power > 0.3 and music_power / total_power > 0.15:
            return ContentMode.PODCAST

        # Voice dominant, minimal music → clean speech
        return ContentMode.SPEECH

    def _compute_snr(self, raw: torch.Tensor, enhanced: torch.Tensor) -> float:
        raw, enhanced = _match_length(raw.cpu(), enhanced.cpu())
        sig_p = enhanced.pow(2).mean().item()
        noi_p = (raw - enhanced).pow(2).mean().item() + 1e-10
        return 10 * np.log10(max(sig_p, 1e-10) / noi_p)

    def _compute_lufs(self, w: torch.Tensor) -> float:
        try:
            meter = pyln.Meter(self.TARGET_SR)
            loudness = meter.integrated_loudness(w.cpu().squeeze(0).numpy().astype(np.float64))
            return loudness if np.isfinite(loudness) else -99.0
        except Exception:
            rms = w.pow(2).mean().sqrt().item()
            return float(20 * np.log10(rms + 1e-8))

    def _compute_quality_score(self, snr_db: float, clipping: bool, lufs: float) -> float:
        snr_score = min(1.0, max(0.0, (snr_db + 5) / 40))
        lufs_score = 1.0 - min(1.0, abs(lufs - self.TARGET_LUFS) / 20)
        clip_pen = 0.3 if clipping else 0.0
        return round(max(0.0, snr_score * 0.6 + lufs_score * 0.4 - clip_pen), 3)

    def _deepfilter_denoise(self, w: torch.Tensor) -> torch.Tensor:
        original_len = w.shape[1]
        try:
            w_cpu = w.cpu()
            w_48k = self._resample(w_cpu, self.TARGET_SR, self.DFN_SR)
            with self._dfn_lock:
                _, fresh_state, _ = init_df()
                with torch.no_grad():
                    clean = df_enhance(self._dfn_model, fresh_state, w_48k)
            if isinstance(clean, np.ndarray):
                clean = torch.from_numpy(clean.copy())
            if clean.dim() == 1:
                clean = clean.unsqueeze(0)
            clean = self._resample(clean.float(), self.DFN_SR, self.TARGET_SR).to(self._device)
            if clean.shape[1] < original_len:
                pad = torch.zeros((1, original_len - clean.shape[1]), device=clean.device)
                clean = torch.cat([clean, pad], dim=1)
            return clean[:, :original_len]
        except Exception as e:
            logger.error("DeepFilterNet failed: %s", e)
            return w

    def _demucs_separate(self, waveform: torch.Tensor, sr: int) -> dict[str, torch.Tensor]:
        original_len = waveform.shape[1]
        stereo = waveform.repeat(2, 1) if waveform.shape[0] == 1 else waveform
        resampled = self._resample(stereo, sr, self._demucs.samplerate)
        with torch.no_grad():
            sources = apply_model(
                self._demucs, resampled[None],
                progress=False, num_workers=0,
            )[0]
        result: dict[str, torch.Tensor] = {}
        for i, name in enumerate(self._demucs.sources):
            stem = self._resample(sources[i], self._demucs.samplerate, self.TARGET_SR)
            stem = self._to_mono(stem)
            if stem.shape[1] < original_len:
                pad = torch.zeros((1, original_len - stem.shape[1]), device=stem.device)
                stem = torch.cat([stem, pad], dim=1)
            result[name] = stem[:, :original_len]
        return result

    def _apply_eq_speech(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            w = F.highpass_biquad(w, sr, cutoff_freq=100.0)
            w = F.equalizer_biquad(w, sr, center_freq=300.0, gain=-2.0, Q=1.5)
            w = F.equalizer_biquad(w, sr, center_freq=1000.0, gain=1.0, Q=2.0)
            w = F.equalizer_biquad(w, sr, center_freq=3000.0, gain=2.5, Q=1.8)
            w = F.equalizer_biquad(w, sr, center_freq=8000.0, gain=1.5, Q=2.0)
            w = F.lowpass_biquad(w, sr, cutoff_freq=12000.0)
            return w
        except Exception:
            return w

    def _apply_eq_music(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            w = F.highpass_biquad(w, sr, cutoff_freq=30.0)
            w = F.equalizer_biquad(w, sr, center_freq=100.0, gain=1.0, Q=1.2)
            w = F.equalizer_biquad(w, sr, center_freq=3000.0, gain=-1.0, Q=2.0)
            w = F.equalizer_biquad(w, sr, center_freq=8000.0, gain=2.0, Q=1.5)
            return w
        except Exception:
            return w

    def _iir_coefs(self, time_ms: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
        coef = np.exp(-1.0 / (sr * time_ms / 1000.0))
        b = np.array([1.0 - coef])
        a = np.array([1.0, -coef])
        return b, a

    def _noise_gate(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            threshold_lin = 10 ** (self._GATE_THRESHOLD_DB / 20)
            hold_samples = int(sr * self._GATE_HOLD_MS / 1000)

            sig_np = w.squeeze(0).cpu().numpy().astype(np.float64)
            abs_sig = np.abs(sig_np)

            b_a, a_a = self._iir_coefs(self._GATE_ATTACK_MS, sr)
            b_r, a_r = self._iir_coefs(self._GATE_RELEASE_MS, sr)

            env = _iir_envelope(abs_sig, float(1.0 - b_a[0]), float(1.0 - b_r[0]))

            gate_open = env > threshold_lin
            held = np.zeros(len(gate_open), dtype=bool)
            counter = 0
            for i in range(len(gate_open)):
                if gate_open[i]:
                    counter = hold_samples
                    held[i] = True
                elif counter > 0:
                    held[i] = True
                    counter -= 1

            target_np = held.astype(np.float64)
            b_gs, a_gs = self._iir_coefs(self._GATE_ATTACK_MS, sr)
            b_gr, a_gr = self._iir_coefs(self._GATE_RELEASE_MS, sr)

            gain_np = np.zeros_like(target_np)
            prev_g = 1.0
            for i, t in enumerate(target_np):
                c_b, c_a = (b_gs, a_gs) if t > prev_g else (b_gr, a_gr)
                c = float(1.0 - c_b[0])
                prev_g = c * prev_g + (1.0 - c) * t
                gain_np[i] = prev_g

            gain = torch.from_numpy(gain_np.astype(np.float32)).unsqueeze(0).to(w.device)
            return w * gain
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
                    sampling_rate=self.VAD_SR,
                    threshold=self._VAD_SPEECH_THRESHOLD,
                    min_speech_duration_ms=self._VAD_MIN_SPEECH_MS,
                    min_silence_duration_ms=self._VAD_MIN_SILENCE_MS,
                    return_seconds=False,
                )

        total_n = w.shape[1]
        mask_np = np.zeros(total_n, dtype=np.float32)
        pad_16k = int(self.VAD_SR * self._VAD_PAD_MS / 1000)
        min_len_16k = int(self.VAD_SR * self._VAD_MIN_SPEECH_MS / 1000)
        max_16k = w_16k.shape[0]
        scale = sr / self.VAD_SR
        xfade_n = int(sr * self._VAD_CROSSFADE_MS / 1000)

        scaled_timestamps = []
        for ts in timestamps:
            if (ts["end"] - ts["start"]) < min_len_16k:
                continue
            s16 = max(0, ts["start"] - pad_16k)
            e16 = min(max_16k, ts["end"] + pad_16k)
            s44 = min(int(round(s16 * scale)), total_n - 1)
            e44 = min(int(round(e16 * scale)), total_n)
            if e44 <= s44:
                continue

            fade_in_end = min(s44 + xfade_n, e44)
            fade_out_st = max(e44 - xfade_n, fade_in_end)
            fi_n = fade_in_end - s44
            fo_n = e44 - fade_out_st

            if fi_n > 0:
                t = np.linspace(0.0, 1.0, fi_n)
                mask_np[s44:fade_in_end] = np.maximum(
                    mask_np[s44:fade_in_end], (1.0 - np.cos(t * np.pi)) * 0.5
                )
            if fade_in_end < fade_out_st:
                mask_np[fade_in_end:fade_out_st] = 1.0
            if fo_n > 0:
                t = np.linspace(0.0, 1.0, fo_n)
                mask_np[fade_out_st:e44] = np.maximum(
                    mask_np[fade_out_st:e44], (1.0 + np.cos(t * np.pi)) * 0.5
                )

            scaled_timestamps.append({"start": s44, "end": e44})

        mask = torch.from_numpy(mask_np).unsqueeze(0).to(self._device)
        return mask, scaled_timestamps

    def _vad_suppress(self, w: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w, mask = _match_length(w, mask)
        floor = self._VAD_SUPPRESS_FLOOR
        return w * (floor + mask * (1.0 - floor))

    def _strip_silence(self, w: torch.Tensor, timestamps: list[dict], sr: int) -> torch.Tensor:
        try:
            if not timestamps:
                return w
            start = max(0, timestamps[0]["start"])
            end = min(w.shape[1], timestamps[-1]["end"])
            if end <= start:
                return w
            trimmed = w[:, start:end].clone()
            fade_n = max(1, int(sr * self._STRIP_FADE_MS / 1000))
            if trimmed.shape[1] > fade_n * 2:
                trimmed[0, :fade_n] *= _cosine_fade(fade_n, w.device)
                trimmed[0, -fade_n:] *= _cosine_fade(fade_n, w.device).flip(0)
            return trimmed
        except Exception:
            return w

    def _compress(self, w: torch.Tensor, sr: int, mode: ContentMode) -> torch.Tensor:
        try:
            if mode == ContentMode.MUSIC:
                threshold_db = self._MUSIC_COMP_THRESHOLD_DB
                ratio = self._MUSIC_COMP_RATIO
                makeup_db = self._MUSIC_COMP_MAKEUP_DB
                attack_ms = self._MUSIC_COMP_ATTACK_MS
                release_ms = self._MUSIC_COMP_RELEASE_MS
            elif mode == ContentMode.PODCAST:
                threshold_db = self._PODCAST_COMP_THRESHOLD_DB
                ratio = self._PODCAST_COMP_RATIO
                makeup_db = self._PODCAST_COMP_MAKEUP_DB
                attack_ms = self._PODCAST_COMP_ATTACK_MS
                release_ms = self._PODCAST_COMP_RELEASE_MS
            else:
                threshold_db = self._SPEECH_COMP_THRESHOLD_DB
                ratio = self._SPEECH_COMP_RATIO
                makeup_db = self._SPEECH_COMP_MAKEUP_DB
                attack_ms = self._SPEECH_COMP_ATTACK_MS
                release_ms = self._SPEECH_COMP_RELEASE_MS

            threshold_lin = 10 ** (threshold_db / 20)
            attack_coef = np.exp(-1.0 / (sr * attack_ms / 1000))
            release_coef = np.exp(-1.0 / (sr * release_ms / 1000))
            makeup_lin = 10 ** (makeup_db / 20)

            sig_np = w.squeeze(0).cpu().numpy().astype(np.float64)
            env_np = _iir_envelope(np.abs(sig_np), attack_coef, release_coef)

            gain_np = np.ones_like(env_np)
            over = env_np > threshold_lin
            gain_np[over] = (
                (threshold_lin + (env_np[over] - threshold_lin) / ratio)
                / (env_np[over] + 1e-12)
            )

            b_r, a_r = self._iir_coefs(release_ms, sr)
            gain_np = ss.lfilter(b_r, a_r, gain_np)
            gain_np = np.clip(gain_np, 0.0, 1.0)

            gain = torch.from_numpy(gain_np.astype(np.float32)).unsqueeze(0).to(w.device)
            out = w * gain * makeup_lin
            peak = out.abs().max().item()
            if peak > 0.99:
                out = out * (0.99 / peak)
            return out
        except Exception:
            return w

    def _normalise_lufs(self, w: torch.Tensor) -> torch.Tensor:
        try:
            if w.shape[1] < int(self.TARGET_SR * 0.5):
                return self._peak_normalise(w)
            meter = pyln.Meter(self.TARGET_SR)
            audio_np = w.cpu().squeeze(0).numpy().astype(np.float64)
            loudness = meter.integrated_loudness(audio_np)
            if not np.isfinite(loudness) or loudness < -70.0 or loudness > 0.0:
                return self._peak_normalise(w)
            if abs(loudness - self.TARGET_LUFS) <= 0.5:
                return w
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                normalised = pyln.normalize.loudness(audio_np, loudness, self.TARGET_LUFS)
            if not np.isfinite(normalised).all():
                return self._peak_normalise(w)
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
        return w * (10 ** (self.TRUE_PEAK_DBTP / 20) / peak)

    def _true_peak_limit(self, w: torch.Tensor) -> torch.Tensor:
        ceiling = 10 ** (self.TRUE_PEAK_DBTP / 20)
        w_up = self._resample(w, self.TARGET_SR, self.TARGET_SR * 4)
        tp_up = w_up.abs().max().item()
        if tp_up > ceiling:
            w = w * (ceiling / tp_up)
        return w

    def _normalise_lufs_stereo(self, w: torch.Tensor, target_lufs: float = -14.0) -> torch.Tensor:
        try:
            if w.shape[-1] < int(self.TARGET_SR * 0.5):
                return self._peak_normalise(w)
            meter = pyln.Meter(self.TARGET_SR)
            audio_np = w.cpu().numpy().astype(np.float64)
            if audio_np.ndim == 1:
                pass
            elif audio_np.shape[0] == 1:
                audio_np = audio_np.squeeze(0)
            else:
                audio_np = audio_np.T  # pyloudnorm expects (samples, channels)
            loudness = meter.integrated_loudness(audio_np)
            if not np.isfinite(loudness) or loudness < -70.0 or loudness > 0.0:
                return self._peak_normalise(w)
            if abs(loudness - target_lufs) <= 0.5:
                return w
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                normalised = pyln.normalize.loudness(audio_np, loudness, target_lufs)
            if not np.isfinite(normalised).all():
                return self._peak_normalise(w)
            if normalised.ndim == 1:
                result = torch.from_numpy(normalised.astype(np.float32)).unsqueeze(0)
            else:
                result = torch.from_numpy(normalised.T.astype(np.float32))  # back to (channels, samples)
            result = result.to(self._device)
            peak = result.abs().max().item()
            if peak > 0.99:
                result = result * (0.99 / peak)
            return result
        except Exception:
            return self._peak_normalise(w)

    _PODCAST_STEM_ATTENUATION_DB = -12.0
    _MUSIC_BED_MIN_RMS = 1e-4
    _MUSIC_LUFS = -14.0

    async def enhance(
        self,
        input_path: str,
        recording_id: str,
        job_id: str,
        mode: ContentMode = ContentMode.AUTO,
    ) -> EnhancementResult:
        loop = asyncio.get_running_loop()
        out_path = None

        try:
            waveform, sr = self._load_audio(input_path)
            raw_clone = waveform.clone()
            clipping_input = self._detect_clipping(waveform)

            # Auto-detect content type using a short window to avoid full-file Demucs
            if mode == ContentMode.AUTO:
                mono_detect = self._to_mono(waveform)
                mono_detect = self._resample(mono_detect, sr, self.TARGET_SR).to(self._device)
                window_len = int(self._CLASSIFY_WINDOW_S * self.TARGET_SR)
                clip = mono_detect[:, :window_len]
                window_stems = await loop.run_in_executor(None, self._demucs_separate, clip, self.TARGET_SR)
                mode = self._detect_mode_from_stems(window_stems)
                logger.info("[ENHANCE] Auto-detected mode=%s job=%s", mode.value, job_id)

            logger.info("[ENHANCE] mode=%s job=%s", mode.value, job_id)

            # ── MUSIC: stereo passthrough, no destructive processing ──────────────
            if mode == ContentMode.MUSIC:
                stereo = waveform
                if sr != self.TARGET_SR:
                    stereo = self._resample(stereo, sr, self.TARGET_SR)
                stereo = stereo.to(self._device)

                raw_mono = self._resample(self._to_mono(raw_clone), sr, self.TARGET_SR)
                snr = self._compute_snr(raw_mono, self._to_mono(stereo))

                stereo = await loop.run_in_executor(
                    None, self._normalise_lufs_stereo, stereo, self._MUSIC_LUFS
                )
                stereo = await loop.run_in_executor(None, self._true_peak_limit, stereo)
                enhanced = stereo

            # ── SPEECH / PODCAST: full voice-focused processing ───────────────────
            else:
                mono = self._to_mono(waveform)
                enhanced = self._resample(mono, sr, self.TARGET_SR).to(self._device)
                raw_at_target = self._resample(self._to_mono(raw_clone), sr, self.TARGET_SR)

                separated = await loop.run_in_executor(
                    None, self._demucs_separate, enhanced, self.TARGET_SR
                )

                vocals = separated["vocals"]

                # Neural noise removal on isolated vocals
                vocals = await loop.run_in_executor(None, self._deepfilter_denoise, vocals)

                # VAD: detect and suppress non-speech regions
                vad_mask, speech_timestamps = await loop.run_in_executor(
                    None, self._generate_vad_mask, vocals, self.TARGET_SR
                )
                vocals = await loop.run_in_executor(None, self._vad_suppress, vocals, vad_mask)

                # Clarity EQ and noise gate
                vocals = await loop.run_in_executor(None, self._apply_eq_speech, vocals, self.TARGET_SR)
                vocals = await loop.run_in_executor(None, self._noise_gate, vocals, self.TARGET_SR)

                if mode == ContentMode.SPEECH:
                    # Strip leading/trailing silence for pure speech recordings
                    vocals = await loop.run_in_executor(
                        None, self._strip_silence, vocals, speech_timestamps, self.TARGET_SR
                    )
                    enhanced = vocals
                    snr = self._compute_snr(raw_at_target, enhanced)

                else:  # PODCAST
                    drums_rms = separated["drums"].pow(2).mean().sqrt().item()
                    bass_rms = separated["bass"].pow(2).mean().sqrt().item()
                    has_music_bed = (drums_rms > self._MUSIC_BED_MIN_RMS or bass_rms > self._MUSIC_BED_MIN_RMS)

                    if has_music_bed:
                        attenuation = 10 ** (self._PODCAST_STEM_ATTENUATION_DB / 20)
                        music_stems = [
                            separated[k] * attenuation
                            for k in ("drums", "bass", "other")
                        ]
                        min_len = min(vocals.shape[1], *(s.shape[1] for s in music_stems))
                        music_mix = sum(s[:, :min_len] for s in music_stems)
                        voc_t, nv_t = _match_length(vocals, music_mix)
                        enhanced = voc_t + nv_t
                        logger.info("[ENHANCE] Podcast: music bed detected, mixed back at %sdB", self._PODCAST_STEM_ATTENUATION_DB)
                    else:
                        enhanced = vocals
                        logger.info("[ENHANCE] Podcast: no music bed detected, voice-only output")

                    snr = self._compute_snr(raw_at_target, enhanced)

                enhanced = await loop.run_in_executor(None, self._compress, enhanced, self.TARGET_SR, mode)
                enhanced = await loop.run_in_executor(None, self._normalise_lufs, enhanced)
                enhanced = await loop.run_in_executor(None, self._true_peak_limit, enhanced)

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
                mode_used=mode.value,
            )

        except Exception:
            if out_path and os.path.exists(out_path):
                try:
                    os.unlink(out_path)
                except OSError:
                    pass
            raise