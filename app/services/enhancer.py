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

try:
    from pyrnnoise import RNNoise as _RNNoise
    _RNNOISE_AVAILABLE = True
except Exception:
    _RNNoise = None
    _RNNOISE_AVAILABLE = False
    logger.warning("pyrnnoise unavailable — RNNoise stage will be bypassed")
class ContentMode(str, Enum):
    SPEECH  = "speech"
    MUSIC   = "music"
    PODCAST = "podcast"
    AUTO    = "auto"


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


def _cosine_fade(n: int, device: torch.device) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, n, device=device)
    return (1.0 - torch.cos(t * torch.pi)) * 0.5


def _match_length(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(a.shape[-1], b.shape[-1])
    return a[..., :n], b[..., :n]


def _iir_envelope(sig: np.ndarray, attack_coef: float, release_coef: float) -> np.ndarray:
    env  = np.zeros_like(sig)
    prev = 0.0
    for i, x in enumerate(sig):
        c      = attack_coef if x > prev else release_coef
        prev   = c * prev + (1.0 - c) * x
        env[i] = prev
    return env


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
    _MUSIC_COMP_ATTACK_MS     = 20
    _MUSIC_COMP_RELEASE_MS    = 200

    _GATE_THRESHOLD_DB = -72.0
    _GATE_ATTACK_MS    = 1
    _GATE_RELEASE_MS   = 200
    _GATE_HOLD_MS      = 80

    _VAD_SPEECH_THRESHOLD  = 0.30
    _VAD_MIN_SPEECH_MS     = 100
    _VAD_MIN_SILENCE_MS    = 600
    _VAD_PAD_MS            = 150
    _VAD_CROSSFADE_MS      = 25
    _VAD_SUPPRESS_FLOOR    = 0.12

    _BREATH_ENERGY_THRESHOLD  = 0.002
    _BREATH_MAX_DURATION_MS   = 180
    _BREATH_CROSSFADE_MS      = 15

    _STRIP_FADE_MS               = 30
    _CLASSIFY_WINDOW_S           = 30
    _PODCAST_STEM_ATTENUATION_DB = -8.0

    def __init__(self):
        self._demucs    = None
        self._dfn_model = None
        self._dfn_state = None
        self._vad_model = None
        self._device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._vad_lock  = threading.Lock()
        self._dfn_lock  = threading.Lock()
        self._rnn      = _RNNoise(sample_rate=48000) if _RNNOISE_AVAILABLE else None
        self._rnn_lock  = threading.Lock()

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

    def _iir_coefs(self, time_ms: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
        coef = np.exp(-1.0 / (sr * time_ms / 1000.0))
        b    = np.array([1.0 - coef])
        a    = np.array([1.0, -coef])
        return b, a

    def _detect_content_mode(
        self, w: torch.Tensor, sr: int, precomputed_ts: list[dict] | None = None
    ) -> ContentMode:
        max_samples = int(self._CLASSIFY_WINDOW_S * sr)
        segment     = w[:, :max_samples] if w.shape[1] > max_samples else w
        n_fft       = 2048
        hop         = 512
        window      = torch.hann_window(n_fft, device=segment.device)
        stft        = torch.stft(
            segment.squeeze(0), n_fft=n_fft, hop_length=hop,
            win_length=n_fft, window=window, return_complex=True,
        )
        mag   = stft.abs()
        freqs = torch.linspace(0, sr / 2, mag.shape[0], device=segment.device)

        speech_energy  = mag[(freqs >= 250) & (freqs <= 4000)].pow(2).mean().item()
        music_energy   = mag[(freqs > 5000) & (freqs <= 16000)].pow(2).mean().item()
        total_energy   = speech_energy + music_energy + 1e-10
        spectral_ratio = music_energy / total_energy

        if precomputed_ts is not None:
            ts = precomputed_ts
            total_16k = int(segment.shape[1] * self.VAD_SR / sr)
        else:
            w_16k = self._resample(segment.cpu(), sr, self.VAD_SR).squeeze(0)
            with self._vad_lock:
                with torch.no_grad():
                    ts = get_speech_timestamps(
                        w_16k, self._vad_model,
                        sampling_rate=self.VAD_SR,
                        threshold=self._VAD_SPEECH_THRESHOLD,
                        return_seconds=False,
                    )
            total_16k = w_16k.shape[0]

        speech_samples = sum(t["end"] - t["start"] for t in ts)
        speech_ratio   = speech_samples / max(total_16k, 1)

        if spectral_ratio > 0.35 and speech_ratio > 0.40:
            return ContentMode.PODCAST
        if spectral_ratio > 0.65 and speech_ratio <= 0.40:
            return ContentMode.MUSIC
        return ContentMode.SPEECH

    def _compute_snr(self, raw: torch.Tensor, enhanced: torch.Tensor) -> float:
        raw, enhanced = _match_length(raw.cpu(), enhanced.cpu())
        sig_p = enhanced.pow(2).mean().item()
        noi_p = (raw - enhanced).pow(2).mean().item() + 1e-10
        return 10 * np.log10(max(sig_p, 1e-10) / noi_p)

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
                pad   = torch.zeros((1, original_len - clean.shape[1]), device=clean.device)
                clean = torch.cat([clean, pad], dim=1)
            return clean[:, :original_len]
        except Exception as e:
            logger.error("DeepFilterNet failed: %s", e)
            return w

    def _rnnoise_denoise(self, w: torch.Tensor) -> torch.Tensor:
        if not _RNNOISE_AVAILABLE or self._rnn is None:
            logger.warning("RNNoise unavailable — skipping transient suppression stage")
            return w
        try:
            target_sr = 48000
            x48       = self._resample(w.cpu(), self.TARGET_SR, target_sr).squeeze().numpy()
            x16       = (np.clip(x48, -1.0, 1.0) * 32767).astype(np.int16)

            out_chunks: list[np.ndarray] = []
            with self._rnn_lock:
                for _, denoised in self._rnn.denoise_chunk(x16[np.newaxis, :]):
                    out_chunks.append(denoised)

            if not out_chunks:
                return w

            out16 = np.concatenate(out_chunks, axis=-1).squeeze()
            out_f = out16.astype(np.float32) / 32767.0
            pad   = len(x48) - len(out_f)
            if pad > 0:
                out_f = np.pad(out_f, (0, pad))
            else:
                out_f = out_f[:len(x48)]

            y = torch.from_numpy(out_f).unsqueeze(0).to(self._device)
            return self._resample(y, target_sr, self.TARGET_SR)
        except Exception as e:
            logger.error("RNNoise failed: %s", e)
            return w

    def _demucs_separate(self, waveform: torch.Tensor, sr: int) -> dict[str, torch.Tensor]:
        original_len = waveform.shape[1]
        stereo       = waveform.repeat(2, 1) if waveform.shape[0] == 1 else waveform
        resampled    = self._resample(stereo, sr, self._demucs.samplerate)
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
                pad  = torch.zeros((1, original_len - stem.shape[1]), device=stem.device)
                stem = torch.cat([stem, pad], dim=1)
            result[name] = stem[:, :original_len]
        return result

    def _generate_vad_mask(self, w: torch.Tensor, sr: int) -> tuple[torch.Tensor, list[dict]]:
        w_16k = self._resample(w.cpu(), sr, self.VAD_SR)
        if w_16k.dim() > 1:
            w_16k = w_16k.squeeze(0)

        with self._vad_lock:
            with torch.no_grad():
                timestamps = get_speech_timestamps(
                    w_16k,
                    self._vad_model,
                    sampling_rate           = self.VAD_SR,
                    threshold               = self._VAD_SPEECH_THRESHOLD,
                    min_speech_duration_ms  = self._VAD_MIN_SPEECH_MS,
                    min_silence_duration_ms = self._VAD_MIN_SILENCE_MS,
                    return_seconds          = False,
                )

        total_n     = w.shape[1]
        mask_np     = np.zeros(total_n, dtype=np.float32)
        pad_16k     = int(self.VAD_SR * self._VAD_PAD_MS / 1000)
        min_len_16k = int(self.VAD_SR * self._VAD_MIN_SPEECH_MS / 1000)
        max_16k     = w_16k.shape[0]
        scale       = sr / self.VAD_SR
        xfade_n     = int(sr * self._VAD_CROSSFADE_MS / 1000)

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
            fi_n        = fade_in_end - s44
            fo_n        = e44 - fade_out_st

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

    def _dual_vad_mask(
        self, raw: torch.Tensor, dfn: torch.Tensor, sr: int
    ) -> tuple[torch.Tensor, list[dict]]:
        mask_raw, ts_raw = self._generate_vad_mask(raw, sr)
        mask_dfn, ts_dfn = self._generate_vad_mask(dfn, sr)
        mask_raw, mask_dfn = _match_length(mask_raw, mask_dfn)
        union_mask = torch.maximum(mask_raw, mask_dfn)
        seen = set()
        merged_ts = []
        for ts in ts_raw + ts_dfn:
            key = (ts["start"], ts["end"])
            if key not in seen:
                seen.add(key)
                merged_ts.append(ts)
        merged_ts.sort(key=lambda x: x["start"])
        return union_mask, merged_ts

    def _vad_suppress(self, w: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w, mask = _match_length(w, mask)
        floor   = self._VAD_SUPPRESS_FLOOR
        return w * (floor + mask * (1.0 - floor))

    def _noise_gate(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            window_n   = int(sr * 0.10)
            sig_np     = w.squeeze(0).cpu().numpy().astype(np.float64)
            local_rms  = np.sqrt(np.mean(sig_np[:window_n] ** 2) + 1e-12)
            bypass_thr = 10 ** (-30.0 / 20)
            if local_rms > bypass_thr:
                return w

            threshold_lin = 10 ** (self._GATE_THRESHOLD_DB / 20)
            hold_samples  = int(sr * self._GATE_HOLD_MS / 1000)
            abs_sig       = np.abs(sig_np)
            b_a, a_a      = self._iir_coefs(self._GATE_ATTACK_MS, sr)
            b_r, a_r      = self._iir_coefs(self._GATE_RELEASE_MS, sr)
            env           = _iir_envelope(abs_sig, float(1.0 - b_a[0]), float(1.0 - b_r[0]))

            gate_open = env > threshold_lin
            held      = np.zeros(len(gate_open), dtype=bool)
            counter   = 0
            for i in range(len(gate_open)):
                if gate_open[i]:
                    counter = hold_samples
                    held[i] = True
                elif counter > 0:
                    held[i] = True
                    counter -= 1

            target_np  = held.astype(np.float64)
            b_gs, a_gs = self._iir_coefs(self._GATE_ATTACK_MS, sr)
            b_gr, a_gr = self._iir_coefs(self._GATE_RELEASE_MS, sr)
            gain_np    = np.zeros_like(target_np)
            prev_g     = 1.0
            for i, t in enumerate(target_np):
                c_b, c_a = (b_gs, a_gs) if t > prev_g else (b_gr, a_gr)
                c          = float(1.0 - c_b[0])
                prev_g     = c * prev_g + (1.0 - c) * t
                gain_np[i] = prev_g

            gain = torch.from_numpy(gain_np.astype(np.float32)).unsqueeze(0).to(w.device)
            return w * gain
        except Exception:
            return w

    def _remove_breaths_and_smacks(
        self, w: torch.Tensor, sr: int, vad_mask: torch.Tensor
    ) -> torch.Tensor:
        try:
            w, vad_mask = _match_length(w, vad_mask)
            sig_np      = w.squeeze(0).cpu().numpy().astype(np.float64)
            mask_np     = vad_mask.squeeze(0).cpu().numpy().astype(np.float32)

            window_n     = max(1, int(sr * 0.010))
            n_windows    = len(sig_np) // window_n
            energy       = np.array([
                np.abs(sig_np[i * window_n:(i + 1) * window_n]).mean()
                for i in range(n_windows)
            ])

            max_breath_w = max(1, int(sr * self._BREATH_MAX_DURATION_MS / 1000 / window_n))
            xfade_n      = max(1, int(sr * self._BREATH_CROSSFADE_MS / 1000))
            result_np    = sig_np.copy()

            i = 0
            while i < len(energy):
                if energy[i] > self._BREATH_ENERGY_THRESHOLD:
                    j = i
                    while j < len(energy) and energy[j] > self._BREATH_ENERGY_THRESHOLD:
                        j += 1
                    duration_w = j - i
                    s_samp     = i * window_n
                    e_samp     = min(j * window_n, len(result_np))

                    event_mask_vals = mask_np[s_samp:e_samp]
                    is_non_speech   = np.all(event_mask_vals < 0.2) if len(event_mask_vals) > 0 else False

                    if duration_w <= max_breath_w and is_non_speech:
                        fi_s = max(0, s_samp - xfade_n)
                        fo_e = min(len(result_np), e_samp + xfade_n)

                        if fi_s < s_samp:
                            fn = s_samp - fi_s
                            t  = np.linspace(0.0, 1.0, fn)
                            result_np[fi_s:s_samp] *= (1.0 + np.cos(t * np.pi)) * 0.5

                        result_np[s_samp:e_samp] = 0.0

                        if e_samp < fo_e:
                            fn = fo_e - e_samp
                            t  = np.linspace(0.0, 1.0, fn)
                            result_np[e_samp:fo_e] *= (1.0 - np.cos(t * np.pi)) * 0.5

                    i = j
                else:
                    i += 1

            return torch.from_numpy(result_np.astype(np.float32)).unsqueeze(0).to(w.device)
        except Exception:
            return w

    def _strip_silence(self, w: torch.Tensor, vad_mask: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            w, vad_mask   = _match_length(w, vad_mask)
            mask_np       = vad_mask.squeeze(0).cpu().numpy()
            min_silence_n = int(sr * 0.300)
            threshold     = 0.05
            fade_n        = max(1, int(sr * self._STRIP_FADE_MS / 1000))
            total         = len(mask_np)

            head = 0
            run  = 0
            for k in range(total):
                if mask_np[k] <= threshold:
                    run += 1
                else:
                    run = 0
                if run >= min_silence_n:
                    head = k + 1

            tail = total
            run  = 0
            for k in range(total - 1, -1, -1):
                if mask_np[k] <= threshold:
                    run += 1
                else:
                    run = 0
                if run >= min_silence_n:
                    tail = k

            if tail <= head:
                return w

            trimmed = w[:, head:tail].clone()
            if trimmed.shape[1] > fade_n * 2:
                trimmed[0, :fade_n]  *= _cosine_fade(fade_n, w.device)
                trimmed[0, -fade_n:] *= _cosine_fade(fade_n, w.device).flip(0)
            return trimmed
        except Exception:
            return w

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
            w = F.equalizer_biquad(w, sr, center_freq=100.0,  gain=1.0,  Q=1.2)
            w = F.equalizer_biquad(w, sr, center_freq=3000.0, gain=-1.0, Q=2.0)
            w = F.equalizer_biquad(w, sr, center_freq=8000.0, gain=2.0,  Q=1.5)
            return w
        except Exception:
            return w

    def _compress(self, w: torch.Tensor, sr: int, mode: ContentMode) -> torch.Tensor:
        try:
            if mode == ContentMode.MUSIC:
                threshold_db = self._MUSIC_COMP_THRESHOLD_DB
                ratio        = self._MUSIC_COMP_RATIO
                makeup_db    = self._MUSIC_COMP_MAKEUP_DB
                attack_ms    = self._MUSIC_COMP_ATTACK_MS
                release_ms   = self._MUSIC_COMP_RELEASE_MS
            else:
                threshold_db = self._SPEECH_COMP_THRESHOLD_DB
                ratio        = self._SPEECH_COMP_RATIO
                makeup_db    = self._SPEECH_COMP_MAKEUP_DB
                attack_ms    = self._SPEECH_COMP_ATTACK_MS
                release_ms   = self._SPEECH_COMP_RELEASE_MS

            threshold_lin = 10 ** (threshold_db / 20)
            attack_coef   = np.exp(-1.0 / (sr * attack_ms / 1000))
            release_coef  = np.exp(-1.0 / (sr * release_ms / 1000))
            makeup_lin    = 10 ** (makeup_db / 20)

            sig_np  = w.squeeze(0).cpu().numpy().astype(np.float64)
            env_np  = _iir_envelope(np.abs(sig_np), attack_coef, release_coef)

            gain_np      = np.ones_like(env_np)
            over         = env_np > threshold_lin
            gain_np[over] = (
                (threshold_lin + (env_np[over] - threshold_lin) / ratio)
                / (env_np[over] + 1e-12)
            )

            b_r, a_r = self._iir_coefs(release_ms, sr)
            gain_np  = ss.lfilter(b_r, a_r, gain_np)
            gain_np  = np.clip(gain_np, 0.0, 1.0)

            gain = torch.from_numpy(gain_np.astype(np.float32)).unsqueeze(0).to(w.device)
            out  = w * gain * makeup_lin
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
            meter    = pyln.Meter(self.TARGET_SR)
            audio_np = w.cpu().squeeze(0).numpy().astype(np.float64)
            loudness = meter.integrated_loudness(audio_np)
            if not np.isfinite(loudness) or loudness > 0.0 or loudness < -70.0:
                return self._peak_normalise(w)
            if abs(loudness - self.TARGET_LUFS) <= 0.5:
                return w
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                normalised = pyln.normalize.loudness(audio_np, loudness, self.TARGET_LUFS)
            if not np.isfinite(normalised).all():
                return self._peak_normalise(w)
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
        input_path:   str,
        recording_id: str,
        job_id:       str,
        mode:         ContentMode = ContentMode.AUTO,
    ) -> EnhancementResult:
        loop     = asyncio.get_running_loop()
        out_path = None

        try:
            waveform, sr   = self._load_audio(input_path)
            raw_clone      = waveform.clone()
            clipping_input = self._detect_clipping(waveform)

            mono     = self._to_mono(waveform)
            enhanced = self._resample(mono, sr, self.TARGET_SR).to(self._device)

            if mode == ContentMode.AUTO:
                mode = await loop.run_in_executor(
                    None, self._detect_content_mode, enhanced, self.TARGET_SR
                )

            raw_at_target = self._resample(self._to_mono(raw_clone), sr, self.TARGET_SR)

            if mode == ContentMode.SPEECH:
                dfn_cleaned = await loop.run_in_executor(None, self._deepfilter_denoise, enhanced)

                vad_mask, speech_timestamps = await loop.run_in_executor(
                    None, self._dual_vad_mask, enhanced, dfn_cleaned, self.TARGET_SR
                )

                noise_energy = (
                    (enhanced - dfn_cleaned[:, :enhanced.shape[1]]).pow(2).mean().sqrt().item()
                )
                if noise_energy > 0.005:
                    separated   = await loop.run_in_executor(
                        None, self._demucs_separate, dfn_cleaned, self.TARGET_SR
                    )
                    enhanced = separated["vocals"]
                else:
                    enhanced = dfn_cleaned

                enhanced = await loop.run_in_executor(
                    None, self._rnnoise_denoise, enhanced
                )
                enhanced = await loop.run_in_executor(
                    None, self._noise_gate, enhanced, self.TARGET_SR
                )
                enhanced = await loop.run_in_executor(
                    None, self._vad_suppress, enhanced, vad_mask
                )
                enhanced = await loop.run_in_executor(
                    None, self._remove_breaths_and_smacks, enhanced, self.TARGET_SR, vad_mask
                )

                snr = self._compute_snr(raw_at_target, enhanced)

                enhanced = await loop.run_in_executor(
                    None, self._strip_silence, enhanced, vad_mask, self.TARGET_SR
                )
                enhanced = await loop.run_in_executor(
                    None, self._apply_eq_speech, enhanced, self.TARGET_SR
                )
                enhanced = await loop.run_in_executor(
                    None, self._compress, enhanced, self.TARGET_SR, mode
                )

            elif mode == ContentMode.PODCAST:
                dfn_cleaned = await loop.run_in_executor(None, self._deepfilter_denoise, enhanced)

                vad_mask, _ = await loop.run_in_executor(
                    None, self._dual_vad_mask, enhanced, dfn_cleaned, self.TARGET_SR
                )

                separated = await loop.run_in_executor(
                    None, self._demucs_separate, dfn_cleaned, self.TARGET_SR
                )

                vocals   = separated["vocals"]
                non_keys = [k for k in separated if k != "vocals"]

                vocals = await loop.run_in_executor(
                    None, self._noise_gate, vocals, self.TARGET_SR
                )
                vocals = await loop.run_in_executor(
                    None, self._vad_suppress, vocals, vad_mask
                )
                vocals = await loop.run_in_executor(
                    None, self._remove_breaths_and_smacks, vocals, self.TARGET_SR, vad_mask
                )
                vocals = await loop.run_in_executor(
                    None, self._apply_eq_speech, vocals, self.TARGET_SR
                )
                vocals = await loop.run_in_executor(
                    None, self._compress, vocals, self.TARGET_SR, ContentMode.SPEECH
                )

                attenuation = 10 ** (self._PODCAST_STEM_ATTENUATION_DB / 20)
                non_vocal_stems = [separated[k] * attenuation for k in non_keys]
                min_len = min(vocals.shape[1], *(s.shape[1] for s in non_vocal_stems))
                non_vocal_mix = sum(s[:, :min_len] for s in non_vocal_stems)
                non_vocal_mix = await loop.run_in_executor(
                    None, self._apply_eq_music, non_vocal_mix, self.TARGET_SR
                )

                voc_trim, nv_trim = _match_length(vocals, non_vocal_mix)
                enhanced = voc_trim + nv_trim

                snr = self._compute_snr(raw_at_target, enhanced)

                enhanced = await loop.run_in_executor(
                    None, self._strip_silence, enhanced, vad_mask, self.TARGET_SR
                )

            else:
                dfn_cleaned = await loop.run_in_executor(None, self._deepfilter_denoise, enhanced)
                separated   = await loop.run_in_executor(
                    None, self._demucs_separate, dfn_cleaned, self.TARGET_SR
                )
                stems   = list(separated.values())
                min_len = min(s.shape[1] for s in stems)
                enhanced = sum(s[:, :min_len] for s in stems)

                snr = self._compute_snr(raw_at_target, enhanced)

                enhanced = await loop.run_in_executor(
                    None, self._apply_eq_music, enhanced, self.TARGET_SR
                )
                enhanced = await loop.run_in_executor(
                    None, self._compress, enhanced, self.TARGET_SR, mode
                )

            enhanced = await loop.run_in_executor(None, self._normalise_lufs, enhanced)
            enhanced = await loop.run_in_executor(None, self._true_peak_limit, enhanced)

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

        except Exception:
            if out_path and os.path.exists(out_path):
                try:
                    os.unlink(out_path)
                except OSError:
                    pass
            raise