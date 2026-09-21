import warnings
from collections import deque

import numpy as np
import pyloudnorm as pyln
import torch
from scipy.ndimage import uniform_filter1d

from ..models import ContentMode
from .audio_io import AudioIO


class DynamicsProcessor:
    TARGET_LUFS = -16.0
    TRUE_PEAK_DBTP = -1.0

    SPEECH_COMP_THRESHOLD_DB = -24.0
    SPEECH_COMP_RATIO = 2.5
    SPEECH_COMP_MAKEUP_DB = 1.0
    SPEECH_COMP_ATTACK_MS = 5
    SPEECH_COMP_RELEASE_MS = 80

    MUSIC_COMP_THRESHOLD_DB = -12.0
    MUSIC_COMP_RATIO = 1.8
    MUSIC_COMP_MAKEUP_DB = 0.5
    MUSIC_COMP_ATTACK_MS = 20
    MUSIC_COMP_RELEASE_MS = 200

    LIMITER_LOOKAHEAD_MS = 5
    LIMITER_RELEASE_MS = 50

    LEVEL_BLOCK_MS = 500
    LEVEL_TARGET_LUFS = -16.0
    LEVEL_MAX_GAIN_DB = 6.0
    LEVEL_SMOOTH_MS = 1000

    def __init__(self, device: torch.device):
        self._device = device

    def _compress_pass(
        self,
        w: torch.Tensor,
        sr: int,
        threshold_db: float,
        ratio: float,
        makeup_db: float,
        attack_ms: float,
        release_ms: float,
    ) -> torch.Tensor:
        if sr <= 0 or ratio < 1.0 or attack_ms <= 0 or release_ms <= 0:
            raise ValueError("invalid compressor configuration")

        was_mono_vector = w.ndim == 1
        working = w.unsqueeze(0) if was_mono_vector else w
        if working.ndim != 2:
            raise ValueError("compressor expects [channels, samples] audio")
        if working.shape[-1] == 0:
            return w
        if not torch.isfinite(working).all():
            raise ValueError("compressor input contains non-finite samples")

        attack_coef = np.exp(-1.0 / (sr * attack_ms / 1000))
        release_coef = np.exp(-1.0 / (sr * release_ms / 1000))
        makeup_lin = 10 ** (makeup_db / 20)

        signal = working.detach().cpu().numpy().astype(np.float64)


        detector = np.max(np.abs(signal), axis=0)
        envelope = np.empty_like(detector)
        envelope[0] = detector[0]
        for index in range(1, detector.size):
            coefficient = attack_coef if detector[index] > envelope[index - 1] else release_coef
            envelope[index] = (
                coefficient * envelope[index - 1] + (1.0 - coefficient) * detector[index]
            )

        level_db = 20.0 * np.log10(np.maximum(envelope, 1e-12))
        target_gain_db = np.zeros_like(level_db)
        over = level_db > threshold_db
        target_gain_db[over] = (
            threshold_db + (level_db[over] - threshold_db) / ratio - level_db[over]
        )



        smoothed_gain_db = np.empty_like(target_gain_db)
        smoothed_gain_db[0] = target_gain_db[0]
        for index in range(1, target_gain_db.size):
            previous = smoothed_gain_db[index - 1]
            coefficient = attack_coef if target_gain_db[index] < previous else release_coef
            smoothed_gain_db[index] = (
                coefficient * previous + (1.0 - coefficient) * target_gain_db[index]
            )

        gain = torch.from_numpy(np.power(10.0, smoothed_gain_db / 20.0).astype(np.float32)).to(
            device=working.device, dtype=working.dtype
        )
        out = working * gain.unsqueeze(0) * makeup_lin
        peak = out.abs().max().item()
        if peak > 0.99:
            out = out * (0.99 / peak)
        if not torch.isfinite(out).all():
            raise RuntimeError("compressor produced non-finite samples")
        return out.squeeze(0) if was_mono_vector else out

    def level_loudness(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        """Short-term loudness leveling for consistent volume.

        Measures loudness in overlapping blocks and applies smooth gain
        corrections so every section hits the target LUFS.
        """
        try:
            was_mono_vector = w.ndim == 1
            working = w.unsqueeze(0) if was_mono_vector else w
            if working.ndim != 2:
                raise ValueError("loudness leveling expects [channels, samples] audio")
            sig = working.detach().cpu().numpy().astype(np.float64)
            n = sig.shape[-1]
            meter = pyln.Meter(sr)

            block_size = int(sr * self.LEVEL_BLOCK_MS / 1000)
            smooth_size = int(sr * self.LEVEL_SMOOTH_MS / 1000)
            max_gain_lin = 10 ** (self.LEVEL_MAX_GAIN_DB / 20)

            if n < block_size * 2:
                return w

            n_blocks = n // block_size
            gains = np.ones(n_blocks + 1)

            for i in range(n_blocks):
                start = i * block_size
                end = start + block_size
                block = sig[:, start:end]
                block_rms = np.sqrt(np.mean(block**2))
                if block_rms < 1e-6:
                    gains[i] = 1.0
                    continue
                try:
                    meter_input = block[0] if block.shape[0] == 1 else block.T
                    loudness = meter.integrated_loudness(meter_input)
                except Exception:
                    loudness = -70.0
                if not np.isfinite(loudness) or loudness < -70.0:
                    gains[i] = 1.0
                    continue
                diff_db = self.LEVEL_TARGET_LUFS - loudness
                diff_db = max(-12.0, min(diff_db, self.LEVEL_MAX_GAIN_DB))
                gains[i] = 10 ** (diff_db / 20)

            gains[-1] = gains[-2]

            smooth_kernel = max(1, smooth_size // block_size)
            if smooth_kernel > 1:
                gains = uniform_filter1d(gains, size=smooth_kernel)

            block_centers = np.array(
                [i * block_size + block_size // 2 for i in range(n_blocks + 1)]
            )
            block_centers[-1] = n - 1
            sample_indices = np.arange(n)
            gain_curve = np.interp(sample_indices, block_centers, gains)
            gain_curve = np.clip(gain_curve, 1.0 / max_gain_lin, max_gain_lin)

            output = sig * gain_curve[np.newaxis, :]
            result = torch.from_numpy(output.astype(np.float32)).to(
                device=working.device,
                dtype=working.dtype,
            )
            return result.squeeze(0) if was_mono_vector else result
        except Exception:
            return w

    def compress(self, w: torch.Tensor, sr: int, mode: ContentMode) -> torch.Tensor:
        if mode == ContentMode.MUSIC:
            return self._compress_pass(
                w,
                sr,
                self.MUSIC_COMP_THRESHOLD_DB,
                self.MUSIC_COMP_RATIO,
                self.MUSIC_COMP_MAKEUP_DB,
                self.MUSIC_COMP_ATTACK_MS,
                self.MUSIC_COMP_RELEASE_MS,
            )

        result = self._compress_pass(
            w,
            sr,
            self.SPEECH_COMP_THRESHOLD_DB,
            self.SPEECH_COMP_RATIO,
            self.SPEECH_COMP_MAKEUP_DB,
            self.SPEECH_COMP_ATTACK_MS,
            self.SPEECH_COMP_RELEASE_MS,
        )
        result = self.level_loudness(result, sr)
        return result

    def normalise_lufs(self, w: torch.Tensor) -> torch.Tensor:
        try:
            was_mono_vector = w.ndim == 1
            working = w.unsqueeze(0) if was_mono_vector else w
            if working.ndim != 2:
                raise ValueError("loudness normalization expects [channels, samples] audio")
            if working.shape[-1] < int(AudioIO.TARGET_SR * 0.5):
                return self.peak_normalise(w)
            meter = pyln.Meter(AudioIO.TARGET_SR)
            audio_np = working.detach().cpu().numpy().astype(np.float64)
            meter_input = audio_np[0] if audio_np.shape[0] == 1 else audio_np.T
            loudness = meter.integrated_loudness(meter_input)
            if not np.isfinite(loudness) or loudness < -70.0 or loudness > 0.0:
                return self.peak_normalise(w)
            if abs(loudness - self.TARGET_LUFS) <= 0.5:
                return w
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                normalised = pyln.normalize.loudness(
                    meter_input,
                    loudness,
                    self.TARGET_LUFS,
                )
            if not np.isfinite(normalised).all():
                return self.peak_normalise(w)
            if working.shape[0] > 1:
                normalised = normalised.T
            else:
                normalised = normalised[np.newaxis, :]
            result = torch.from_numpy(normalised.astype(np.float32)).to(
                device=working.device,
                dtype=working.dtype,
            )
            peak = result.abs().max().item()
            if peak > 0.99:
                result = result * (0.99 / peak)
            return result.squeeze(0) if was_mono_vector else result
        except Exception:
            return self.peak_normalise(w)

    def peak_normalise(self, w: torch.Tensor) -> torch.Tensor:
        peak = w.abs().max().item()
        if peak < 1e-8:
            return w
        return w * (10 ** (self.TRUE_PEAK_DBTP / 20) / peak)

    def true_peak_limit(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        ceiling = 10 ** (self.TRUE_PEAK_DBTP / 20)
        w_up = AudioIO.resample(w, sr, sr * 4)
        tp_up = w_up.abs().max().item()
        if tp_up > ceiling:
            w = w * (ceiling / tp_up)
        return w

    def lookahead_limit(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        """Brick-wall lookahead limiter for professional output."""
        try:
            ceiling = 10 ** (self.TRUE_PEAK_DBTP / 20)
            lookahead_samples = int(sr * self.LIMITER_LOOKAHEAD_MS / 1000)
            release_coef = np.exp(-1.0 / (sr * self.LIMITER_RELEASE_MS / 1000))

            was_mono_vector = w.ndim == 1
            working = w.unsqueeze(0) if was_mono_vector else w
            if working.ndim != 2:
                raise ValueError("limiter expects [channels, samples] audio")
            if working.shape[-1] == 0:
                return w
            sig_np = working.detach().cpu().numpy().astype(np.float64)
            n = sig_np.shape[-1]


            abs_sig = np.max(np.abs(sig_np), axis=0)



            peak_lookahead = np.empty(n, dtype=np.float64)
            candidates: deque[int] = deque()
            right = -1
            for i in range(n):
                window_end = min(i + lookahead_samples, n - 1)
                while right < window_end:
                    right += 1
                    while candidates and abs_sig[candidates[-1]] <= abs_sig[right]:
                        candidates.pop()
                    candidates.append(right)
                while candidates and candidates[0] < i:
                    candidates.popleft()
                peak_lookahead[i] = abs_sig[candidates[0]]

            gain = np.ones(n, dtype=np.float64)
            prev_g = 1.0
            for i in range(n):
                g_needed = ceiling / (peak_lookahead[i] + 1e-10)
                g_needed = min(g_needed, 1.0)
                if g_needed < prev_g:
                    prev_g = g_needed
                else:
                    prev_g = release_coef * prev_g + (1.0 - release_coef) * g_needed
                gain[i] = prev_g

            delay = min(lookahead_samples, n)
            delayed_gain = np.ones(n, dtype=np.float64)
            delayed_gain[delay:] = gain[: n - delay]
            delayed_gain[:delay] = gain[0]

            gain_tensor = torch.from_numpy(delayed_gain.astype(np.float32)).to(
                device=working.device, dtype=working.dtype
            )

            result = working * gain_tensor.unsqueeze(0)
            result = self.true_peak_limit(result, sr)
            return result.squeeze(0) if was_mono_vector else result
        except Exception:
            return self.true_peak_limit(w, sr)
