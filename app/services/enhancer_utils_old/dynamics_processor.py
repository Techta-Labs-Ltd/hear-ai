import warnings
import numpy as np
import pyloudnorm as pyln
import torch
from .audio_io import AudioIO
from .helpers import iir_envelope_simple, iir_coefs
from .models import ContentMode


from scipy.ndimage import uniform_filter1d
from scipy.signal import lfilter
class DynamicsProcessor:
    TARGET_LUFS    = -16.0
    TRUE_PEAK_DBTP = -1.0


    SPEECH_COMP_THRESHOLD_DB = -24.0
    SPEECH_COMP_RATIO        = 4.0
    SPEECH_COMP_MAKEUP_DB    = 2.0
    SPEECH_COMP_ATTACK_MS    = 5
    SPEECH_COMP_RELEASE_MS   = 80

    MUSIC_COMP_THRESHOLD_DB  = -12.0
    MUSIC_COMP_RATIO         = 1.8
    MUSIC_COMP_MAKEUP_DB     = 0.5
    MUSIC_COMP_ATTACK_MS     = 20
    MUSIC_COMP_RELEASE_MS    = 200

    LIMITER_LOOKAHEAD_MS = 5
    LIMITER_RELEASE_MS   = 50

    LEVEL_BLOCK_MS    = 500
    LEVEL_TARGET_LUFS = -16.0
    LEVEL_MAX_GAIN_DB = 15.0
    LEVEL_SMOOTH_MS   = 100

    def __init__(self, device: torch.device):
        self._device = device

    def _compress_pass(
        self, w: torch.Tensor, sr: int,
        threshold_db: float, ratio: float, makeup_db: float,
        attack_ms: float, release_ms: float,
    ) -> torch.Tensor:
        try:
            threshold_lin = 10 ** (threshold_db / 20)
            attack_coef   = np.exp(-1.0 / (sr * attack_ms / 1000))
            release_coef  = np.exp(-1.0 / (sr * release_ms / 1000))
            makeup_lin    = 10 ** (makeup_db / 20)

            sig_np = w.squeeze(0).cpu().numpy().astype(np.float64)
            env_np = iir_envelope_simple(np.abs(sig_np), attack_coef, release_coef)

            gain_np = np.ones_like(env_np)
            over = env_np > threshold_lin
            gain_np[over] = (
                (threshold_lin + (env_np[over] - threshold_lin) / ratio)
                / (env_np[over] + 1e-12)
            )

            b_r, a_r = iir_coefs(release_ms, sr)
            gain_np = lfilter(b_r, a_r, gain_np)
            gain_np = np.clip(gain_np, 0.0, 1.0)

            gain = torch.from_numpy(gain_np.astype(np.float32)).unsqueeze(0).to(w.device)
            out  = w * gain * makeup_lin
            peak = out.abs().max().item()
            if peak > 0.99:
                out = out * (0.99 / peak)
            return out
        except Exception:
            return w

    def level_loudness(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        """Short-term loudness leveling for consistent volume.

        Measures loudness in overlapping blocks and applies smooth gain
        corrections so every section hits the target LUFS.
        """
        try:
            sig = w.squeeze(0).cpu().numpy().astype(np.float64)
            n = len(sig)
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
                block = sig[start:end]
                block_rms = np.sqrt(np.mean(block ** 2))
                if block_rms < 1e-6:
                    gains[i] = 1.0
                    continue
                try:
                    loudness = meter.integrated_loudness(block)
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


            block_centers = np.array([i * block_size + block_size // 2 for i in range(n_blocks + 1)])
            block_centers[-1] = n - 1
            sample_indices = np.arange(n)
            gain_curve = np.interp(sample_indices, block_centers, gains)
            gain_curve = np.clip(gain_curve, 1.0 / max_gain_lin, max_gain_lin)


            output = sig * gain_curve
            output = torch.from_numpy(output.astype(np.float32)).unsqueeze(0).to(w.device)
            return output
        except Exception:
            return w

    def compress(self, w: torch.Tensor, sr: int, mode: ContentMode) -> torch.Tensor:
        if mode == ContentMode.MUSIC:
            return self._compress_pass(
                w, sr,
                self.MUSIC_COMP_THRESHOLD_DB,
                self.MUSIC_COMP_RATIO,
                self.MUSIC_COMP_MAKEUP_DB,
                self.MUSIC_COMP_ATTACK_MS,
                self.MUSIC_COMP_RELEASE_MS,
            )


        result = self._compress_pass(
            w, sr,
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
            if w.shape[1] < int(AudioIO.TARGET_SR * 0.5):
                return self.peak_normalise(w)
            meter    = pyln.Meter(AudioIO.TARGET_SR)
            audio_np = w.cpu().squeeze(0).numpy().astype(np.float64)
            loudness = meter.integrated_loudness(audio_np)
            if not np.isfinite(loudness) or loudness < -70.0 or loudness > 0.0:
                return self.peak_normalise(w)
            if abs(loudness - self.TARGET_LUFS) <= 0.5:
                return w
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                normalised = pyln.normalize.loudness(audio_np, loudness, self.TARGET_LUFS)
            if not np.isfinite(normalised).all():
                return self.peak_normalise(w)
            result = torch.from_numpy(normalised.astype(np.float32)).unsqueeze(0).to(self._device)
            peak   = result.abs().max().item()
            if peak > 0.99:
                result = result * (0.99 / peak)
            return result
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

            sig_np = w.squeeze(0).cpu().numpy().astype(np.float64)
            n = len(sig_np)

            abs_sig = np.abs(sig_np)

            peak_lookahead = np.zeros(n, dtype=np.float64)
            for i in range(n):
                window_end = min(i + lookahead_samples + 1, n)
                peak_lookahead[i] = np.max(abs_sig[i:window_end])

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

            delayed_gain = np.ones(n, dtype=np.float64)
            delayed_gain[lookahead_samples:] = gain[:n - lookahead_samples]
            delayed_gain[:lookahead_samples] = gain[0]

            gain_tensor = torch.from_numpy(
                delayed_gain.astype(np.float32)
            ).unsqueeze(0).to(w.device)

            return w * gain_tensor
        except Exception:
            return self.true_peak_limit(w, sr)
