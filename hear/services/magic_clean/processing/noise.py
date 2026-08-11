import logging

import numpy as np
import torch

from scipy.ndimage import uniform_filter1d
logger = logging.getLogger(__name__)


class NoiseReducer:
    GATE_THRESHOLD_DB = -40.0
    GATE_ATTACK_MS    = 5
    GATE_RELEASE_MS   = 400
    GATE_HOLD_MS      = 200

    POST_GATE_THRESHOLD_DB = -45.0

    def noise_gate(self, w: torch.Tensor, sr: int, threshold_db: float | None = None) -> torch.Tensor:
        try:
            threshold_lin = 10 ** ((threshold_db if threshold_db is not None else self.GATE_THRESHOLD_DB) / 20)
            hold_samples  = int(sr * self.GATE_HOLD_MS / 1000)

            sig_np  = w.squeeze(0).cpu().numpy().astype(np.float64)
            abs_sig = np.abs(sig_np)

            attack_coef  = np.exp(-1.0 / (sr * self.GATE_ATTACK_MS / 1000.0))
            release_coef = np.exp(-1.0 / (sr * self.GATE_RELEASE_MS / 1000.0))

            env = np.zeros_like(abs_sig)
            prev = 0.0
            for i in range(len(abs_sig)):
                c = attack_coef if abs_sig[i] > prev else release_coef
                prev = c * prev + (1.0 - c) * abs_sig[i]
                env[i] = prev

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

            target = held.astype(np.float64)

            attack_smooth  = np.exp(-1.0 / (sr * self.GATE_ATTACK_MS / 1000.0))
            release_smooth = np.exp(-1.0 / (sr * self.GATE_RELEASE_MS / 1000.0))

            gain_np = np.ones_like(target)
            prev_g = target[0]
            gain_np[0] = prev_g
            for i in range(1, len(target)):
                c = attack_smooth if target[i] > prev_g else release_smooth
                prev_g = c * prev_g + (1.0 - c) * target[i]
                gain_np[i] = prev_g

            gain = torch.from_numpy(gain_np.astype(np.float32)).unsqueeze(0).to(w.device)
            return w * gain
        except Exception:
            return w

    def spectral_suppress(
        self, w: torch.Tensor, sr: int, strength: float = 0.98
    ) -> torch.Tensor:
        """Spectral noise suppression for residual noise after SE.

        Estimates a noise floor from the minimum spectral energy across frames
        and subtracts it across the spectrum.  ``strength`` controls how
        aggressively the noise is removed (0 = none, 1 = full subtraction).
        Short recordings (< 3 s) automatically use a higher strength.
        """
        try:
            sig = w.squeeze(0).cpu().numpy().astype(np.float64)

            duration_s = len(sig) / sr
            if duration_s < 3.0:
                strength = max(strength, 0.95)


            win_len = 1024
            hop = 256
            n_fft = win_len

            window = np.hanning(win_len)

            n_frames = 1 + (len(sig) - win_len) // hop
            stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
            for i in range(n_frames):
                start = i * hop
                frame = sig[start:start + win_len] * window
                stft[:, i] = np.fft.rfft(frame, n=n_fft)

            mag = np.abs(stft)
            phase = np.angle(stft)

            noise_floor = self._estimate_noise_floor(mag, n_frames)

            cleaned_mag = mag - noise_floor * strength
            cleaned_mag = np.maximum(cleaned_mag, mag * (1.0 - strength) * 0.05)

            cleaned_stft = cleaned_mag * np.exp(1j * phase)


            output = np.zeros(len(sig))
            window_sum = np.zeros(len(sig))
            for i in range(n_frames):
                start = i * hop
                frame = np.fft.irfft(cleaned_stft[:, i], n=n_fft) * window
                end = min(start + win_len, len(sig))
                length = end - start
                output[start:end] += frame[:length]
                window_sum[start:end] += window[:length] ** 2

            valid = window_sum > 1e-8
            output[valid] /= window_sum[valid]

            return torch.from_numpy(output.astype(np.float32)).unsqueeze(0).to(w.device)
        except Exception as e:
            logger.warning("Spectral suppress failed: %s", e)
            return w

    def _estimate_noise_floor(
        self, mag: np.ndarray, n_frames: int
    ) -> np.ndarray:
        """Robust noise floor estimation.

        Uses the minimum spectral magnitude across all frames as the primary
        estimate, which is more reliable than the quietest-N-percentile
        approach when the recording has no truly silent segments (e.g.
        constant market or traffic noise).
        """
        freq_bins = mag.shape[0]


        min_mag = mag.min(axis=1, keepdims=True)


        frame_energy = mag.sum(axis=0)
        n_noise = max(1, n_frames // 10)
        quiet_idx = np.argsort(frame_energy)[:n_noise]
        avg_quiet = mag[:, quiet_idx].mean(axis=1, keepdims=True)


        noise_floor = np.minimum(min_mag, avg_quiet)


        noise_floor = uniform_filter1d(
            noise_floor.squeeze(), size=max(1, freq_bins // 64)
        ).reshape(-1, 1)

        return noise_floor
