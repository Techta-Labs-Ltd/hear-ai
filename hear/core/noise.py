import logging

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)


class NoiseReducer:
    GATE_THRESHOLD_DB = -40.0
    GATE_ATTACK_MS = 5
    GATE_RELEASE_MS = 400
    GATE_HOLD_MS = 200

    POST_GATE_THRESHOLD_DB = -45.0

    def noise_gate(
        self, w: torch.Tensor, sr: int, threshold_db: float | None = None
    ) -> torch.Tensor:
        try:
            threshold_lin = 10 ** (
                (threshold_db if threshold_db is not None else self.GATE_THRESHOLD_DB) / 20
            )
            hold_samples = int(sr * self.GATE_HOLD_MS / 1000)

            sig_np = w.squeeze(0).cpu().numpy().astype(np.float64)
            abs_sig = np.abs(sig_np)

            attack_coef = np.exp(-1.0 / (sr * self.GATE_ATTACK_MS / 1000.0))
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

            attack_smooth = np.exp(-1.0 / (sr * self.GATE_ATTACK_MS / 1000.0))
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

    def spectral_suppress(self, w: torch.Tensor, sr: int, strength: float = 0.98) -> torch.Tensor:
        """Suppress stationary residual noise with safe weighted overlap/add.

        Estimates a noise floor from the minimum spectral energy across frames
        and subtracts it across the spectrum.  ``strength`` controls how
        aggressively the noise is removed (0 = none, 1 = full subtraction).

        Center-padding keeps the original samples away from the zero-valued
        endpoints of the Hann window. Right-padding ensures that a final partial
        frame is always processed. This makes reconstruction finite and exact in
        length for clips shorter than one window and for arbitrary remainders.
        """
        strength = float(strength)
        if strength <= 0.0:
            # This is an explicit bypass contract, not merely an approximately
            # transparent STFT round-trip.
            return w

        try:
            if sr <= 0:
                raise ValueError("sample rate must be positive")
            if not np.isfinite(strength):
                raise ValueError("suppression strength must be finite")
            if w.ndim < 1:
                raise ValueError("waveform must have a sample dimension")
            if w.shape[-1] == 0:
                return w

            strength = min(strength, 1.0)
            win_len = 1024
            hop = 256
            window = np.hanning(win_len)
            signal = w.detach().cpu().numpy().astype(np.float64, copy=True)
            original_shape = signal.shape
            channels = signal.reshape(-1, signal.shape[-1])
            output = np.empty_like(channels)

            for channel_index, channel in enumerate(channels):
                if not np.isfinite(channel).all():
                    raise ValueError("waveform contains non-finite samples")
                output[channel_index] = self._suppress_channel(
                    channel,
                    strength=strength,
                    window=window,
                    hop=hop,
                )

            result = output.reshape(original_shape)
            if not np.isfinite(result).all():
                raise RuntimeError("spectral reconstruction produced non-finite samples")
            return torch.from_numpy(result).to(device=w.device, dtype=w.dtype)
        except Exception as exc:
            logger.warning("Spectral suppress failed: %s", exc)
            return w

    def _suppress_channel(
        self,
        signal: np.ndarray,
        *,
        strength: float,
        window: np.ndarray,
        hop: int,
    ) -> np.ndarray:
        win_len = len(window)
        center_pad = win_len // 2
        padded = (
            np.pad(signal, (center_pad, center_pad), mode="reflect")
            if len(signal) > 1
            else np.pad(signal, (center_pad, center_pad), mode="edge")
        )

        n_frames = max(1, int(np.ceil((len(padded) - win_len) / hop)) + 1)
        covered_length = (n_frames - 1) * hop + win_len
        if covered_length > len(padded):
            padding = (0, covered_length - len(padded))
            padded = (
                np.pad(padded, padding, mode="reflect")
                if len(signal) > 1
                else np.pad(padded, padding, mode="edge")
            )

        stft = np.empty((win_len // 2 + 1, n_frames), dtype=np.complex128)
        for frame_index in range(n_frames):
            start = frame_index * hop
            frame = padded[start : start + win_len] * window
            stft[:, frame_index] = np.fft.rfft(frame, n=win_len)

        magnitude = np.abs(stft)
        phase = np.angle(stft)
        noise_floor = self._estimate_noise_floor(magnitude, n_frames)
        cleaned_magnitude = magnitude - noise_floor * strength
        spectral_floor = magnitude * (1.0 - strength) * 0.05
        cleaned_magnitude = np.maximum(cleaned_magnitude, spectral_floor)
        cleaned_stft = cleaned_magnitude * np.exp(1j * phase)

        reconstructed = np.zeros(covered_length, dtype=np.float64)
        window_sum = np.zeros(covered_length, dtype=np.float64)
        window_power = window**2
        for frame_index in range(n_frames):
            start = frame_index * hop
            frame = np.fft.irfft(cleaned_stft[:, frame_index], n=win_len) * window
            reconstructed[start : start + win_len] += frame
            window_sum[start : start + win_len] += window_power

        core = slice(center_pad, center_pad + len(signal))
        core_weights = window_sum[core]
        if np.any(core_weights <= np.finfo(np.float64).eps):
            raise RuntimeError("spectral reconstruction left uncovered samples")
        cleaned = reconstructed[core] / core_weights

        # Magnitude subtraction is not intended to amplify the signal. Bound
        # any numerical overlap/add overshoot without ever boosting the result.
        input_peak = float(np.max(np.abs(signal)))
        output_peak = float(np.max(np.abs(cleaned)))
        if input_peak == 0.0:
            cleaned.fill(0.0)
        elif output_peak > input_peak:
            cleaned *= input_peak / output_peak
        return cleaned

    def _estimate_noise_floor(self, mag: np.ndarray, n_frames: int) -> np.ndarray:
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
        noise_floor = uniform_filter1d(noise_floor.squeeze(), size=max(1, freq_bins // 64)).reshape(
            -1, 1
        )
        return noise_floor
