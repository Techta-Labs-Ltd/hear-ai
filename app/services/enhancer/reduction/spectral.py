import numpy as np
import torch

from app.services.enhancer.base import ProcessingStage
from app.services.enhancer.models import ProcessingContext


class SpectralPreFilter(ProcessingStage):
    name = "spectral_prefilter"

    def __init__(self, config):
        self._c = config

    async def process(self, ctx: ProcessingContext) -> ProcessingContext:
        try:
            w = ctx.audio.data
            sr = ctx.audio.sample_rate
            sig = w.squeeze(0).cpu().numpy().astype(np.float64)

            n_fft = 2048
            hop = n_fft // 2
            window = np.hanning(n_fft)

            frames = 1 + (len(sig) - n_fft) // hop
            if frames < 8:
                return ctx

            mag = np.zeros((n_fft // 2 + 1, frames), dtype=np.float64)
            for i in range(frames):
                start = i * hop
                frame = sig[start:start + n_fft] * window
                mag[:, i] = np.abs(np.fft.rfft(frame, n=n_fft))

            if ctx.noise_profile is not None:
                np_noise = ctx.noise_profile["noise_estimate"]
                if np_noise.shape[1] >= frames:
                    noise_spec = np.mean(np_noise[:, :frames], axis=1)
                else:
                    repeats = frames // np_noise.shape[1] + 1
                    tiled = np.tile(np_noise, (1, repeats))
                    noise_spec = np.mean(tiled[:, :frames], axis=1)
            else:
                frame_energies = np.sum(mag, axis=0)
                noise_count = max(1, frames // 5)
                quietest = np.argsort(frame_energies)[:noise_count]
                noise_spec = np.mean(mag[:, quietest], axis=1)

            noise_spec = np.maximum(noise_spec, 1e-10)
            avg_spec = np.mean(mag, axis=1)
            noise_ratio = np.mean(noise_spec) / (np.mean(avg_spec) + 1e-10)
            if noise_ratio < 0.02 or noise_ratio > 0.7:
                return ctx

            strength = getattr(self._c, "spectral_strength", 0.20)
            alpha = 0.92

            output = np.zeros(len(sig))
            norm = np.zeros(len(sig))

            prev_gain = np.ones(n_fft // 2 + 1, dtype=np.float64)

            for i in range(frames):
                start = i * hop
                end = start + n_fft
                frame = sig[start:end]
                if len(frame) < n_fft:
                    frame = np.pad(frame, (0, n_fft - len(frame)))
                frame_win = frame * window
                spec = np.fft.rfft(frame_win, n=n_fft)
                mag_i = np.abs(spec)
                phase = np.angle(spec)

                snr_post = (mag_i / noise_spec) ** 2 - 1.0
                snr_post = np.maximum(snr_post, 0.0)
                snr_priori = alpha * (prev_gain ** 2) * (mag_i / noise_spec) ** 2
                snr_priori += (1.0 - alpha) * snr_post

                wiener_gain = snr_priori / (1.0 + snr_priori)
                wiener_gain = np.maximum(wiener_gain, self._c.spectral_floor)

                oversub = 1.0 + strength * 0.5 * (1.0 - np.linspace(0, 1, len(wiener_gain)))
                wiener_gain = wiener_gain ** oversub

                smoothed = np.copy(wiener_gain)
                smoothed[1:-1] = 0.25 * wiener_gain[:-2] + 0.5 * wiener_gain[1:-1] + 0.25 * wiener_gain[2:]
                smoothed = np.clip(smoothed, self._c.spectral_floor, 1.0)

                prev_gain = smoothed.copy()

                cleaned = mag_i * smoothed * np.exp(1j * phase)
                recon = np.fft.irfft(cleaned, n=n_fft) * window
                out_end = min(end, len(sig))
                length = out_end - start
                output[start:out_end] += recon[:length]
                norm[start:out_end] += window[:length]

            valid = norm > 1e-8
            output[valid] /= norm[valid]

            ctx.audio.data = torch.from_numpy(output.astype(np.float32)).unsqueeze(0).to(w.device)
        except Exception:
            pass
        return ctx
