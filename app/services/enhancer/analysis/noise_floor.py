import numpy as np
import torch

from app.services.enhancer.base import ProcessingStage
from app.services.enhancer.models import ProcessingContext


class NoiseFloorEstimator(ProcessingStage):
    name = "noise_floor"

    def __init__(self, config):
        self._c = config

    async def process(self, ctx: ProcessingContext) -> ProcessingContext:
        try:
            w = ctx.audio.data
            sr = ctx.audio.sample_rate
            sig = w.squeeze(0).cpu().numpy().astype(np.float64)

            win_len = 512
            hop = 128
            n_fft = win_len
            window = np.hanning(win_len)
            n_frames = 1 + (len(sig) - win_len) // hop
            if n_frames < 5:
                return ctx

            stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
            for i in range(n_frames):
                start = i * hop
                frame = sig[start:start + win_len] * window
                stft[:, i] = np.fft.rfft(frame, n=n_fft)

            mag = np.abs(stft)
            n_bands = mag.shape[0]

            alpha = 0.7
            delta = 1.5
            D_frames = min(100, n_frames // 2)
            alpha_d = 0.85

            smoothed = np.zeros_like(mag)
            noise = np.zeros_like(mag)
            min_mag = np.zeros_like(mag)

            smoothed[:, 0] = mag[:, 0]
            noise[:, 0] = mag[:, 0]
            min_mag[:, 0] = mag[:, 0]

            for i in range(1, n_frames):
                smoothed[:, i] = alpha * smoothed[:, i - 1] + (1.0 - alpha) * mag[:, i]

                if i % D_frames == 0:
                    min_mag[:, i] = smoothed[:, i]
                else:
                    min_mag[:, i] = np.minimum(min_mag[:, i - 1], smoothed[:, i])

                spp_ratio = smoothed[:, i] / (min_mag[:, i] + 1e-10)

                spp = (spp_ratio > delta).astype(np.float64)

                update = 1.0 - spp
                noise[:, i] = alpha_d * noise[:, i - 1] + (1.0 - alpha_d) * (update * mag[:, i] + (1.0 - update) * noise[:, i - 1])

            ctx.noise_profile = {
                "noise_estimate": noise,
                "n_frames": n_frames,
                "hop": hop,
                "win_len": win_len,
                "n_fft": n_fft,
                "sr": sr,
            }
        except Exception:
            pass
        return ctx
