import numpy as np
import torch

from app.services.enhancer.base import ProcessingStage
from app.services.enhancer.models import ProcessingContext


def _cosine_fade(n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return (1.0 - np.cos(t * np.pi)) * 0.5


class SilenceStripper(ProcessingStage):
    """Strip only leading and trailing silence from audio.

    Internal pauses between speech segments are **never** removed because they
    are essential for intelligibility, natural rhythm, and meaning.
    """

    name = "silence_strip"

    def __init__(self, config):
        self._c = config

    async def process(self, ctx: ProcessingContext) -> ProcessingContext:
        try:
            w = ctx.audio.data
            sr = ctx.audio.sample_rate
            sig = w.squeeze(0).cpu().numpy().astype(np.float64)

            analysis_frame_ms = 30
            frame_size = int(sr * analysis_frame_ms / 1000)
            n_frames = len(sig) // frame_size
            if n_frames < 2:
                return ctx

            rms_values = np.zeros(n_frames)
            for i in range(n_frames):
                start = i * frame_size
                end = start + frame_size
                rms_values[i] = np.sqrt(np.mean(sig[start:end] ** 2))

            sorted_rms = np.sort(rms_values)
            high_pct = np.percentile(sorted_rms, 70)
            noise_floor = np.percentile(sorted_rms, 5)

            if high_pct < 1e-10:
                return ctx

            ratio = noise_floor / (high_pct + 1e-12)
            if ratio > 0.3:
                # Very little dynamic range — nothing to strip
                return ctx

            threshold = noise_floor * 3.0
            threshold = max(threshold, high_pct * 0.0158)

            is_speech = rms_values > threshold
            if not is_speech.any():
                return ctx

            # Find the first and last frame with speech activity
            speech_indices = np.where(is_speech)[0]
            first_speech_frame = speech_indices[0]
            last_speech_frame = speech_indices[-1]

            # Convert to sample positions with padding
            pre_pad = int(sr * self._c.pre_speech_pad_ms / 1000)
            post_pad = int(sr * self._c.post_speech_pad_ms / 1000)
            total = len(sig)

            trim_start = max(0, first_speech_frame * frame_size - pre_pad)
            trim_end = min(total, (last_speech_frame + 1) * frame_size + post_pad)

            # Only trim if we're actually removing something meaningful
            min_trim_samples = int(sr * 0.05)  # At least 50ms to bother trimming
            leading_silence = trim_start
            trailing_silence = total - trim_end

            if leading_silence < min_trim_samples and trailing_silence < min_trim_samples:
                return ctx

            trimmed = w[:, trim_start:trim_end].clone()

            # Apply short fades at the new edges
            fade_n = max(1, int(sr * self._c.strip_fade_ms / 1000))
            if trimmed.shape[1] > fade_n * 2:
                fade_in = torch.from_numpy(_cosine_fade(fade_n).astype(np.float32)).to(w.device)
                fade_out = torch.from_numpy(_cosine_fade(fade_n).astype(np.float32)).flip(0).to(w.device)
                trimmed[0, :fade_n] *= fade_in
                trimmed[0, -fade_n:] *= fade_out

            ctx.audio.data = trimmed
        except Exception:
            pass
        return ctx

