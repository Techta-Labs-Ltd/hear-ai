import numpy as np
import torch

from app.services.enhancer.base import ProcessingStage
from app.services.enhancer.models import ProcessingContext, SoundEvent


class SoundEventDetector(ProcessingStage):
    name = "event_detector"

    def __init__(self, config):
        self._c = config

    async def process(self, ctx: ProcessingContext) -> ProcessingContext:
        w = ctx.audio.data
        sr = ctx.audio.sample_rate
        sig = w.squeeze(0).cpu().numpy().astype(np.float64)

        frame_ms = 10
        hop_ms = 5
        frame_len = int(sr * frame_ms / 1000)
        hop_len = int(sr * hop_ms / 1000)

        if len(sig) < frame_len * 2:
            return ctx

        n_frames = 1 + (len(sig) - frame_len) // hop_len

        win = np.hanning(frame_len)
        n_fft = 2048
        freq_bins = n_fft // 2 + 1

        flux_per_frame = np.zeros(n_frames)
        band_ratio_per_frame = np.zeros(n_frames)
        zcr_per_frame = np.zeros(n_frames)
        energy_per_frame = np.zeros(n_frames)

        prev_mag = None
        low_bin = int(500 / sr * n_fft)
        high_bin = int(4000 / sr * n_fft)
        total_bin = int(8000 / sr * n_fft)

        for i in range(n_frames):
            start = i * hop_len
            end = start + frame_len
            if end > len(sig):
                break
            frame = sig[start:end] * win
            mag = np.abs(np.fft.rfft(frame, n=n_fft))

            energy_per_frame[i] = np.sum(mag)

            if prev_mag is not None:
                flux = np.sum(np.abs(mag - prev_mag)) / (np.sum(mag) + 1e-10)
            else:
                flux = 0.0
            flux_per_frame[i] = flux

            if total_bin < len(mag):
                band_energy = np.sum(mag[low_bin:high_bin])
                total_energy = np.sum(mag[:total_bin])
                band_ratio_per_frame[i] = band_energy / (total_energy + 1e-10)
            else:
                band_ratio_per_frame[i] = 0.0

            zcr = np.sum(np.abs(np.diff(frame > 0))) / len(frame)
            zcr_per_frame[i] = zcr

            prev_mag = mag

        sensitivity = self._c.sound_event_sensitivity
        flux_thr = 0.20 + (1.0 - sensitivity) * 0.15
        band_thr = 0.35
        zcr_thr = 0.15

        energy_high_thr = np.percentile(energy_per_frame, 85)

        is_event = (
            (flux_per_frame > flux_thr)
            & (band_ratio_per_frame > band_thr)
            & (zcr_per_frame > zcr_thr)
            & (energy_per_frame > energy_high_thr)
        )

        min_event_frames = max(1, int(self._c.event_min_ms / hop_ms))
        max_event_frames = int(self._c.event_max_ms / hop_ms)
        merge_frames = int(self._c.event_merge_gap_ms / hop_ms)

        in_event = False
        event_start = 0
        event_length = 0
        gap = 0
        events = []

        for i in range(n_frames):
            if is_event[i]:
                if not in_event:
                    in_event = True
                    event_start = i
                    event_length = 1
                    gap = 0
                else:
                    event_length += 1
                    gap = 0
            else:
                if in_event:
                    gap += 1
                    if gap > merge_frames:
                        if event_length >= min_event_frames:
                            start_s = event_start * hop_len
                            end_s = min(i - gap, n_frames - 1) * hop_len + frame_len
                            events.append(SoundEvent(
                                start_sample=start_s,
                                end_sample=end_s,
                                confidence=min(1.0, event_length / min_event_frames * 0.5),
                            ))
                        in_event = False
                        event_length = 0
                        gap = 0

        if in_event and event_length >= min_event_frames:
            start_s = event_start * hop_len
            end_s = n_frames * hop_len + frame_len
            events.append(SoundEvent(
                start_sample=start_s,
                end_sample=end_s,
                confidence=min(1.0, event_length / min_event_frames * 0.5),
            ))

        events = [e for e in events if (e.end_sample - e.start_sample) <= max_event_frames * hop_len + frame_len]

        if events:
            mask = torch.ones(w.shape[1], device=w.device, dtype=torch.float32)
            fade_len = int(sr * 0.005)
            for ev in events:
                s = max(0, ev.start_sample)
                e = min(w.shape[1], ev.end_sample)
                mask[s:e] = 0.0
                if s + fade_len < w.shape[1]:
                    fade_in = torch.linspace(0, 1, min(fade_len, w.shape[1] - s), device=w.device)
                    mask[s:s + len(fade_in)] = torch.max(mask[s:s + len(fade_in)], fade_in)
                if e - fade_len >= 0:
                    fade_out = torch.linspace(1, 0, min(fade_len, e), device=w.device)
                    mask[e - len(fade_out):e] = torch.max(mask[e - len(fade_out):e], fade_out)
            ctx.event_mask = mask.unsqueeze(0)
            ctx.events = events

        return ctx
