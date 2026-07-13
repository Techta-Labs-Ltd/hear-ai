import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _cosine_fade(n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return (1.0 - np.cos(t * np.pi)) * 0.5


class SilenceProcessor:
    PRE_SPEECH_PAD_MS: int = 100
    POST_SPEECH_PAD_MS: int = 100
    SEGMENT_MERGE_GAP_MS: int = 300
    MIN_SPEECH_SEGMENT_MS: int = 200

    def detect_and_strip_silence(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            sig = waveform.squeeze(0).cpu().numpy().astype(np.float64)

            analysis_frame_ms = 30
            frame_size = int(sr * analysis_frame_ms / 1000)
            n_frames = len(sig) // frame_size
            if n_frames < 2:
                return waveform

            rms_values = np.zeros(n_frames)
            for i in range(n_frames):
                start = i * frame_size
                end = start + frame_size
                rms_values[i] = np.sqrt(np.mean(sig[start:end] ** 2))

            sorted_rms = np.sort(rms_values)
            high_pct = np.percentile(sorted_rms, 70)
            noise_floor = np.percentile(sorted_rms, 5)

            if high_pct < 1e-10:
                return waveform

            ratio = noise_floor / (high_pct + 1e-12)
            if ratio > 0.3:
                return waveform

            threshold = noise_floor * 3.0
            threshold = max(threshold, high_pct * 0.0158)

            is_speech = rms_values > threshold
            if not is_speech.any():
                return waveform

            speech_regions = self._frames_to_regions(is_speech, frame_size, len(sig))
            if not speech_regions:
                return waveform

            merge_gap = int(sr * self.SEGMENT_MERGE_GAP_MS / 1000)
            merged = self._merge_regions(speech_regions, merge_gap)

            pre_pad = int(sr * self.PRE_SPEECH_PAD_MS / 1000)
            post_pad = int(sr * self.POST_SPEECH_PAD_MS / 1000)
            min_samp = int(sr * self.MIN_SPEECH_SEGMENT_MS / 1000)
            total = len(sig)

            padded = []
            for seg in merged:
                s = max(0, seg["start"] - pre_pad)
                e = min(total, seg["end"] + post_pad)
                if (e - s) >= min_samp:
                    padded.append({"start": s, "end": e})

            if not padded:
                return waveform

            fade_n = max(1, int(sr * 25 / 1000))

            segments = []
            w = waveform
            for seg in padded:
                chunk = w[:, seg["start"]:seg["end"]].clone()
                if chunk.shape[1] > fade_n * 2:
                    fade_in = torch.from_numpy(_cosine_fade(fade_n).astype(np.float32)).to(w.device)
                    fade_out = torch.from_numpy(_cosine_fade(fade_n).astype(np.float32)).flip(0).to(w.device)
                    chunk[0, :fade_n] *= fade_in
                    chunk[0, -fade_n:] *= fade_out
                segments.append(chunk)

            return torch.cat(segments, dim=1)
        except Exception as e:
            logger.warning("SilenceProcessor failed: %s", e)
            return waveform

    def _frames_to_regions(self, is_speech: np.ndarray, frame_size: int, total_samples: int) -> list[dict]:
        regions = []
        in_region = False
        start = 0
        for i in range(len(is_speech)):
            if is_speech[i] and not in_region:
                start = i * frame_size
                in_region = True
            elif not is_speech[i] and in_region:
                end = i * frame_size
                regions.append({"start": start, "end": end})
                in_region = False
        if in_region:
            regions.append({"start": start, "end": total_samples})
        return regions

    def _merge_regions(self, regions: list[dict], merge_gap: int) -> list[dict]:
        if not regions:
            return []
        merged = [dict(regions[0])]
        for seg in regions[1:]:
            prev = merged[-1]
            if (seg["start"] - prev["end"]) <= merge_gap:
                prev["end"] = max(prev["end"], seg["end"])
            else:
                merged.append(dict(seg))
        return merged
