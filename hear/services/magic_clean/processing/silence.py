import logging

import numpy as np
import torch

from .helpers import cosine_fade

logger = logging.getLogger(__name__)


class SilenceProcessor:
    STRIP_FADE_MS = 300
    SEGMENT_MERGE_GAP_MS = 300
    PRE_SPEECH_PAD_MS = 200
    POST_SPEECH_PAD_MS = 150
    MIN_SPEECH_SEGMENT_MS = 100

    ANALYSIS_FRAME_MS = 30

    def detect_and_strip_silence(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        """Detect silence regions by adaptive energy threshold and strip them."""
        try:
            sig = w.squeeze(0).cpu().numpy().astype(np.float64)

            frame_size = int(sr * self.ANALYSIS_FRAME_MS / 1000)
            n_frames = len(sig) // frame_size
            if n_frames < 2:
                return w

            duration_s = len(sig) / sr

            rms_values = np.zeros(n_frames)
            for i in range(n_frames):
                start = i * frame_size
                end = start + frame_size
                rms_values[i] = np.sqrt(np.mean(sig[start:end] ** 2))

            sorted_rms = np.sort(rms_values)
            low_pctile = 20 if duration_s < 2.0 else 30
            low_pct = np.percentile(sorted_rms, low_pctile)
            high_pct = np.percentile(sorted_rms, 70)

            if high_pct < 1e-10:
                return w

            ratio = low_pct / (high_pct + 1e-12)
            # Moderately noisy recordings can still contain removable pauses.
            # The former cutoff rejected borderline real-world speech outright.
            if ratio > 0.65:
                logger.debug("No clear speech/silence distinction, keeping all")
                return w


            threshold = np.sqrt(low_pct * high_pct)

            min_threshold = high_pct * 0.0158
            threshold = max(threshold, min_threshold)

            is_speech = rms_values > threshold


            if not is_speech.any():
                return w

            speech_regions = self._frames_to_regions(is_speech, frame_size, len(sig))

            if not speech_regions:
                return w

            merge_gap = int(sr * self.SEGMENT_MERGE_GAP_MS / 1000)
            merged = self._merge_close_samples(speech_regions, merge_gap)

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
                return w

            # Padding can make separate regions overlap. Merge again so samples
            # in those overlaps are not duplicated in the output.
            padded = self._merge_close_samples(padded, 0)

            fade_n = max(1, int(sr * self.STRIP_FADE_MS / 1000))

            segments = []
            for seg in padded:
                chunk = w[:, seg["start"]:seg["end"]].clone()
                if chunk.shape[1] > fade_n * 2:
                    fade_in = cosine_fade(fade_n, w.device)
                    fade_out = cosine_fade(fade_n, w.device).flip(0)
                    chunk[:, :fade_n] *= fade_in
                    chunk[:, -fade_n:] *= fade_out
                segments.append(chunk)

            result = torch.cat(segments, dim=1)
            logger.info(
                "Silence stripped: %.1fs -> %.1fs",
                w.shape[1] / sr, result.shape[1] / sr,
            )
            return result
        except Exception as e:
            logger.warning("Silence strip failed: %s", e)
            return w

    def _frames_to_regions(
        self, is_speech: np.ndarray, frame_size: int, total_samples: int
    ) -> list[dict]:
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

    def _merge_close_samples(
        self, regions: list[dict], merge_gap: int
    ) -> list[dict]:
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
