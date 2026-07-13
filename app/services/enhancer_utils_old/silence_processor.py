import logging
import warnings

import numpy as np
import torch
import torchaudio.transforms as T
import torch.hub

from .helpers import cosine_fade

logger = logging.getLogger(__name__)


class SilenceProcessor:
    STRIP_FADE_MS = 300
    SEGMENT_MERGE_GAP_MS = 300
    PRE_SPEECH_PAD_MS = 200
    POST_SPEECH_PAD_MS = 150
    MIN_SPEECH_SEGMENT_MS = 100

    ANALYSIS_FRAME_MS = 30

    def __init__(self):
        self._vad_model = None
        self._vad_utils = None
        self._resamplers = {}

    def load(self):
        self._load_vad()

    def _load_vad(self):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._vad_model, self._vad_utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    trust_repo=True,
                    verbose=False
                )
        except Exception as e:
            logger.warning("Failed to load Silero VAD: %s", e)

    def apply_transcript_gate(
        self, w: torch.Tensor, word_timestamps: list[dict], sr: int
    ) -> torch.Tensor:
        try:
            if not word_timestamps:
                return w

            total_samples = w.shape[1]
            pre_pad = int(sr * 150 / 1000)
            post_pad = int(sr * 150 / 1000)

            padded_timestamps = []
            for word in word_timestamps:
                start_sec = word.get("start", 0.0)
                end_sec = word.get("end", 0.0)

                s = max(0, int(start_sec * sr) - pre_pad)
                e = min(total_samples, int(end_sec * sr) + post_pad)
                padded_timestamps.append({"start": s, "end": e})

            padded_timestamps.sort(key=lambda x: x["start"])

            merged = []
            for seg in padded_timestamps:
                if not merged:
                    merged.append(seg)
                else:
                    last = merged[-1]
                    if seg["start"] <= last["end"]:
                        last["end"] = max(last["end"], seg["end"])
                    else:
                        merged.append(seg)

            min_samp = int(sr * self.MIN_SPEECH_SEGMENT_MS / 1000)
            valid_segments = []
            for seg in merged:
                if (seg["end"] - seg["start"]) >= min_samp:
                    valid_segments.append(seg)

            if not valid_segments:
                return w

            fade_samples = max(1, int(sr * 150 / 1000))
            mask = torch.zeros(total_samples, device=w.device)

            for seg in valid_segments:
                s_idx = seg["start"]
                e_idx = seg["end"]
                mask[s_idx:e_idx] = 1.0

            for seg in valid_segments:
                s_idx = seg["start"]
                e_idx = seg["end"]
                
                if fade_samples > 0:
                    fade_in = torch.linspace(0, 1, fade_samples, device=w.device)
                    fade_out = torch.linspace(1, 0, fade_samples, device=w.device)

                    if s_idx + fade_samples <= total_samples:
                        mask[s_idx:s_idx+fade_samples] = torch.max(mask[s_idx:s_idx+fade_samples], fade_in)
                    if e_idx - fade_samples >= 0:
                        mask[e_idx-fade_samples:e_idx] = torch.max(mask[e_idx-fade_samples:e_idx], fade_out)

            mask = mask.clamp(0.0, 1.0)
            FLOOR_GAIN = 10 ** (-45.0 / 20)
            mask = torch.where(mask > FLOOR_GAIN, mask, torch.tensor(FLOOR_GAIN, device=w.device))
            
            result = w * mask.unsqueeze(0)
            logger.info("Applied transcript gate smoothly.")
            return result
        except Exception as e:
            logger.warning("apply_transcript_gate failed: %s", e)
            return w

    def apply_vad_gate(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        try:
            if not self._vad_model or not self._vad_utils:
                return w

            get_speech_timestamps = self._vad_utils[0]

            w_16k = w
            if sr != 16000:
                if sr not in self._resamplers:
                    self._resamplers[sr] = T.Resample(sr, 16000).to(w.device)
                w_16k = self._resamplers[sr](w)

            wav_16k_mono = w_16k.mean(dim=0, keepdim=True).squeeze(0).cpu()

            speech_timestamps = get_speech_timestamps(
                wav_16k_mono,
                self._vad_model,
                sampling_rate=16000,
                threshold=0.35,
                min_speech_duration_ms=100,
                min_silence_duration_ms=250
            )

            if not speech_timestamps:
                return w

            mask = torch.zeros(w.shape[1], device=w.device)
            ratio = sr / 16000.0
            fade_samples = int(sr * 0.3)

            for ts in speech_timestamps:
                start_idx = int(ts['start'] * ratio)
                end_idx = int(ts['end'] * ratio)

                start_idx = max(0, start_idx - fade_samples)
                end_idx = min(w.shape[1], end_idx + fade_samples)

                mask[start_idx:end_idx] = 1.0

            for ts in speech_timestamps:
                s_idx = max(0, int(ts['start'] * ratio) - fade_samples)
                e_idx = min(w.shape[1], int(ts['end'] * ratio) + fade_samples)

                if fade_samples > 0:
                    fade_in = torch.linspace(0, 1, fade_samples, device=w.device)
                    fade_out = torch.linspace(1, 0, fade_samples, device=w.device)

                    if s_idx + fade_samples <= w.shape[1]:
                        mask[s_idx:s_idx+fade_samples] = torch.max(mask[s_idx:s_idx+fade_samples], fade_in)
                    if e_idx - fade_samples >= 0:
                        mask[e_idx-fade_samples:e_idx] = torch.max(mask[e_idx-fade_samples:e_idx], fade_out)

            mask = mask.clamp(0.0, 1.0)
            result = w * mask.unsqueeze(0)
            logger.info("Applied Silero VAD gate")
            return result
        except Exception as e:
            logger.warning("apply_vad_gate failed: %s", e)
            return w

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
            if ratio > 0.5:
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

            fade_n = max(1, int(sr * self.STRIP_FADE_MS / 1000))

            segments = []
            for seg in padded:
                chunk = w[:, seg["start"]:seg["end"]].clone()
                if chunk.shape[1] > fade_n * 2:
                    fade_in = cosine_fade(fade_n, w.device)
                    fade_out = cosine_fade(fade_n, w.device).flip(0)
                    chunk[0, :fade_n] *= fade_in
                    chunk[0, -fade_n:] *= fade_out
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

    def _merge_close_segments(
        self, timestamps: list[dict], sr: int
    ) -> list[dict]:
        if not timestamps:
            return []
        merge_gap = int(sr * self.SEGMENT_MERGE_GAP_MS / 1000)
        merged = [dict(timestamps[0])]
        for seg in timestamps[1:]:
            prev = merged[-1]
            if (seg["start"] - prev["end"]) <= merge_gap:
                prev["end"] = max(prev["end"], seg["end"])
            else:
                merged.append(dict(seg))
        return merged
