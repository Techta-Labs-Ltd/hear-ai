import logging
import shutil

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)


class SilenceProcessor:
    JOIN_FADE_MS = 20
    SEGMENT_MERGE_GAP_MS = 300
    PRE_SPEECH_PAD_MS = 200
    POST_SPEECH_PAD_MS = 150


    MIN_SPEECH_SEGMENT_MS = 1




    ANALYSIS_FRAME_MS = 10

    def detect_and_strip_silence_file(
        self,
        source_path: str,
        enhanced_path: str,
        output_path: str,
    ) -> int:
        """Apply one bounded-memory edit using source/output activity union."""
        source_rms, source_rate, source_channels, source_samples = self._scan_file_rms(source_path)
        enhanced_rms, sample_rate, channels, sample_count = self._scan_file_rms(enhanced_path)
        if (
            sample_rate != source_rate
            or channels != source_channels
            or sample_count != source_samples
            or enhanced_rms.shape != source_rms.shape
        ):
            raise RuntimeError("silence editor source/output timelines do not match")

        source_activity = self._activity_mask(source_rms, sample_count, sample_rate)
        enhanced_activity = self._activity_mask(enhanced_rms, sample_count, sample_rate)
        if source_activity is None or enhanced_activity is None:
            shutil.copyfile(enhanced_path, output_path)
            return sample_count
        activity = np.logical_or(source_activity, enhanced_activity)
        retained = self._retained_regions(activity, sample_count, sample_rate)
        if not retained:
            shutil.copyfile(enhanced_path, output_path)
            return sample_count
        return self._write_retained_regions(
            enhanced_path,
            output_path,
            retained,
            sample_rate,
            channels,
            sample_count,
        )

    def _scan_file_rms(
        self,
        path: str,
    ) -> tuple[np.ndarray, int, int, int]:
        with sf.SoundFile(path) as audio:
            sample_rate = int(audio.samplerate)
            channels = int(audio.channels)
            sample_count = int(len(audio))
            if sample_rate <= 0 or channels not in {1, 2} or sample_count < 1:
                raise ValueError("silence editor requires non-empty mono or stereo audio")
            frame_size = max(1, int(sample_rate * self.ANALYSIS_FRAME_MS / 1000))
            remainder = np.empty((0, channels), dtype=np.float32)
            rms_parts: list[np.ndarray] = []
            while True:
                block = audio.read(
                    frame_size * 4096,
                    dtype="float32",
                    always_2d=True,
                )
                if block.size == 0:
                    break
                if not np.isfinite(block).all():
                    raise ValueError("silence editor input contains non-finite samples")
                combined = np.concatenate((remainder, block), axis=0)
                complete = (combined.shape[0] // frame_size) * frame_size
                if complete:
                    frames = combined[:complete].reshape(-1, frame_size, channels)
                    channel_rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
                    rms_parts.append(np.max(channel_rms, axis=1))
                remainder = combined[complete:].copy()
            if remainder.shape[0]:
                channel_rms = np.sqrt(np.mean(remainder.astype(np.float64) ** 2, axis=0))
                rms_parts.append(np.array([np.max(channel_rms)]))
        return np.concatenate(rms_parts), sample_rate, channels, sample_count

    def _activity_mask(
        self,
        rms_values: np.ndarray,
        sample_count: int,
        sample_rate: int,
    ) -> np.ndarray | None:
        duration_s = sample_count / sample_rate
        low_pctile = 20 if duration_s < 2.0 else 30
        low_pct = float(np.percentile(rms_values, low_pctile))




        high_pct = float(np.max(rms_values))
        if high_pct < 1e-10:
            return None
        if low_pct / (high_pct + 1e-12) > 0.65:
            return None
        threshold = max(
            np.sqrt(max(low_pct, 1e-12) * high_pct),
            high_pct * 0.0158,
        )
        protective_threshold = min(
            threshold,
            low_pct * 1.10 + max(high_pct * 1e-4, 1e-7),
        )
        return rms_values > protective_threshold

    def _retained_regions(
        self,
        activity: np.ndarray,
        sample_count: int,
        sample_rate: int,
    ) -> list[dict]:
        if not activity.any():
            return []
        frame_size = max(1, int(sample_rate * self.ANALYSIS_FRAME_MS / 1000))
        speech_regions = self._frames_to_regions(activity, frame_size, sample_count)
        merge_gap = int(sample_rate * self.SEGMENT_MERGE_GAP_MS / 1000)
        merged = self._merge_close_samples(speech_regions, merge_gap)
        pre_pad = int(sample_rate * self.PRE_SPEECH_PAD_MS / 1000)
        post_pad = int(sample_rate * self.POST_SPEECH_PAD_MS / 1000)
        min_samples = int(sample_rate * self.MIN_SPEECH_SEGMENT_MS / 1000)
        padded = []
        for region in merged:
            if region["end"] - region["start"] < min_samples:
                continue
            padded.append(
                {
                    "start": max(0, region["start"] - pre_pad),
                    "end": min(sample_count, region["end"] + post_pad),
                    "speech_start": region["start"],
                    "speech_end": region["end"],
                }
            )
        return self._merge_close_samples(padded, 0)

    def _write_retained_regions(
        self,
        input_path: str,
        output_path: str,
        regions: list[dict],
        sample_rate: int,
        channels: int,
        input_samples: int,
    ) -> int:
        target_fade = max(1, int(sample_rate * self.JOIN_FADE_MS / 1000))
        fades: list[int] = []
        for previous, current in zip(regions, regions[1:], strict=False):
            fades.append(
                min(
                    target_fade,
                    previous["end"] - previous["speech_end"],
                    current["speech_start"] - current["start"],
                    previous["end"] - previous["start"],
                    current["end"] - current["start"],
                )
            )

        output_samples = 0
        with (
            sf.SoundFile(input_path) as source,
            sf.SoundFile(
                output_path,
                mode="w",
                samplerate=sample_rate,
                channels=channels,
                format="RF64",
                subtype="FLOAT",
            ) as destination,
        ):
            skipped_prefix = 0
            for index, region in enumerate(regions):
                fade = fades[index] if index < len(fades) else 0
                copy_start = region["start"] + skipped_prefix
                copy_end = region["end"] - fade
                output_samples += self._copy_file_range(
                    source,
                    destination,
                    copy_start,
                    copy_end,
                )
                if fade:
                    source.seek(region["end"] - fade)
                    previous_tail = source.read(fade, dtype="float32", always_2d=True)
                    next_region = regions[index + 1]
                    source.seek(next_region["start"])
                    next_head = source.read(fade, dtype="float32", always_2d=True)
                    if previous_tail.shape != next_head.shape or previous_tail.shape[0] != fade:
                        raise RuntimeError("silence editor could not read a complete join")
                    phase = np.linspace(0.0, np.pi / 2.0, fade, dtype=np.float32)
                    joined = (
                        previous_tail * np.cos(phase)[:, None] + next_head * np.sin(phase)[:, None]
                    )
                    destination.write(joined)
                    output_samples += fade
                skipped_prefix = fade
        if not 0 < output_samples <= input_samples:
            raise RuntimeError("silence editor returned an invalid timeline")
        return output_samples

    @staticmethod
    def _copy_file_range(source, destination, start: int, end: int) -> int:
        if end <= start:
            return 0
        source.seek(start)
        remaining = end - start
        written = 0
        while remaining:
            block = source.read(
                min(remaining, 1024 * 1024),
                dtype="float32",
                always_2d=True,
            )
            if block.shape[0] < 1:
                raise RuntimeError("silence editor encountered a truncated source")
            destination.write(block)
            remaining -= block.shape[0]
            written += block.shape[0]
        return written

    def detect_and_strip_silence(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        """Detect silence regions by adaptive energy threshold and strip them."""
        try:
            if sr <= 0:
                raise ValueError("sample rate must be positive")
            was_mono_vector = w.ndim == 1
            working = w.unsqueeze(0) if was_mono_vector else w
            if working.ndim != 2 or working.shape[0] < 1:
                raise ValueError("silence processing expects [channels, samples] audio")
            if working.shape[-1] == 0:
                return w
            if not torch.isfinite(working).all():
                raise ValueError("silence-processing input contains non-finite samples")

            signal = working.detach().cpu().numpy().astype(np.float64)
            sample_count = signal.shape[-1]

            frame_size = int(sr * self.ANALYSIS_FRAME_MS / 1000)
            frame_size = max(1, frame_size)
            n_frames = (sample_count + frame_size - 1) // frame_size
            if n_frames < 2:
                return w

            rms_values = np.zeros(n_frames)
            for i in range(n_frames):
                start = i * frame_size
                end = min(start + frame_size, sample_count)
                channel_rms = np.sqrt(np.mean(signal[:, start:end] ** 2, axis=1))

                rms_values[i] = float(np.max(channel_rms))

            is_speech = self._activity_mask(rms_values, sample_count, sr)
            if is_speech is None or not is_speech.any():
                return w

            speech_regions = self._frames_to_regions(
                is_speech,
                frame_size,
                sample_count,
            )

            if not speech_regions:
                return w

            merge_gap = int(sr * self.SEGMENT_MERGE_GAP_MS / 1000)
            merged = self._merge_close_samples(speech_regions, merge_gap)

            pre_pad = int(sr * self.PRE_SPEECH_PAD_MS / 1000)
            post_pad = int(sr * self.POST_SPEECH_PAD_MS / 1000)
            min_samp = int(sr * self.MIN_SPEECH_SEGMENT_MS / 1000)
            total = sample_count

            padded = []
            for seg in merged:
                if (seg["end"] - seg["start"]) < min_samp:
                    continue
                s = max(0, seg["start"] - pre_pad)
                e = min(total, seg["end"] + post_pad)
                padded.append(
                    {
                        "start": s,
                        "end": e,
                        "speech_start": seg["start"],
                        "speech_end": seg["end"],
                    }
                )

            if not padded:
                return w



            padded = self._merge_close_samples(padded, 0)

            target_fade = max(1, int(sr * self.JOIN_FADE_MS / 1000))
            pieces: list[torch.Tensor] = []
            previous_region = padded[0]
            pending = working[:, previous_region["start"] : previous_region["end"]].clone()
            for region in padded[1:]:
                current = working[:, region["start"] : region["end"]].clone()
                previous_post_pad = previous_region["end"] - previous_region["speech_end"]
                current_pre_pad = region["speech_start"] - region["start"]
                fade_samples = min(
                    target_fade,
                    previous_post_pad,
                    current_pre_pad,
                    pending.shape[-1],
                    current.shape[-1],
                )
                if fade_samples > 0:
                    pieces.append(pending[:, :-fade_samples])
                    phase = torch.linspace(
                        0.0,
                        torch.pi / 2.0,
                        fade_samples,
                        device=working.device,
                        dtype=working.dtype,
                    )
                    pieces.append(
                        pending[:, -fade_samples:] * torch.cos(phase)
                        + current[:, :fade_samples] * torch.sin(phase)
                    )
                    pending = current[:, fade_samples:]
                else:
                    pieces.append(pending)
                    pending = current
                previous_region = region
            pieces.append(pending)

            result = torch.cat(pieces, dim=-1)
            logger.info(
                "Silence stripped: %.1fs -> %.1fs",
                working.shape[-1] / sr,
                result.shape[-1] / sr,
            )
            return result.squeeze(0) if was_mono_vector else result
        except Exception as e:
            raise RuntimeError("global silence editing failed") from e

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

    def _merge_close_samples(self, regions: list[dict], merge_gap: int) -> list[dict]:
        if not regions:
            return []
        merged = [dict(regions[0])]
        for seg in regions[1:]:
            prev = merged[-1]
            if (seg["start"] - prev["end"]) <= merge_gap:
                prev["end"] = max(prev["end"], seg["end"])
                if "speech_start" in prev and "speech_start" in seg:
                    prev["speech_start"] = min(
                        prev["speech_start"],
                        seg["speech_start"],
                    )
                    prev["speech_end"] = max(prev["speech_end"], seg["speech_end"])
            else:
                merged.append(dict(seg))
        return merged
