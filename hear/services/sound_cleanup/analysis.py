"""Bounded per-channel analysis; original and denoised speech both protect repairs."""

import csv
import gc
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import ndimage, signal

from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode
from hear.services.sound_cleanup.assets import SoundCleanupAssets


@dataclass(frozen=True)
class SoundAnalysis:
    frames: int
    channels: int
    speech: np.ndarray
    source_rms: np.ndarray
    clean_rms: np.ndarray
    events: dict[str, np.ndarray]
    protected: np.ndarray
    model_digest: str
    step_frames: int = 1536


class SoundAnalyser:
    RATE = 48000
    STEP = 1536
    GROUP_LABELS = {
        "handling": ("Rustle", "Crumpling, crinkling", "Shuffling cards", "Shuffle", "Scrape"),
        "impact": ("Slam", "Door", "Knock", "Tap", "Thump, thud", "Bang"),
        "animal": ("Bark", "Yip", "Howl", "Growling", "Bow-wow"),
        "cough": ("Cough", "Throat clearing", "Sneeze"),
        "click": ("Clicking",),
        "protected_content": ("Music", "Singing", "Laughter", "Breathing", "Whispering"),
    }

    def __init__(self, assets: SoundCleanupAssets, *, device: str = "cpu"):
        if device not in ("cpu", "cuda:0"):
            raise ValueError("invalid_sound_analysis_device")
        self.assets = assets
        self.device = device

    def analyse(
        self, source: Path, baseline: Path, guard: ResourceGuard, *, detect_events: bool = True
    ) -> SoundAnalysis:
        guard.check()
        self.assets.verify()
        with sf.SoundFile(source) as a, sf.SoundFile(baseline) as b:
            if (
                (a.frames, a.channels, a.samplerate) != (b.frames, b.channels, b.samplerate)
                or a.channels not in (1, 2)
                or a.samplerate != self.RATE
            ):
                raise CleanExecutionError(
                    ErrorCode.INVALID_AUDIO, "sound_cleanup_analysis_layout_mismatch"
                )
            frames, channels = a.frames, a.channels
        source_prob, source_rms = self._vad(source, guard)
        clean_prob, clean_rms = self._vad(baseline, guard)
        probability = np.maximum(source_prob, clean_prob)
        speech = np.max(probability, axis=1)
        # Any channel protects the same interval on every channel, including
        # split-speaker and anti-phase stereo. These are conservative risk flags,
        # not a declaration that low-probability audio contains no wanted words.
        protected = (
            ndimage.maximum_filter1d((speech >= 0.12).astype(np.uint8), size=17, mode="constant")
            > 0
        )
        events: dict[str, np.ndarray] = {}
        if detect_events:
            events = self._events(source, baseline, len(speech), guard)
            protected |= events["protected_content"] >= 0.30
        return SoundAnalysis(
            frames,
            channels,
            speech,
            source_rms,
            clean_rms,
            events,
            protected,
            self.assets.manifest_sha256,
        )

    def speech_probability(self, path: Path, guard: ResourceGuard) -> np.ndarray:
        """Per-step Silero speech probability (max over channels) on the 48 kHz grid."""
        probability, _ = self._vad(path, guard)
        return probability.max(axis=1)

    # Silero carries recurrent state between windows, so each lane runs its windows in
    # order; LANES lanes with independent state share one batched call, which cuts the
    # per-window overhead about four-fold. The file is read in slabs that bound memory.
    SLAB_FRAMES = 1536 * 2000
    LANES = 8

    def _vad(self, path: Path, guard: ResourceGuard) -> tuple[np.ndarray, np.ndarray]:
        model = torch.jit.load(str(self.assets.path("silero_vad.jit")), map_location="cpu").eval()
        with sf.SoundFile(path) as audio:
            count = math.ceil(audio.frames / self.STEP)
            probability = np.zeros((count, audio.channels), dtype=np.float32)
            rms = np.zeros_like(probability)
            for channel in range(audio.channels):
                for slab_start in range(0, audio.frames, self.SLAB_FRAMES):
                    guard.check()
                    slab_end = min(audio.frames, slab_start + self.SLAB_FRAMES)
                    left = max(0, slab_start - 96)
                    right = min(audio.frames, slab_end + 96)
                    audio.seek(left)
                    block = audio.read(right - left, dtype="float32", always_2d=True)[:, channel]
                    if not np.isfinite(block).all():
                        raise CleanExecutionError(
                            ErrorCode.INVALID_AUDIO, "invalid_sound_analysis_samples"
                        )
                    first = slab_start // self.STEP
                    last = math.ceil(slab_end / self.STEP)
                    windows = last - first
                    core = block[slab_start - left : slab_end - left]
                    padded = np.zeros(windows * self.STEP, dtype=np.float64)
                    padded[: len(core)] = core
                    rms[first:last, channel] = np.sqrt(
                        np.mean(padded.reshape(windows, self.STEP) ** 2, axis=1)
                    )
                    resampled = signal.resample_poly(block, 1, 3).astype(np.float32)
                    resampled = np.pad(resampled, (0, 512))
                    offsets = (np.arange(first, last) * self.STEP - left) // 3
                    matrix = np.stack([resampled[o : o + 512] for o in offsets]) if windows else np.zeros((0, 512), np.float32)
                    lanes = max(1, min(self.LANES, windows))
                    per_lane = math.ceil(windows / lanes)
                    model.reset_states()
                    for step in range(per_lane):
                        rows = step + per_lane * np.arange(lanes)
                        valid = rows < windows
                        batch = torch.from_numpy(matrix[np.minimum(rows, windows - 1)])
                        with torch.inference_mode():
                            values = model(batch, 16000).reshape(-1).numpy()
                        if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
                            raise CleanExecutionError(
                                ErrorCode.INVALID_AUDIO, "invalid_speech_probability"
                            )
                        probability[first + rows[valid], channel] = values[valid]
        return probability, rms

    def _events(
        self, source: Path, baseline: Path, count: int, guard: ResourceGuard
    ) -> dict[str, np.ndarray]:
        with self.assets.path("labels.csv").open() as stream:
            labels = {row["display_name"]: int(row["index"]) for row in csv.DictReader(stream)}
        groups = {
            name: [labels[label] for label in choices if label in labels]
            for name, choices in self.GROUP_LABELS.items()
        }
        if any(not values for values in groups.values()):
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "sound_event_labels_missing")
        scores = {name: np.zeros(count, dtype=np.float32) for name in groups}
        model = torch.jit.load(
            str(self.assets.path("panns_sed.jit")), map_location=self.device
        ).eval()
        try:
            for path in (source, baseline):
                with sf.SoundFile(path) as audio:
                    for start in range(0, audio.frames, 384000):
                        guard.check()
                        left = max(0, start - 48000)
                        audio.seek(left)
                        data = audio.read(480000, dtype="float32", always_2d=True)
                        for channel in range(audio.channels):
                            samples = signal.resample_poly(data[:, channel], 2, 3)
                            samples = np.pad(samples, (0, 320000 - len(samples)))
                            with torch.inference_mode():
                                result = (
                                    model(torch.from_numpy(samples).unsqueeze(0).to(self.device))
                                    .detach()
                                    .cpu()
                                    .numpy()[0]
                                )
                            if (
                                result.ndim != 2
                                or result.shape[1] != 527
                                or not np.isfinite(result).all()
                                or np.min(result) < 0
                                or np.max(result) > 1
                            ):
                                raise CleanExecutionError(
                                    ErrorCode.INVALID_AUDIO, "invalid_sound_event_prediction"
                                )
                            lo = math.ceil(start / self.STEP)
                            hi = min(
                                count, math.ceil(min(start + 384000, audio.frames) / self.STEP)
                            )
                            times = (np.arange(lo, hi) * self.STEP - left) / 480
                            index = np.clip(times.astype(int), 0, len(result) - 1)
                            for name, indices in groups.items():
                                values = np.max(result[index[:, None], indices], axis=1)
                                scores[name][lo:hi] = np.maximum(scores[name][lo:hi], values)
        finally:
            del model
            gc.collect()
            if self.device == "cuda:0" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        guard.check()
        return scores
