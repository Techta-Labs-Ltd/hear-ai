from __future__ import annotations

import json
import math
import os
import re
import selectors
import subprocess
import tempfile
import time
from dataclasses import dataclass

import numpy as np
import torch

from .audio_io import AudioIO
from .quality import QualityMetrics

TRUE_PEAK_CEILING_DBTP = -1.0
MIN_MP3_DURATION_TOLERANCE_SECONDS = 0.03
MPEG_1_LAYER_III_FRAME_SAMPLES = 1152
MPEG_2_LAYER_III_FRAME_SAMPLES = 576
SILENCE_PEAK_THRESHOLD = 1e-06
SILENCE_RMS_THRESHOLD = 1e-07
CHANNEL_COLLAPSE_RELATIVE_THRESHOLD = 0.0001
FFMPEG_TIMEOUT_SECONDS = 4 * 60 * 60
PCM_READ_BYTES = 1024 * 1024


class AudioValidationError(RuntimeError):
    """The encoded Magic Clean artifact failed an audio-integrity gate."""


@dataclass(frozen=True, slots=True)
class DeliveredAudioMetrics:
    duration_seconds: float
    duration_delta_seconds: float
    sample_rate: int
    channels: int
    sample_count: int
    peak_db: float
    lufs: float
    snr_db: float
    clipping_detected: bool
    clipped_sample_count: int
    quality_score: float


@dataclass(frozen=True, slots=True)
class _DecodedAudioScan:
    sample_rate: int
    channels: int
    sample_count: int
    absolute_peak: float
    rms: float
    channel_absolute_peaks: tuple[float, ...]
    channel_rms: tuple[float, ...]
    clipped_sample_count: int
    frame_powers: np.ndarray
    test_waveform: torch.Tensor | None = None

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate

    @property
    def is_silent(self) -> bool:
        return self.absolute_peak < SILENCE_PEAK_THRESHOLD and self.rms < SILENCE_RMS_THRESHOLD


class AudioValidator:
    @staticmethod
    def validate_pcm(
        waveform: torch.Tensor, *, expected_samples: int | None, preserve_timeline: bool
    ) -> None:
        """Validate an exact pre-encode processing tensor."""
        AudioValidator._validate_pcm_shape_and_values(waveform, label="Magic Clean PCM")
        if preserve_timeline:
            if expected_samples is None or expected_samples < 1:
                raise AudioValidationError(
                    "Expected PCM sample count is required when preserving the timeline"
                )
            if waveform.shape[1] != expected_samples:
                raise AudioValidationError(
                    f"Magic Clean changed the PCM timeline without silence cutting: expected={expected_samples}, actual={waveform.shape[1]}"
                )
        peak = waveform.abs().max().item()
        if peak > 4.0:
            raise AudioValidationError(
                f"Magic Clean produced an unsafe intermediate peak: {peak:.3f}"
            )

    @staticmethod
    def mp3_duration_tolerance_seconds(sample_rate: int) -> float:
        """Return one Layer III frame or 30 ms, whichever is larger."""
        if sample_rate <= 0:
            raise ValueError("delivered sample rate must be positive")
        frame_samples = (
            MPEG_1_LAYER_III_FRAME_SAMPLES
            if sample_rate >= 32000
            else MPEG_2_LAYER_III_FRAME_SAMPLES
        )
        return max(MIN_MP3_DURATION_TOLERANCE_SECONDS, frame_samples / sample_rate)

    @staticmethod
    def validate_delivered_audio(
        source_path: str,
        output_path: str,
        *,
        cut_silence: bool,
        expect_audible: bool,
        metrics: QualityMetrics,
        retained_reference_path: str | None = None,
    ) -> DeliveredAudioMetrics:
        """Stream-decode and validate the exact local MP3 before upload.

        Real files are scanned in bounded PCM blocks and retain only compact 20 ms
        power summaries, rather than whole decoded waveforms. The tensor fallback
        exists only for focused unit tests that inject an in-memory decoder.
        """
        source = AudioValidator._scan_decoded_audio(source_path, label="source")
        delivered = AudioValidator._scan_decoded_audio(output_path, label="delivered MP3")
        retained_reference = (
            AudioValidator._scan_decoded_audio(retained_reference_path, label="retained mix")
            if retained_reference_path is not None
            else None
        )
        if delivered.channels != source.channels:
            raise AudioValidationError(
                f"Magic Clean changed the delivered channel count: source={source.channels}, output={delivered.channels}"
            )
        if retained_reference is not None:
            if retained_reference.channels != source.channels:
                raise AudioValidationError(
                    f"Magic Clean changed the retained-mix channel count: source={source.channels}, retained={retained_reference.channels}"
                )
            retained_duration_delta = abs(
                delivered.duration_seconds - retained_reference.duration_seconds
            )
            retained_tolerance = AudioValidator.mp3_duration_tolerance_seconds(
                delivered.sample_rate
            )
            if retained_duration_delta > retained_tolerance + 1e-09:
                raise AudioValidationError(
                    f"Magic Clean delivered duration differs from the retained mix: retained={retained_reference.duration_seconds:.6f}s, output={delivered.duration_seconds:.6f}s, allowance={retained_tolerance:.6f}s"
                )
        duration_delta = abs(delivered.duration_seconds - source.duration_seconds)
        duration_tolerance = AudioValidator.mp3_duration_tolerance_seconds(delivered.sample_rate)
        if not cut_silence:
            if duration_delta > duration_tolerance + 1e-09:
                raise AudioValidationError(
                    f"Magic Clean encoded duration mismatch: source={source.duration_seconds:.6f}s, output={delivered.duration_seconds:.6f}s, allowance={duration_tolerance:.6f}s"
                )
        else:
            if (
                retained_reference is not None
                and retained_reference.duration_seconds > source.duration_seconds + 1e-09
            ):
                raise AudioValidationError(
                    "Magic Clean silence editing unexpectedly lengthened the retained mix"
                )
            if delivered.duration_seconds > source.duration_seconds + duration_tolerance + 1e-09:
                raise AudioValidationError(
                    "Magic Clean silence editing unexpectedly lengthened the recording"
                )
            if expect_audible:
                minimum_retained_seconds = AudioValidator._minimum_retained_activity_seconds(source)
                if delivered.duration_seconds + duration_tolerance < minimum_retained_seconds:
                    raise AudioValidationError(
                        f"Magic Clean silence editing removed protected source activity: minimum={minimum_retained_seconds:.6f}s, output={delivered.duration_seconds:.6f}s"
                    )
        if retained_reference is not None:
            if not retained_reference.is_silent and delivered.is_silent:
                raise AudioValidationError("Magic Clean unexpectedly erased the retained mix")
            retained_collapsed_channel = AudioValidator._collapsed_audible_channel(
                retained_reference, delivered
            )
            if retained_collapsed_channel is not None:
                raise AudioValidationError(
                    f"Magic Clean unexpectedly erased a retained-mix channel: channel={retained_collapsed_channel + 1}"
                )
            if retained_reference.is_silent and (not delivered.is_silent):
                raise AudioValidationError(
                    "Magic Clean unexpectedly produced audible audio from a silent retained mix"
                )
        if expect_audible and (not source.is_silent) and delivered.is_silent:
            raise AudioValidationError(
                "Magic Clean unexpectedly produced silent audio from an audible source"
            )
        if expect_audible:
            collapsed_channel = AudioValidator._collapsed_audible_channel(source, delivered)
            if collapsed_channel is not None:
                raise AudioValidationError(
                    f"Magic Clean unexpectedly erased an audible source channel: channel={collapsed_channel + 1}"
                )
        if source.is_silent and (not delivered.is_silent):
            raise AudioValidationError(
                "Magic Clean unexpectedly produced audible audio from a silent source"
            )
        if delivered.clipped_sample_count:
            raise AudioValidationError(
                f"Magic Clean artifact contains clipped decoded samples: count={delivered.clipped_sample_count}"
            )
        if delivered.is_silent:
            peak_db = -99.0
            lufs = -99.0
            snr = 0.0
            score = 0.0
        elif delivered.test_waveform is not None:
            peak_db = float(
                metrics.compute_true_peak_db(delivered.test_waveform, delivered.sample_rate)
            )
            lufs = float(metrics.compute_lufs(delivered.test_waveform))
            snr = AudioValidator._snr_from_frame_powers(delivered.frame_powers)
            score = float(metrics.compute_quality_score(snr, False, lufs, snr_available=snr != 0.0))
        else:
            lufs, peak_db = AudioValidator._measure_decoded_loudness_and_true_peak(output_path)
            snr = AudioValidator._snr_from_frame_powers(delivered.frame_powers)
            score = float(metrics.compute_quality_score(snr, False, lufs, snr_available=snr != 0.0))
        if not np.isfinite([peak_db, lufs, snr, score]).all():
            raise AudioValidationError("Magic Clean artifact measurements are non-finite")
        if peak_db > TRUE_PEAK_CEILING_DBTP + 1e-06:
            raise AudioValidationError(
                f"Magic Clean artifact exceeds the delivered true-peak ceiling: measured={peak_db:.2f} dBTP, ceiling={TRUE_PEAK_CEILING_DBTP:.2f} dBTP"
            )
        return DeliveredAudioMetrics(
            duration_seconds=delivered.duration_seconds,
            duration_delta_seconds=duration_delta,
            sample_rate=delivered.sample_rate,
            channels=delivered.channels,
            sample_count=delivered.sample_count,
            peak_db=round(peak_db, 2),
            lufs=round(lufs, 2),
            snr_db=round(snr, 2),
            clipping_detected=False,
            clipped_sample_count=0,
            quality_score=score,
        )

    @staticmethod
    def _scan_decoded_audio(path: str, *, label: str) -> _DecodedAudioScan:
        if not os.path.isfile(path):
            return AudioValidator._scan_injected_tensor(path, label=label)
        sample_rate, channels = AudioValidator._probe_audio_format(path, label=label)
        if channels not in {1, 2}:
            raise AudioValidationError(f"Decoded {label} audio must contain one or two channels")
        frame_samples = max(1, round(sample_rate * 0.02))
        values_per_power_frame = frame_samples * channels
        byte_remainder = b""
        power_remainder = np.empty(0, dtype=np.float32)
        frame_power_parts: list[np.ndarray] = []
        value_count = 0
        square_sum = 0.0
        absolute_peak = 0.0
        channel_value_counts = np.zeros(channels, dtype=np.int64)
        channel_square_sums = np.zeros(channels, dtype=np.float64)
        channel_absolute_peaks = np.zeros(channels, dtype=np.float64)
        clipped_sample_count = 0
        command = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            path,
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
            "pipe:1",
        ]
        with tempfile.TemporaryFile() as stderr:
            try:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr)
            except OSError as exc:
                raise AudioValidationError("ffmpeg executable is unavailable") from exc
            if process.stdout is None:
                process.kill()
                process.wait()
                raise AudioValidationError(f"Could not decode {label} audio")
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + FFMPEG_TIMEOUT_SECONDS
            try:
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, FFMPEG_TIMEOUT_SECONDS)
                    events = selector.select(timeout=min(30.0, remaining))
                    if not events:
                        if process.poll() is not None:
                            break
                        continue
                    raw = os.read(process.stdout.fileno(), PCM_READ_BYTES)
                    if not raw:
                        break
                    combined = byte_remainder + raw
                    complete_bytes = len(combined) - len(combined) % 4
                    byte_remainder = combined[complete_bytes:]
                    if not complete_bytes:
                        continue
                    values = np.frombuffer(combined[:complete_bytes], dtype="<f4")
                    if not np.isfinite(values).all():
                        raise AudioValidationError(f"Decoded {label} contains non-finite samples")
                    first_channel = value_count % channels
                    for channel in range(channels):
                        offset = (channel - first_channel) % channels
                        channel_values = values[offset::channels]
                        if not channel_values.size:
                            continue
                        channel_values_64 = channel_values.astype(np.float64)
                        channel_value_counts[channel] += channel_values.size
                        channel_square_sums[channel] += float(
                            np.dot(channel_values_64, channel_values_64)
                        )
                        channel_absolute_peaks[channel] = max(
                            channel_absolute_peaks[channel], float(np.max(np.abs(channel_values)))
                        )
                    value_count += values.size
                    absolute_peak = max(absolute_peak, float(np.max(np.abs(values))))
                    clipped_sample_count += int(np.count_nonzero(np.abs(values) >= 1.0))
                    square_sum += float(np.dot(values.astype(np.float64), values))
                    power_values = (
                        np.concatenate((power_remainder, values))
                        if power_remainder.size
                        else values
                    )
                    complete_power_values = (
                        power_values.size // values_per_power_frame * values_per_power_frame
                    )
                    if complete_power_values:
                        frames = power_values[:complete_power_values].reshape(
                            -1, frame_samples, channels
                        )
                        frame_power_parts.append(
                            np.mean(frames.astype(np.float64) ** 2, axis=(1, 2))
                        )
                    power_remainder = power_values[complete_power_values:].copy()
                return_code = process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except BaseException:
                process.kill()
                process.wait()
                raise
            finally:
                selector.close()
                process.stdout.close()
        if return_code != 0 or byte_remainder or value_count % channels:
            raise AudioValidationError(f"Could not decode {label} audio")
        sample_count = value_count // channels
        if sample_count < 1:
            raise AudioValidationError(f"Decoded {label} audio is empty")
        if not bool(np.all(channel_value_counts == sample_count)):
            raise AudioValidationError(f"Could not decode {label} audio")
        if power_remainder.size:
            frame_power_parts.append(np.array([np.mean(power_remainder.astype(np.float64) ** 2)]))
        frame_powers = (
            np.concatenate(frame_power_parts)
            if frame_power_parts
            else np.empty(0, dtype=np.float64)
        )
        return _DecodedAudioScan(
            sample_rate=sample_rate,
            channels=channels,
            sample_count=sample_count,
            absolute_peak=absolute_peak,
            rms=math.sqrt(square_sum / value_count),
            channel_absolute_peaks=tuple(float(value) for value in channel_absolute_peaks),
            channel_rms=tuple(
                math.sqrt(float(channel_square_sums[channel]) / sample_count)
                for channel in range(channels)
            ),
            clipped_sample_count=clipped_sample_count,
            frame_powers=frame_powers,
        )

    @staticmethod
    def _scan_injected_tensor(path: str, *, label: str) -> _DecodedAudioScan:
        try:
            waveform, sample_rate = AudioIO.load(path)
        except Exception as exc:
            raise AudioValidationError(f"Could not decode {label} audio") from exc
        if sample_rate <= 0:
            raise AudioValidationError(f"Decoded {label} audio has an invalid sample rate")
        waveform = waveform.detach().to(device="cpu", dtype=torch.float32)
        AudioValidator._validate_pcm_shape_and_values(waveform, label=f"Decoded {label}")
        frame_samples = max(1, round(sample_rate * 0.02))
        frame_count = waveform.shape[1] // frame_samples
        if frame_count:
            core = waveform[:, : frame_count * frame_samples]
            powers = core.square().reshape(waveform.shape[0], frame_count, frame_samples)
            frame_powers = powers.mean(dim=(0, 2)).numpy()
        else:
            frame_powers = np.array([waveform.square().mean().item()])
        return _DecodedAudioScan(
            sample_rate=int(sample_rate),
            channels=int(waveform.shape[0]),
            sample_count=int(waveform.shape[1]),
            absolute_peak=float(waveform.abs().max().item()),
            rms=float(waveform.square().mean().sqrt().item()),
            channel_absolute_peaks=tuple(
                float(value) for value in waveform.abs().amax(dim=1).tolist()
            ),
            channel_rms=tuple(
                float(value) for value in waveform.square().mean(dim=1).sqrt().tolist()
            ),
            clipped_sample_count=int((waveform.abs() >= 1.0).sum().item()),
            frame_powers=frame_powers,
            test_waveform=waveform,
        )

    @staticmethod
    def _collapsed_audible_channel(
        source: _DecodedAudioScan, delivered: _DecodedAudioScan
    ) -> int | None:
        """Return the first audible source channel erased in the delivery."""
        for channel, (source_peak, source_rms, delivered_peak, delivered_rms) in enumerate(
            zip(
                source.channel_absolute_peaks,
                source.channel_rms,
                delivered.channel_absolute_peaks,
                delivered.channel_rms,
                strict=True,
            )
        ):
            peak_is_audible = source_peak >= SILENCE_PEAK_THRESHOLD
            rms_is_audible = source_rms >= SILENCE_RMS_THRESHOLD
            peak_collapsed = peak_is_audible and delivered_peak < max(
                SILENCE_PEAK_THRESHOLD, source_peak * CHANNEL_COLLAPSE_RELATIVE_THRESHOLD
            )
            rms_collapsed = rms_is_audible and delivered_rms < max(
                SILENCE_RMS_THRESHOLD, source_rms * CHANNEL_COLLAPSE_RELATIVE_THRESHOLD
            )
            if peak_collapsed or rms_collapsed:
                return channel
        return None

    @staticmethod
    def _probe_audio_format(path: str, *, label: str) -> tuple[int, int]:
        try:
            completed = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "a:0",
                    "-show_entries",
                    "stream=sample_rate,channels",
                    "-of",
                    "json",
                    path,
                ],
                capture_output=True,
                check=True,
                text=True,
                timeout=30,
            )
            stream = json.loads(completed.stdout)["streams"][0]
            sample_rate = int(stream["sample_rate"])
            channels = int(stream["channels"])
        except (
            OSError,
            subprocess.SubprocessError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:
            raise AudioValidationError(f"Could not inspect {label} audio") from exc
        if sample_rate <= 0 or channels <= 0:
            raise AudioValidationError(f"Decoded {label} audio has an invalid format")
        return (sample_rate, channels)

    @staticmethod
    def _measure_decoded_loudness_and_true_peak(path: str) -> tuple[float, float]:
        try:
            completed = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-nostats",
                    "-xerror",
                    "-i",
                    path,
                    "-af",
                    "loudnorm=I=-16:LRA=11:TP=-1:print_format=json",
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                check=True,
                text=True,
                timeout=FFMPEG_TIMEOUT_SECONDS,
            )
            matches = re.findall("\\{[^{]*\\}", completed.stderr, flags=re.DOTALL)
            payload = json.loads(matches[-1])
            lufs = float(payload["input_i"])
            true_peak = float(payload["input_tp"])
        except (
            OSError,
            subprocess.SubprocessError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:
            raise AudioValidationError(
                "Could not measure delivered loudness and true peak"
            ) from exc
        if lufs == -math.inf:
            lufs = -99.0
        if not math.isfinite(lufs) or not math.isfinite(true_peak):
            raise AudioValidationError("Delivered loudness or true-peak measurement is non-finite")
        return (lufs, true_peak)

    @staticmethod
    def _minimum_retained_activity_seconds(source: _DecodedAudioScan) -> float:
        """Return a conservative lower bound for a silence-edited timeline."""
        if source.is_silent:
            return 0.0
        if source.frame_powers.size < 1:
            return source.duration_seconds
        rms_values = np.sqrt(np.maximum(source.frame_powers, 0.0))
        high_reference = float(np.max(rms_values))
        if high_reference < 1e-10:
            return 0.0
        low_percentile = 20 if source.duration_seconds < 2.0 else 30
        low_reference = float(np.percentile(rms_values, low_percentile))
        if low_reference / (high_reference + 1e-12) > 0.65:
            return source.duration_seconds
        threshold = max(
            math.sqrt(max(low_reference, 1e-12) * high_reference), high_reference * 0.0158
        )
        protective_threshold = min(
            threshold, low_reference * 1.1 + max(high_reference * 0.0001, 1e-07)
        )
        active_frames = int(np.count_nonzero(rms_values > protective_threshold))
        return source.duration_seconds * active_frames / rms_values.size

    @staticmethod
    def _snr_from_frame_powers(frame_powers: np.ndarray) -> float:
        if frame_powers.size < 10:
            return 0.0
        noise_power = float(np.percentile(frame_powers, 20))
        signal_power = float(np.percentile(frame_powers, 80))
        if signal_power <= 1e-10 or signal_power <= noise_power * 1.05:
            return 0.0
        return float(10 * np.log10(signal_power / max(noise_power, 1e-10)))

    @staticmethod
    def _validate_pcm_shape_and_values(waveform: torch.Tensor, *, label: str) -> None:
        if not isinstance(waveform, torch.Tensor) or waveform.ndim != 2:
            raise AudioValidationError(f"{label} must be two-dimensional channel-first PCM")
        if waveform.shape[0] < 1:
            raise AudioValidationError(f"{label} has no audio channels")
        if waveform.shape[1] < 1:
            raise AudioValidationError(f"{label} is empty")
        if not bool(torch.isfinite(waveform).all().item()):
            raise AudioValidationError(f"{label} contains non-finite samples")
