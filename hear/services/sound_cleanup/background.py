"""Evidence-driven stationary noise cleanup; never assumes every bass tone is hum."""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import ndimage, signal

from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode
from hear.services.magic_clean.mastering import AudioMasteringService


@dataclass(frozen=True)
class BackgroundPolicy:
    version: str = "background-stationary-v1"
    maximum_voice_reduction_db: float = 4.0
    maximum_quiet_reduction_db: float = 12.0
    minimum_reference_seconds: float = 0.5


class BackgroundCleanup:
    RATE = 48000
    FFT = 2048
    HOP = 512
    BLOCK = 512 * 900

    def __init__(self, policy: BackgroundPolicy | None = None):
        self.policy = policy or BackgroundPolicy()

    @staticmethod
    def persistent_mains(path: Path, guard: ResourceGuard) -> list[float]:
        """Detect stable narrow lines, not broad 50/60-Hz neighbourhood energy."""
        spectra = []
        with sf.SoundFile(path) as audio:
            if audio.samplerate != 48000 or audio.frames < 96000:
                return []
            for start in np.linspace(
                0, audio.frames - 96000, min(80, audio.frames // 48000)
            ).astype(int):
                guard.check()
                audio.seek(int(start))
                samples = audio.read(96000, dtype="float64", always_2d=True)
                if not np.isfinite(samples).all():
                    raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "nonfinite_background_input")
                freq, power = signal.welch(samples, fs=48000, nperseg=96000, axis=0)
                spectra.append(np.max(power, axis=1))
        powers = np.asarray(spectra)
        nearby = ndimage.median_filter(powers, size=(1, 41), mode="nearest")
        prominence = 10 * np.log10(np.maximum(powers, 1e-25) / np.maximum(nearby, 1e-25))
        stable = (np.mean(prominence > 12, axis=0) >= 0.70) & (np.median(prominence, axis=0) >= 12)
        selected = []
        for fundamental in (50, 60):
            family = []
            for multiple in range(1, 11):
                expected = fundamental * multiple
                bins = np.flatnonzero((np.abs(freq - expected) <= 1) & stable)
                if len(bins):
                    index = int(bins[np.argmax(np.median(powers[:, bins], axis=0))])
                    family.append(float(freq[index]))
            if any(abs(f - fundamental) <= 1 for f in family) or len(family) >= 3:
                selected.extend(family)
        return sorted(set(selected))[:8]

    def render(self, source: Path, target: Path, evidence, guard: ResourceGuard) -> dict:
        guard.check()
        if target.exists() or source.is_symlink() or target.is_symlink():
            raise CleanExecutionError(ErrorCode.ARTIFACT_CONFLICT, "background_output_conflict")
        for path in (source, target):
            if not path.resolve().is_relative_to(guard.workspace.resolve()):
                raise CleanExecutionError(
                    ErrorCode.INVALID_AUDIO, "background_path_outside_workspace"
                )
        with sf.SoundFile(source) as audio:
            frames, channels = audio.frames, audio.channels
            if audio.samplerate != self.RATE or channels not in (1, 2) or frames != evidence.frames:
                raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "background_layout_mismatch")
        guard.preflight_pcm(frames, channels, copies=1, output_bytes=8192)
        voice = evidence.speech >= 0.8
        protected = evidence.protected
        levels = np.max(evidence.clean_rms, axis=1)
        threshold = (
            float(np.quantile(levels[voice], 0.2)) * 0.5 if np.count_nonzero(voice) > 10 else 0
        )
        eligible = (~protected) & (evidence.speech < 0.03) & (levels < threshold)
        powers = []
        with sf.SoundFile(source) as audio:
            selected = np.flatnonzero(eligible)
            if len(selected) > 900:
                selected = selected[np.linspace(0, len(selected) - 1, 900).astype(int)]
            for index in selected:
                start = int(index) * evidence.step_frames
                lo = max(0, (start - self.FFT // 2) // evidence.step_frames)
                hi = min(len(eligible), math.ceil((start + self.FFT // 2) / evidence.step_frames))
                if (
                    start < self.FFT // 2
                    or start + self.FFT // 2 > frames
                    or not np.all(eligible[lo:hi])
                ):
                    continue
                guard.check()
                audio.seek(start - self.FFT // 2)
                data = audio.read(self.FFT, dtype="float64", always_2d=True)
                _, _, spectrum = signal.stft(
                    data,
                    fs=self.RATE,
                    nperseg=self.FFT,
                    noverlap=0,
                    axis=0,
                    boundary=None,
                    padded=False,
                )
                powers.append(np.abs(spectrum[:, :, 0]) ** 2)
        reference_seconds = len(powers) * evidence.step_frames / self.RATE
        noise = (
            np.median(powers, axis=0)
            if reference_seconds >= self.policy.minimum_reference_seconds
            else None
        )
        frequencies = self.persistent_mains(source, guard)
        # Hum removal is narrow-band and conditional on sustained line evidence.
        sos = (
            np.vstack(
                [signal.tf2sos(*signal.iirnotch(f, max(20, f / 2), self.RATE)) for f in frequencies]
            )
            if frequencies
            else None
        )
        state = np.zeros((len(sos), 2, channels)) if sos is not None else None
        hum_source = source
        temporary = guard.workspace / "background-dehum-intermediate.wav"
        owned_temporary = False
        owned_target = False
        try:
            if sos is not None:
                if temporary.exists():
                    raise CleanExecutionError(
                        ErrorCode.ARTIFACT_CONFLICT, "background_intermediate_exists"
                    )
                with (
                    sf.SoundFile(source) as audio,
                    temporary.open("xb") as temp_stream,
                    sf.SoundFile(
                        temp_stream,
                        "w",
                        format="WAV",
                        samplerate=self.RATE,
                        channels=channels,
                        subtype="FLOAT",
                    ) as out,
                ):
                    owned_temporary = True
                    while True:
                        guard.check()
                        block = audio.read(32768, dtype="float64", always_2d=True)
                        if not len(block):
                            break
                        filtered, state = signal.sosfilt(sos, block, axis=0, zi=state)
                        out.write(filtered)
                hum_source = temporary
            with sf.SoundFile(hum_source) as audio, target.open("xb") as stream:
                owned_target = True
                with sf.SoundFile(
                    stream,
                    "w",
                    format="WAV",
                    samplerate=self.RATE,
                    channels=channels,
                    subtype="FLOAT",
                ) as out:
                    for start in range(0, frames, self.BLOCK):
                        guard.check_scratch()
                        end = min(frames, start + self.BLOCK)
                        left = max(0, start - 2 * self.FFT)
                        right = min(frames, end + 2 * self.FFT)
                        audio.seek(left)
                        data = audio.read(right - left, dtype="float64", always_2d=True)
                        if not np.isfinite(data).all():
                            raise CleanExecutionError(
                                ErrorCode.INVALID_AUDIO, "invalid_background_audio"
                            )
                        if noise is not None:
                            padded = np.pad(data, ((0, max(0, self.FFT - len(data))), (0, 0)))
                            _, times, s = signal.stft(
                                padded,
                                fs=self.RATE,
                                nperseg=self.FFT,
                                noverlap=self.FFT - self.HOP,
                                axis=0,
                                boundary="zeros",
                            )
                            indices = np.minimum(
                                ((times * self.RATE + left) // evidence.step_frames).astype(int),
                                len(protected) - 1,
                            )
                            ceiling = np.where(
                                protected[indices],
                                self.policy.maximum_voice_reduction_db,
                                self.policy.maximum_quiet_reduction_db,
                            )
                            floor = 10 ** (-ceiling / 20)
                            gain = np.sqrt(
                                np.maximum(
                                    0,
                                    1 - 1.2 * noise[:, :, None] / np.maximum(np.abs(s) ** 2, 1e-25),
                                )
                            )
                            gain = ndimage.uniform_filter1d(
                                ndimage.uniform_filter1d(gain, 3, axis=0), 5, axis=2
                            )
                            # One linked gain preserves stereo balance and protects either channel.
                            gain = np.maximum(np.max(gain, axis=1), floor[None, :])
                            _, rendered = signal.istft(
                                s * gain[:, None, :],
                                fs=self.RATE,
                                nperseg=self.FFT,
                                noverlap=self.FFT - self.HOP,
                                time_axis=2,
                                freq_axis=0,
                            )
                            data = rendered.T[: len(data)]
                        out.write(data[start - left : end - left].astype("float32"))
            AudioMasteringService.scan(
                target, guard, rate=self.RATE, channels=channels, frames=frames
            )
        except BaseException:
            if owned_target:
                target.unlink(missing_ok=True)
            raise
        finally:
            if owned_temporary:
                temporary.unlink(missing_ok=True)
        return {
            "version": self.policy.version,
            "status": "processed"
            if noise is not None or frequencies
            else "no_confident_stationary_reference",
            "detected_mains_lines_hz": frequencies,
            "noise_reference_seconds": round(reference_seconds, 3),
            "spectral_cleanup_applied": noise is not None,
            "maximum_voice_reduction_db": self.policy.maximum_voice_reduction_db,
            "maximum_quiet_reduction_db": self.policy.maximum_quiet_reduction_db,
            "requires_listening_review": True,
            "hum_removal_certified": False,
        }
