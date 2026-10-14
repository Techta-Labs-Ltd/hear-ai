from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, welch

from hear.runtime.cleaner.resource_guard import ResourceGuard


@dataclass(frozen=True)
class AcousticMatchReport:
    context_level_dbfs: float | None
    generated_level_dbfs: float | None
    gain_db: float
    noise_floor_dbfs: float | None
    room_tone_added: bool
    lowpass_hz: float | None
    context_seconds: float


class AcousticMatcher:
    """Per-segment matching of generated speech to its surrounding source audio."""

    RATE = 48000
    FRAME = 960  # 20 ms analysis frames
    CONTEXT_SECONDS = 2.0
    MAX_GAIN_DB = 12.0
    ROOM_TONE_SECONDS = 0.5
    MIN_ROOM_TONE_SECONDS = 0.25
    CROSSFADE = 960  # 20 ms
    # Bandwidth is matched only when the source is clearly band-limited; a 99 % energy
    # roll-off above this leaves the generated audio untouched.
    MAX_LOWPASS_HZ = 16000.0
    MIN_LOWPASS_HZ = 3000.0
    SILENCE_DBFS = -80.0

    @staticmethod
    def _dbfs(value: float) -> float:
        return float(20 * np.log10(max(value, 1e-9)))

    @staticmethod
    def frame_rms(values: np.ndarray) -> np.ndarray:
        count = len(values) // AcousticMatcher.FRAME
        if count == 0:
            return np.array([float(np.sqrt(np.mean(values.astype("float64") ** 2)))])
        block = (
            values[: count * AcousticMatcher.FRAME]
            .astype("float64")
            .reshape(count, AcousticMatcher.FRAME)
        )
        return np.sqrt(np.mean(block**2, axis=1))

    @staticmethod
    def voiced_level(values: np.ndarray) -> tuple[float | None, float | None]:
        """(voiced RMS, distinct noise floor RMS) for mono float audio.

        Voiced frames sit within 20 dB of the loudest frames, which holds for continuous
        speech as well as speech with pauses. The floor is only reported when the quiet
        frames are clearly separated from the voiced ones, i.e. the audio has pauses.
        """
        rms = AcousticMatcher.frame_rms(values)
        if len(rms) < 5:
            return None, None
        loud = float(np.percentile(rms, 95))
        if loud < 10 ** (AcousticMatcher.SILENCE_DBFS / 20):
            return None, None
        voiced = rms[rms > loud * 0.1]
        level = float(np.sqrt(np.mean(voiced**2)))
        floor = float(np.percentile(rms, 10))
        return level, (floor if floor < loud * 0.1 else None)

    @staticmethod
    def rolloff_hz(values: np.ndarray) -> float | None:
        """Frequency below which 99 % of the context energy lies."""
        if len(values) < 4096:
            return None
        freqs, power = welch(values.astype("float64"), fs=AcousticMatcher.RATE, nperseg=4096)
        total = float(np.sum(power))
        if total <= 0:
            return None
        cumulative = np.cumsum(power) / total
        return float(freqs[int(np.searchsorted(cumulative, 0.99))])

    @staticmethod
    def quietest_window(values: np.ndarray, length: int) -> np.ndarray | None:
        if len(values) < length:
            return None
        rms = AcousticMatcher.frame_rms(values)
        per = max(1, length // AcousticMatcher.FRAME)
        if len(rms) < per:
            return None
        sums = np.convolve(rms**2, np.ones(per), mode="valid")
        start = int(np.argmin(sums)) * AcousticMatcher.FRAME
        return values[start : start + length]

    @staticmethod
    def tile_room_tone(window: np.ndarray, frames: int) -> np.ndarray:
        """Loop a quiet window palindromically with 20 ms crossfades to cover `frames`."""
        if len(window) < 2 * AcousticMatcher.CROSSFADE + 1 or frames <= 0:
            return np.zeros(max(frames, 0), dtype="float32")
        fade = np.linspace(0, 1, AcousticMatcher.CROSSFADE, dtype="float32")
        piece = window.astype("float32").copy()
        piece[: AcousticMatcher.CROSSFADE] *= fade
        piece[-AcousticMatcher.CROSSFADE :] *= fade[::-1]
        out = np.zeros(frames + len(piece), dtype="float32")
        position = 0
        flip = False
        while position < frames:
            # Alternating direction keeps the loop seam continuous in level and phase.
            out[position : position + len(piece)] += piece[::-1] if flip else piece
            flip = not flip
            position += len(piece) - AcousticMatcher.CROSSFADE
        return out[:frames]

    @staticmethod
    def read_context(source: Path, start: int, end: int, channel: int) -> np.ndarray:
        """Mono source audio within CONTEXT_SECONDS on either side of [start, end)."""
        span = round(AcousticMatcher.CONTEXT_SECONDS * AcousticMatcher.RATE)
        parts = []
        with sf.SoundFile(source) as audio:
            before = max(0, start - span)
            if start - before > 0:
                audio.seek(before)
                parts.append(
                    audio.read(start - before, dtype="float32", always_2d=True)[:, channel]
                )
            after = min(audio.frames, end + span)
            if after - end > 0:
                audio.seek(end)
                parts.append(audio.read(after - end, dtype="float32", always_2d=True)[:, channel])
        if not parts:
            return np.zeros(0, dtype="float32")
        return np.concatenate(parts)

    @staticmethod
    def match_channel(
        generated: np.ndarray, context: np.ndarray
    ) -> tuple[np.ndarray, AcousticMatchReport]:
        context_level, floor = (
            AcousticMatcher.voiced_level(context) if len(context) else (None, None)
        )
        generated_level, _ = AcousticMatcher.voiced_level(generated)
        # Fish output has digital silence in its pauses and at its edges; the quietest
        # 100 ms decides whether the source's room tone has to be laid underneath.
        generated_rms = AcousticMatcher.frame_rms(generated)
        span = min(5, len(generated_rms))
        generated_floor = (
            float(
                np.sqrt(np.min(np.convolve(generated_rms**2, np.ones(span) / span, mode="valid")))
            )
            if span
            else 0.0
        )
        gain_db = 0.0
        if context_level is not None and generated_level is not None:
            gain_db = float(
                np.clip(
                    AcousticMatcher._dbfs(context_level) - AcousticMatcher._dbfs(generated_level),
                    -AcousticMatcher.MAX_GAIN_DB,
                    AcousticMatcher.MAX_GAIN_DB,
                )
            )
        out = generated.astype("float32") * np.float32(10 ** (gain_db / 20))
        lowpass = None
        rolloff = AcousticMatcher.rolloff_hz(context) if len(context) else None
        if (
            rolloff is not None
            and AcousticMatcher.MIN_LOWPASS_HZ <= rolloff < AcousticMatcher.MAX_LOWPASS_HZ
        ):
            lowpass = rolloff
            sos = butter(6, lowpass, btype="low", fs=AcousticMatcher.RATE, output="sos")
            out = sosfiltfilt(sos, out.astype("float64")).astype("float32")
        room_tone = False
        if floor is not None and AcousticMatcher._dbfs(floor) > AcousticMatcher.SILENCE_DBFS + 10:
            if (
                AcousticMatcher._dbfs(generated_floor * 10 ** (gain_db / 20))
                < AcousticMatcher._dbfs(floor) - 6
            ):
                window = AcousticMatcher.quietest_window(
                    context, round(AcousticMatcher.ROOM_TONE_SECONDS * AcousticMatcher.RATE)
                )
                if window is None:
                    window = AcousticMatcher.quietest_window(
                        context, round(AcousticMatcher.MIN_ROOM_TONE_SECONDS * AcousticMatcher.RATE)
                    )
                if window is not None:
                    out = out + AcousticMatcher.tile_room_tone(window, len(out))
                    room_tone = True
        peak = float(np.max(np.abs(out))) if len(out) else 0.0
        if peak > 0.98:
            out = out * np.float32(0.98 / peak)
        report = AcousticMatchReport(
            context_level_dbfs=None
            if context_level is None
            else round(AcousticMatcher._dbfs(context_level), 2),
            generated_level_dbfs=None
            if generated_level is None
            else round(AcousticMatcher._dbfs(generated_level), 2),
            gain_db=round(gain_db, 2),
            noise_floor_dbfs=None if floor is None else round(AcousticMatcher._dbfs(floor), 2),
            room_tone_added=room_tone,
            lowpass_hz=None if lowpass is None else round(lowpass, 1),
            context_seconds=round(len(context) / AcousticMatcher.RATE, 3),
        )
        return out, report

    @staticmethod
    def match_segment(
        source: Path,
        generated: Path,
        target: Path,
        start: int,
        end: int,
        guard: ResourceGuard,
    ) -> dict:
        """Write `generated` to `target` matched to the source around [start, end)."""
        guard.check()
        with sf.SoundFile(generated) as audio:
            values = audio.read(dtype="float32", always_2d=True)
            channels = audio.channels
            if audio.samplerate != AcousticMatcher.RATE:
                raise ValueError("acoustic_match_requires_48k")
        reports = []
        matched = np.empty_like(values)
        for channel in range(channels):
            guard.check()
            context = AcousticMatcher.read_context(source, start, end, channel)
            matched[:, channel], report = AcousticMatcher.match_channel(values[:, channel], context)
            reports.append(report)
        if not np.isfinite(matched).all():
            raise ValueError("acoustic_match_nonfinite")
        with target.open("xb") as stream:
            sf.write(stream, matched, AcousticMatcher.RATE, format="WAV", subtype="FLOAT")
        return {
            "channels": [asdict(item) for item in reports],
            "gain_db": round(float(np.mean([r.gain_db for r in reports])), 2),
            "room_tone_added": any(r.room_tone_added for r in reports),
            "lowpass_hz": reports[0].lowpass_hz,
        }
