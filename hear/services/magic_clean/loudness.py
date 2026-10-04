from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy import signal

HOP_SECONDS = 0.1
BLOCK_HOPS = 4
ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_LU = -10.0
TRUE_PEAK_RATE = 192000
TRUE_PEAK_SAFETY_DB = 0.05

@dataclass(frozen=True)
class BlockMeasurement:
    block_powers: np.ndarray
    peak_linear: float
    frames: int

    @staticmethod
    def empty() -> BlockMeasurement:
        return BlockMeasurement(np.zeros(0, dtype=np.float64), 0.0, 0)


class KWeightedMeter:
    ITU_48K = (
        ([1.53512485958697, -2.69169618940638, 1.19839281085285],
         [1.0, -1.69065929318241, 0.73248077421585]),
        ([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621]),
    )

    @classmethod
    def filters(cls, rate: int) -> tuple[tuple[list[float], list[float]], ...]:
        if rate == 48000:
            return cls.ITU_48K
        return cls.design(rate)

    @classmethod
    def design(cls, rate: int) -> tuple[tuple[list[float], list[float]], ...]:
        gain_db, shelf_q, shelf_fc = 3.999843853973347, 0.7071752369554196, 1681.974450955533
        k = math.tan(math.pi * shelf_fc / rate)
        vh = 10 ** (gain_db / 20.0)
        vb = vh**0.499666774155
        norm = 1.0 + k / shelf_q + k * k
        shelf = (
            [
                (vh + vb * k / shelf_q + k * k) / norm,
                2.0 * (k * k - vh) / norm,
                (vh - vb * k / shelf_q + k * k) / norm,
            ],
            [1.0, 2.0 * (k * k - 1.0) / norm, (1.0 - k / shelf_q + k * k) / norm],
        )
        hp_q, hp_fc = 0.5003270373238773, 38.13547087602444
        k = math.tan(math.pi * hp_fc / rate)
        norm = 1.0 + k / hp_q + k * k
        high_pass = ([1.0, -2.0, 1.0], [1.0, 2.0 * (k * k - 1.0) / norm, (1.0 - k / hp_q + k * k) / norm])
        return shelf, high_pass

    @classmethod
    def weighted(cls, samples: np.ndarray, rate: int) -> np.ndarray:
        values = np.asarray(samples, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        for b, a in cls.filters(rate):
            values = signal.lfilter(b, a, values, axis=0)
        return values

    @classmethod
    def measure(
        cls,
        samples: np.ndarray,
        rate: int,
        *,
        discard_frames: int = 0,
        keep_frames: int | None = None,
    ) -> BlockMeasurement:
        values = np.asarray(samples, dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        if not len(values):
            return BlockMeasurement.empty()
        stop = None if keep_frames is None else discard_frames + keep_frames
        weighted = cls.weighted(values, rate)[discard_frames:stop]
        kept = values[discard_frames:stop]
        if not len(kept):
            return BlockMeasurement.empty()
        hop = round(rate * HOP_SECONDS)
        hops = len(weighted) // hop
        powers = np.zeros(0, dtype=np.float64)
        if hops >= BLOCK_HOPS:
            # Channel weights are 1.0 for mono and left/right; power sums across channels.
            energy = (weighted[: hops * hop] ** 2).reshape(hops, hop, -1).sum(axis=(1, 2))
            powers = np.convolve(energy, np.ones(BLOCK_HOPS), mode="valid") / (hop * BLOCK_HOPS)
        return BlockMeasurement(powers, cls.peak_linear(kept, rate), len(kept))

    @classmethod
    def peak_linear(cls, samples: np.ndarray, rate: int) -> float:
        values = np.asarray(samples, dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        if not len(values):
            return 0.0
        factor = math.gcd(TRUE_PEAK_RATE, rate)
        up, down = TRUE_PEAK_RATE // factor, rate // factor
        sample_peak = float(np.max(np.abs(values)))
        if (up == 1 and down == 1) or sample_peak == 0.0:
            return sample_peak
        hop = round(rate * HOP_SECONDS)
        hops = -(-len(values) // hop)
        padded = np.zeros((hops * hop, values.shape[1]), dtype=np.float32)
        padded[: len(values)] = values
        hop_peaks = np.abs(padded).reshape(hops, hop, -1).max(axis=(1, 2))
        candidates = np.flatnonzero(hop_peaks >= sample_peak * 10 ** (-3 / 20))
        peak = sample_peak
        for start, end in cls._merge_runs(candidates, hops):
            block = padded[max(0, start - 1) * hop : min(hops, end + 1) * hop]
            for channel in range(block.shape[1]):
                oversampled = signal.resample_poly(block[:, channel], up, down)
                peak = max(peak, float(np.max(np.abs(oversampled))))
        return peak

    @staticmethod
    def _merge_runs(indices: np.ndarray, limit: int) -> list[tuple[int, int]]:
        """Group consecutive hop indices into [start, end) runs."""
        runs: list[tuple[int, int]] = []
        for index in indices.tolist():
            if runs and runs[-1][1] == index:
                runs[-1] = (runs[-1][0], index + 1)
            else:
                runs.append((index, index + 1))
        return [(a, min(b, limit)) for a, b in runs]

    @classmethod
    def measure_ranges(
        cls, path: str, rate: int, ranges: list[tuple[int, int]], *, warmup_frames: int
    ) -> list[BlockMeasurement]:
        results = []
        with sf.SoundFile(path) as audio:
            for start, end in ranges:
                lead = min(warmup_frames, start)
                audio.seek(start - lead)
                data = audio.read(end - start + lead, dtype="float32", always_2d=True)
                results.append(cls.measure(data, rate, discard_frames=lead, keep_frames=end - start))
        return results

    @staticmethod
    def integrate(measurements: list[BlockMeasurement]) -> float | None:
        powers = np.concatenate([m.block_powers for m in measurements]) if measurements else np.zeros(0)
        if not len(powers):
            return None
        with np.errstate(divide="ignore"):
            loudness = -0.691 + 10 * np.log10(powers)
        absolute = loudness > ABSOLUTE_GATE_LUFS
        if not absolute.any():
            return None
        threshold = -0.691 + 10 * math.log10(float(powers[absolute].mean())) + RELATIVE_GATE_LU
        gated = absolute & (loudness > threshold)
        if not gated.any():
            return None
        return float(-0.691 + 10 * math.log10(float(powers[gated].mean())))

    @staticmethod
    def too_quiet(measurements: list[BlockMeasurement], rate: int) -> str | None:
        frames = sum(m.frames for m in measurements)
        if frames < 0.4 * rate:
            return "too_short"
        return None if KWeightedMeter.integrate(measurements) is not None else "below_measurement_gate"

    @staticmethod
    def true_peak_dbtp(measurements: list[BlockMeasurement]) -> float | None:
        peak = max((m.peak_linear for m in measurements), default=0.0)
        if peak <= 0:
            return None
        return round(20 * math.log10(peak) + TRUE_PEAK_SAFETY_DB, 6)
