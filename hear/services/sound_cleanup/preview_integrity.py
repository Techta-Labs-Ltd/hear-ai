"""Reject new tonal/DC artifacts; this gate does not certify retained words."""

from typing import Any

import numpy as np
from scipy import ndimage, signal


class PreviewIntegrity:
    POLICY = "preview-integrity-v1-full-target-no-new-tone"
    RATE = 48000

    @classmethod
    def assess(cls, before: np.ndarray, after: np.ndarray) -> tuple[str, dict[str, Any]]:
        if (
            before.ndim != 2
            or before.shape != after.shape
            or not len(before)
            or before.shape[1] not in (1, 2)
            or not np.isfinite(before).all()
            or not np.isfinite(after).all()
        ):
            return "preview_integrity_invalid_samples", {"policy": cls.POLICY}
        original = before.astype(np.float64)
        candidate = after.astype(np.float64)
        rms = np.sqrt(np.mean(original**2, axis=0))
        shift = np.abs(np.mean(candidate - original, axis=0))
        metrics: dict[str, Any] = {
            "policy": cls.POLICY,
            "dc_change_by_channel": shift.tolist(),
            "introduced_tones": [],
            "assessment": "passed",
        }
        if np.any(shift > np.maximum(0.001, 0.1 * rms)):
            metrics["assessment"] = "rejected"
            return "introduced_dc_artifact", metrics
        if len(original) < 1024:
            metrics["tone_check"] = "interval_too_short"
            return "", metrics
        fft = min(8192, len(original))
        for channel in range(original.shape[1]):
            freq, _, x = signal.stft(
                original[:, channel],
                fs=cls.RATE,
                nperseg=fft,
                noverlap=3 * fft // 4,
                boundary=None,
                padded=False,
            )
            _, _, y = signal.stft(
                candidate[:, channel],
                fs=cls.RATE,
                nperseg=fft,
                noverlap=3 * fft // 4,
                boundary=None,
                padded=False,
            )
            px, py = np.abs(x) ** 2, np.abs(y) ** 2
            # Detect output-only, persistent narrow tones. A voice harmonic
            # already present in the source is not grounds for notching it out.
            neighbours = ndimage.median_filter(py, size=(17, 1), mode="nearest")
            introduced = (
                (py > 4.0 * px + 1e-10)
                & (py > 16.0 * np.maximum(neighbours, 1e-14))
                & (py > 2.5e-9)
            )
            persistent = np.mean(introduced, axis=1) >= 0.6
            suspect = np.flatnonzero(persistent & (freq >= 20) & (freq <= 3000))
            for index in suspect[:8]:
                metrics["introduced_tones"].append(
                    {
                        "channel": channel,
                        "frequency_hz": round(float(freq[index]), 2),
                        "fraction_of_frames": round(float(np.mean(introduced[index])), 4),
                    }
                )
        if metrics["introduced_tones"]:
            metrics["assessment"] = "rejected"
            return "introduced_tonal_artifact", metrics
        return "", metrics
