import math
import threading
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch
from demucs.apply import apply_model
from demucs.htdemucs import HTDemucs
from demucs.pretrained import get_model

from .audio_io import AudioIO


class StemSeparator:
    def __init__(self, device: torch.device):
        self._demucs: Any | None = None
        self._device = device
        self._lock = threading.Lock()

    def load(self, model_name: str, model_path: str | Path):
        model_root = Path(model_path)
        if not model_root.is_dir():
            raise FileNotFoundError(f"Demucs model repository is missing: {model_root}")



        numpy_multiarray: Any = np.core.multiarray
        numpy_scalar: Any = numpy_multiarray.scalar
        safe_globals: list[Any] = [
            HTDemucs,
            Fraction,



            (numpy_scalar, "numpy.core.multiarray.scalar"),
            np.dtype,
            type(np.dtype(np.float64)),
        ]
        with torch.serialization.safe_globals(safe_globals):
            model = get_model(model_name, repo=model_root)
        if model is None:
            raise RuntimeError(f"Demucs model repository did not provide {model_name}")
        model.to(self._device)
        model.eval()
        self._demucs = model

    @property
    def is_loaded(self) -> bool:
        return self._demucs is not None

    def separate(self, waveform: torch.Tensor, sr: int) -> dict[str, torch.Tensor]:
        if self._demucs is None:
            raise RuntimeError("Demucs is not loaded")
        if sr <= 0 or waveform.ndim != 2 or waveform.shape[1] < 1:
            raise ValueError("Demucs expects non-empty [channels, samples] audio")
        if waveform.shape[0] not in {1, 2}:
            raise ValueError("Magic Clean supports mono or stereo stem separation")
        if not torch.isfinite(waveform).all():
            raise ValueError("Demucs input contains non-finite samples")

        original_len = waveform.shape[1]
        original_channels = waveform.shape[0]
        stereo = waveform.repeat(2, 1) if original_channels == 1 else waveform
        model_sample_rate = int(self._demucs.samplerate)
        if model_sample_rate <= 0:
            raise RuntimeError("Demucs model has an invalid sample rate")
        resampled = AudioIO.resample(stereo, sr, model_sample_rate)
        source_names = tuple(self._demucs.sources)
        if not source_names:
            raise RuntimeError("Demucs model declares no sources")

        with self._lock:
            with torch.inference_mode():
                model_output = apply_model(
                    self._demucs,
                    resampled[None],
                    progress=False,
                    shifts=0,
                )

        if not isinstance(model_output, torch.Tensor) or model_output.ndim != 4:
            raise RuntimeError("Demucs returned an invalid source tensor")
        if model_output.shape[0] != 1:
            raise RuntimeError("Demucs returned an unexpected batch count")
        sources = model_output[0]
        if sources.shape[0] != len(source_names):
            raise RuntimeError("Demucs returned an unexpected source count")
        if sources.shape[1] != stereo.shape[0]:
            raise RuntimeError("Demucs returned an unexpected channel count")
        if sources.shape[2] != resampled.shape[1]:
            raise RuntimeError(
                "Demucs stem length unexpectedly changed at model rate: "
                f"expected={resampled.shape[1]}, actual={sources.shape[2]}"
            )
        if not torch.isfinite(sources).all():
            raise RuntimeError("Demucs returned non-finite model output")

        round_trip_tolerance = max(2, math.ceil(sr / model_sample_rate))

        result: dict[str, torch.Tensor] = {}
        for i, name in enumerate(source_names):
            stem = AudioIO.resample(sources[i], model_sample_rate, sr)
            if original_channels == 1:
                stem = AudioIO.to_mono(stem)
            stem = self._match_resample_round_trip_length(
                stem,
                original_len,
                max_difference=round_trip_tolerance,
            )
            if tuple(stem.shape) != tuple(waveform.shape) or not torch.isfinite(stem).all():
                raise RuntimeError(f"Demucs returned invalid {name} stem")
            result[name] = stem
        return result

    @staticmethod
    def _match_resample_round_trip_length(
        waveform: torch.Tensor,
        expected_samples: int,
        *,
        max_difference: int,
    ) -> torch.Tensor:
        actual_samples = int(waveform.shape[-1])
        difference = actual_samples - expected_samples
        if difference == 0:
            return waveform
        if abs(difference) > max_difference:
            raise RuntimeError(
                "Demucs resample round trip changed stem length unexpectedly: "
                f"expected={expected_samples}, actual={actual_samples}"
            )
        if difference > 0:
            return waveform[..., :expected_samples]
        if actual_samples < 1:
            raise RuntimeError("Demucs resample round trip returned an empty stem")
        padding = waveform[..., -1:].expand(*waveform.shape[:-1], -difference)
        return torch.cat((waveform, padding), dim=-1)
