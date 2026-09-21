import logging
import os
import threading
from pathlib import Path

import numpy as np
import torch
import torchaudio
from clearvoice import ClearVoice

from hear.config import MODEL_ROOT

logger = logging.getLogger(__name__)
MAX_CHUNK_S = 4


class MossFormerOutputError(RuntimeError):
    """MossFormer returned audio that violates its adapter contract."""


class MossFormer2Enhancer:
    SR = 48_000

    def __init__(self):
        self._cv = None
        self._lock = threading.Lock()

    def load(self, model_path: str | Path | None = None):




        model_root = Path(
            model_path
            or os.environ.get(
                "MOSSFORMER_MODEL_PATH", str(MODEL_ROOT / "mossformer2-se-48k")
            )
        )
        if not (model_root / "last_best_checkpoint").is_file():
            raise FileNotFoundError(f"MossFormer2 checkpoint is incomplete: {model_root}")
        runtime_root = Path("/tmp/hear-clearvoice")
        checkpoint_link = runtime_root / "checkpoints" / "MossFormer2_SE_48K"
        checkpoint_link.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_link.is_symlink() and checkpoint_link.resolve() != model_root:
            checkpoint_link.unlink()
        if not checkpoint_link.exists():
            checkpoint_link.symlink_to(model_root, target_is_directory=True)

        prev_cwd = os.getcwd()
        os.chdir(runtime_root)
        try:
            self._cv = ClearVoice(
                task="speech_enhancement",
                model_names=["MossFormer2_SE_48K"],
            )
        finally:
            os.chdir(prev_cwd)

    def _enhance_chunk(self, audio_np: np.ndarray) -> np.ndarray:
        """Run ClearVoice on a single chunk of audio (shape: [1, samples])."""
        output_np = self._cv(audio_np, False)
        if isinstance(output_np, dict):
            if not output_np:
                raise MossFormerOutputError("MossFormer returned an empty output mapping")
            output_np = output_np[next(iter(output_np))]
        if not isinstance(output_np, np.ndarray):
            raise MossFormerOutputError(
                f"MossFormer returned unexpected output type: {type(output_np)}"
            )

        if output_np.ndim == 2 and output_np.shape[0] == 1:
            output_np = output_np[0]
        if output_np.ndim != 1:
            raise MossFormerOutputError(
                f"MossFormer returned invalid shape: {tuple(output_np.shape)}"
            )
        if output_np.shape[0] != audio_np.shape[-1]:
            raise MossFormerOutputError(
                "MossFormer changed chunk length: "
                f"expected={audio_np.shape[-1]}, actual={output_np.shape[0]}"
            )
        if not np.isfinite(output_np).all():
            raise MossFormerOutputError("MossFormer returned non-finite samples")
        return output_np.astype(np.float32, copy=False)

    def _enhance_signal(self, audio: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(audio)))
        norm_factor = 0.9 / peak if peak > 1e-6 else 1.0
        normalized = (audio * norm_factor).astype(np.float32, copy=False)
        sample_count = normalized.shape[0]
        chunk_samples = MAX_CHUNK_S * self.SR
        overlap_samples = int(self.SR * 0.5)

        if sample_count <= chunk_samples:
            return self._enhance_chunk(normalized.reshape(1, -1)) / norm_factor

        step_samples = chunk_samples - overlap_samples
        accumulated = np.zeros(sample_count, dtype=np.float64)
        weight_sum = np.zeros(sample_count, dtype=np.float64)
        start = 0
        while True:
            end = min(start + chunk_samples, sample_count)
            chunk = normalized[start:end]
            enhanced = self._enhance_chunk(chunk.reshape(1, -1))
            weights = np.ones(enhanced.shape[0], dtype=np.float64)
            if start > 0:
                fade_samples = min(overlap_samples, enhanced.shape[0])
                phase = np.linspace(0.0, np.pi / 2.0, fade_samples, dtype=np.float64)
                weights[:fade_samples] *= np.sin(phase) ** 2
            if end < sample_count:
                fade_samples = min(overlap_samples, enhanced.shape[0])
                phase = np.linspace(0.0, np.pi / 2.0, fade_samples, dtype=np.float64)
                weights[-fade_samples:] *= np.cos(phase) ** 2

            accumulated[start:end] += enhanced.astype(np.float64) * weights
            weight_sum[start:end] += weights
            if end == sample_count:
                break
            start += step_samples

        if np.any(weight_sum <= np.finfo(np.float64).eps):
            raise MossFormerOutputError("MossFormer stitching left uncovered samples")
        stitched = accumulated / weight_sum
        if not np.isfinite(stitched).all():
            raise MossFormerOutputError("MossFormer stitching produced non-finite samples")
        return (stitched / norm_factor).astype(np.float32)

    @torch.inference_mode()
    def enhance(self, w: torch.Tensor, sr: int) -> torch.Tensor:
        if sr <= 0:
            raise ValueError("sample rate must be positive")
        if w.ndim != 2 or w.shape[0] < 1 or w.shape[1] < 1:
            raise ValueError("MossFormer expects non-empty [channels, samples] audio")
        if not torch.isfinite(w).all():
            raise ValueError("MossFormer input contains non-finite samples")



        original = w.detach().clone()
        original_shape = tuple(original.shape)
        if self._cv is None:
            logger.warning("MossFormer2 is not loaded; using the immutable input")
            return original

        try:
            working = (
                torchaudio.functional.resample(original, sr, self.SR) if sr != self.SR else original
            )
            channels = working.detach().cpu().numpy().astype(np.float32, copy=True)
            enhanced_channels: list[np.ndarray] = []
            with self._lock:
                for channel in channels:
                    enhanced_channels.append(self._enhance_signal(channel))

            enhanced = torch.from_numpy(np.stack(enhanced_channels)).to(
                device=original.device,
                dtype=original.dtype,
            )
            if sr != self.SR:
                enhanced = torchaudio.functional.resample(
                    enhanced,
                    self.SR,
                    sr,
                )
                enhanced = self._match_resample_round_trip_length(
                    enhanced,
                    original_shape[1],
                )
            if tuple(enhanced.shape) != original_shape:
                raise MossFormerOutputError(
                    "MossFormer adapter changed channel count or sample length: "
                    f"expected={original_shape}, actual={tuple(enhanced.shape)}"
                )
            if not torch.isfinite(enhanced).all():
                raise MossFormerOutputError("MossFormer adapter returned non-finite samples")
            return enhanced
        except MossFormerOutputError:
            raise
        except Exception as exc:
            raise MossFormerOutputError("MossFormer2 enhancement failed") from exc

    @staticmethod
    def _match_resample_round_trip_length(
        waveform: torch.Tensor,
        expected_samples: int,
    ) -> torch.Tensor:
        """Correct only the bounded rounding error of a rate round trip.

        TorchAudio independently rounds the model-rate and source-rate output
        lengths. Their composition can therefore be one sample longer or
        shorter even though the model returned its exact input length. Larger
        differences are not rounding and remain a hard adapter failure.
        """
        actual_samples = int(waveform.shape[-1])
        difference = actual_samples - expected_samples
        if difference == 0:
            return waveform
        if abs(difference) > 2:
            raise MossFormerOutputError(
                "MossFormer resample round trip changed length unexpectedly: "
                f"expected={expected_samples}, actual={actual_samples}"
            )
        if difference > 0:
            return waveform[..., :expected_samples]
        if actual_samples < 1:
            raise MossFormerOutputError("MossFormer resample round trip returned empty audio")
        padding = waveform[..., -1:].expand(*waveform.shape[:-1], -difference)
        return torch.cat((waveform, padding), dim=-1)
