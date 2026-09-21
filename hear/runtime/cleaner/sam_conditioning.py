import importlib
import math

import numpy as np

from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.runtime.cleaner.sam_features import SamFeatureFile
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


class AudioOnlySamForward:
    def __init__(self, core, guard: ResourceGuard, *, max_frames: int = 250):
        if not 1 <= max_frames <= 250:
            raise ValueError("unsupported SAM conditioning window")
        self.core, self.guard, self.max_frames = core, guard, max_frames

    def forward(self, state, mean_features, text_features, text_mask, time):
        torch = importlib.import_module("torch")
        self.guard.check()
        if (
            state.ndim != 3
            or state.shape[0] != 1
            or state.shape[2] != 256
            or not 1 <= state.shape[1] <= self.max_frames
            or mean_features.shape != (1, state.shape[1], 128)
            or text_features.ndim != 3
            or text_features.shape[0] != 1
            or not 1 <= text_features.shape[1] <= 512
            or text_features.shape[2] != 768
            or text_mask.shape != text_features.shape[:2]
            or text_mask.dtype != torch.bool
            or not text_mask.any()
            or time.shape != (1,)
        ):
            raise CleanExecutionError(
                ErrorCode.INVALID_AUDIO, "invalid audio-only SAM conditioning"
            )
        tensors = (state, mean_features, text_features, time)
        if (
            any(
                t.dtype != torch.float32 or t.device != state.device or not torch.isfinite(t).all()
                for t in tensors
            )
            or text_mask.device != state.device
            or not text_features.any()
            or not ((time >= 0) & (time <= 1)).all()
        ):
            raise CleanExecutionError(
                ErrorCode.INVALID_AUDIO, "invalid SAM conditioning precision or values"
            )
        names = (
            "proj",
            "align_masked_video",
            "embed_anchors",
            "timestep_emb",
            "memory_proj",
            "transformer",
        )
        if any(getattr(self.core, name).training for name in names):
            raise ValueError("SAM core must be in evaluation mode")
        with torch.inference_mode(), torch.autocast(state.device.type, enabled=False):
            audio = torch.cat((mean_features, mean_features), dim=2)
            projected = self.core.proj(torch.cat((state, torch.zeros_like(audio), audio), dim=2))
            video = state.new_zeros(1, 1024, state.shape[1])
            aligned = self.core.align_masked_video(projected, video)
            aligned = self.core.embed_anchors(aligned, None, None)
            timestep = self.core.timestep_emb(time, pos=time).unsqueeze(1)
            memory = self.core.memory_proj(text_features) + timestep
            predicted = self.core.transformer(
                aligned, time, padding_mask=None, memory=memory, memory_padding_mask=text_mask
            )
            self.guard.check()
            if (
                predicted.shape != state.shape
                or predicted.dtype != torch.float32
                or not torch.isfinite(predicted).all()
            ):
                raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "invalid SAM vector field")
            return predicted


class SamConditionedField:
    def __init__(self, forward: AudioOnlySamForward, mean: SamFeatureFile, text, text_mask):
        torch = importlib.import_module("torch")
        forward.guard.check()
        if mean.guard is not forward.guard or mean.batch != 1 or mean.channels != 128:
            raise ValueError("incompatible SAM conditioning feature file")
        if (
            text.ndim != 3
            or text.shape[0] != 1
            or text.shape[2] != 768
            or not 1 <= text.shape[1] <= 512
            or text.dtype != torch.float32
            or text_mask.shape != text.shape[:2]
            or text_mask.dtype != torch.bool
            or not torch.isfinite(text).all()
            or not text.any()
            or not text_mask.any()
        ):
            raise ValueError("invalid frozen SAM text conditioning")
        self.forward, self.mean = forward, mean
        self.device = next(forward.core.proj.parameters()).device
        self.text = text.detach().to(self.device).clone()
        self.text_mask = text_mask.detach().to(self.device).clone()
        self.closed = False

    def evaluate(self, state: np.ndarray, *, start_frame: int, time: float) -> np.ndarray:
        self.forward.guard.check()
        if self.closed:
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "SAM conditioning is closed")
        if (
            not isinstance(state, np.ndarray)
            or state.ndim != 2
            or state.shape[1] != 256
            or state.dtype != np.float32
            or not 1 <= len(state) <= self.forward.max_frames
            or start_frame < 0
            or start_frame + len(state) > self.mean.frames
            or not math.isfinite(time)
            or not 0 <= time <= 1
            or not np.isfinite(state).all()
        ):
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "invalid SAM solver window")
        torch = importlib.import_module("torch")
        mean = self.mean.read(start_frame, start_frame + len(state)).transpose(0, 2, 1)
        noisy = torch.from_numpy(np.array(state, copy=True, order="C")).unsqueeze(0).to(self.device)
        features = torch.from_numpy(mean).to(self.device)
        instant = torch.tensor([time], dtype=torch.float32, device=self.device)
        predicted = self.forward.forward(noisy, features, self.text, self.text_mask, instant)
        result = np.array(predicted[0].cpu().numpy(), copy=True, order="C")
        self.forward.guard.check()
        return result

    def close(self) -> None:
        self.text = None
        self.text_mask = None
        self.closed = True
