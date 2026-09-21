"""Contiguous bounded LSTM execution for SAM's retained codec watermark path.

This is not the complete streaming decoder. Convolution halos, sample alignment
and watermark-message state must be handled separately by the codec adapter.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode

if TYPE_CHECKING:
    import torch


class SamLSTMStream:
    def __init__(self, layer: torch.nn.LSTM, *, max_frames: int, batch_size: int):
        if (
            layer.training
            or layer.bidirectional
            or layer.batch_first
            or layer.proj_size
            or layer.dropout
            or layer.input_size != layer.hidden_size
            or not 0 < max_frames <= 480000
            or batch_size not in (1, 2)
        ):
            raise ValueError("unsupported SAM recurrent inference configuration")
        self.layer = layer
        self.max_frames = max_frames
        self.batch_size = batch_size
        self.next_frame = 0
        self._state = None
        self.closed = False

    def process(self, features: torch.Tensor, start_frame: int, guard: ResourceGuard):
        torch = importlib.import_module("torch")

        guard.check()
        if self.closed:
            raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "recurrent stream is closed")
        if start_frame != self.next_frame:
            raise CleanExecutionError(ErrorCode.ARTIFACT_CONFLICT, "noncontiguous recurrent input")
        if (
            features.ndim != 3
            or features.shape[:2] != (self.batch_size, self.layer.input_size)
            or not 0 < features.shape[2] <= self.max_frames
            or features.dtype != torch.float32
            or not torch.isfinite(features).all()
        ):
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "invalid recurrent features")
        try:
            with torch.inference_mode(), torch.autocast(features.device.type, enabled=False):
                sequence = features.permute(2, 0, 1)
                output, state = self.layer(sequence, self._state)
                # Upstream LSTMBlock has skip=True; preserve the residual exactly.
                output = (output + sequence).permute(1, 2, 0)
                if not torch.isfinite(output).all() or any(
                    not torch.isfinite(value).all() for value in state
                ):
                    raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "invalid recurrent output")
            guard.check()
        except BaseException:
            # A failed/cancelled native call cannot leave resumable partial state.
            self.close()
            raise
        self._state = state
        self.next_frame += features.shape[2]
        return output

    def close(self) -> None:
        self._state = None
        self.closed = True
