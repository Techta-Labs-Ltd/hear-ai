from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class AudioBuffer:
    data: torch.Tensor
    sample_rate: int

    @property
    def duration(self) -> float:
        return self.data.shape[1] / self.sample_rate

    @property
    def peak(self) -> float:
        return self.data.abs().max().item()

    @property
    def rms(self) -> float:
        return self.data.pow(2).mean().sqrt().item()

    def clone(self) -> "AudioBuffer":
        return AudioBuffer(data=self.data.clone(), sample_rate=self.sample_rate)


@dataclass
class SoundEvent:
    start_sample: int
    end_sample: int
    confidence: float


@dataclass
class ProcessingContext:
    audio: AudioBuffer
    raw: Optional[AudioBuffer] = None
    events: list[SoundEvent] = field(default_factory=list)
    event_mask: Optional[torch.Tensor] = None
    noise_profile: Optional[dict] = None
    stage_times: dict = field(default_factory=dict)
