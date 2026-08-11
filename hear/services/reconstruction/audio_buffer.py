from dataclasses import dataclass

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
