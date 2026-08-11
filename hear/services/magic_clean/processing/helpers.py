import numpy as np
import torch
from scipy.signal import lfilter


def cosine_fade(n: int, device: torch.device) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, n, device=device)
    return (1.0 - torch.cos(t * torch.pi)) * 0.5


def match_length(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(a.shape[-1], b.shape[-1])
    return a[..., :n], b[..., :n]


def iir_envelope_simple(sig: np.ndarray, attack_coef: float, release_coef: float) -> np.ndarray:
    """Simple two-coefficient envelope for cases where coefficient is constant."""
    b = np.array([1.0 - attack_coef])
    a = np.array([1.0, -attack_coef])
    env_attack = lfilter(b, a, np.abs(sig))

    b = np.array([1.0 - release_coef])
    a = np.array([1.0, -release_coef])
    env_release = lfilter(b, a, np.abs(sig))

    return np.maximum(env_attack, env_release)


def iir_coefs(time_ms: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    coef = np.exp(-1.0 / (sr * time_ms / 1000.0))
    b    = np.array([1.0 - coef])
    a    = np.array([1.0, -coef])
    return b, a
