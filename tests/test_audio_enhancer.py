import asyncio

import numpy as np
import pytest
import torch

from app.services.enhancer.models import ProcessingContext, AudioBuffer
from app.services.enhancer.config import EngineConfig
from app.services.enhancer.shaping.eq import SpeechEQ
from app.services.enhancer.shaping.deesser import DeEsser
from app.services.enhancer.dynamics.limiter import LookaheadLimiter
from app.services.enhancer.dynamics.compressor import Compressor
from app.services.enhancer.dynamics.normalizer import LoudnessNormalizer
from app.services.enhancer.reduction.gate import AdaptiveGate
from app.services.enhancer.quality.metrics import QualityMetrics
from app.services.enhancer.base import ProcessingStage


def _synth_speech(sr: int, duration_s: float, freq: float = 180.0) -> torch.Tensor:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    sig = 0.3 * np.sin(2 * np.pi * freq * t)
    for h in [2, 3, 4, 5]:
        sig += 0.1 / h * np.sin(2 * np.pi * freq * h * t)
    sig += 0.005 * np.random.randn(len(sig))
    sig /= np.max(np.abs(sig)) + 1e-8
    return torch.from_numpy(sig.astype(np.float32)).unsqueeze(0)


def _ctx(sig: torch.Tensor, sr: int = 44100) -> ProcessingContext:
    return ProcessingContext(audio=AudioBuffer(data=sig, sample_rate=sr))


def _run(stage: ProcessingStage, ctx: ProcessingContext) -> ProcessingContext:
    return asyncio.run(stage.process(ctx))


# ── Adaptive gate tests ──────────────────────────────────────────


class TestAdaptiveGate:
    def test_gate_preserves_loud_signal(self):
        sr = 44100
        sig = torch.cat([
            torch.zeros(1, sr // 2),
            torch.ones(1, sr) * 0.5,
        ], dim=1)
        ctx = _run(AdaptiveGate(EngineConfig()), _ctx(sig, sr))
        assert ctx.audio.data.abs().mean().item() > 0.3


# ── Speech EQ tests ──────────────────────────────────────────────


class TestSpeechEQ:
    def test_no_14k_lpf(self):
        sr = 44100
        t = np.linspace(0, 1, sr, endpoint=False)
        high_tone = np.sin(2 * np.pi * 15000 * t).astype(np.float32) * 0.5
        sig = torch.from_numpy(high_tone).unsqueeze(0)
        ctx = _run(SpeechEQ(EngineConfig()), _ctx(sig, sr))
        high_energy = ctx.audio.data.abs().mean().item()
        assert high_energy > 0.01, "14kHz content was killed — LPF too low"

    def test_bass_treble_gentle(self):
        sr = 44100
        sig = _synth_speech(sr, 2.0)
        ctx = _run(SpeechEQ(EngineConfig()), _ctx(sig, sr))
        input_rms = sig.pow(2).mean().sqrt().item()
        output_rms = ctx.audio.data.pow(2).mean().sqrt().item()
        ratio = output_rms / (input_rms + 1e-8)
        assert 0.5 < ratio < 2.0, f"EQ changed volume too much: ratio={ratio}"


# ── De-esser tests ───────────────────────────────────────────────


class TestDeEsser:
    def test_reduces_sibilance(self):
        sr = 44100
        t = np.linspace(0, 1, sr, endpoint=False)
        sibilant = np.sin(2 * np.pi * 7000 * t).astype(np.float32) * 0.5
        sig = torch.from_numpy(sibilant).unsqueeze(0)
        ctx = _run(DeEsser(EngineConfig()), _ctx(sig, sr))
        input_energy = sig.abs().mean().item()
        output_energy = ctx.audio.data.abs().mean().item()
        assert output_energy < input_energy, "De-esser did not reduce sibilance"
        assert output_energy > input_energy * 0.3, "De-esser killed too much"

    def test_preserves_normal_speech(self):
        sr = 44100
        sig = _synth_speech(sr, 2.0, freq=180)
        ctx = _run(DeEsser(EngineConfig()), _ctx(sig, sr))
        diff = (sig - ctx.audio.data).abs().mean().item()
        assert diff < 0.05, "De-esser affected normal speech too much"


# ── Lookahead limiter tests ──────────────────────────────────────


class TestLookaheadLimiter:
    def test_output_below_ceiling(self):
        sr = 44100
        sig = torch.ones(1, sr) * 0.8
        sig[0, sr // 2 : sr // 2 + 100] = 1.5
        ctx = _run(LookaheadLimiter(EngineConfig()), _ctx(sig, sr))
        peak = ctx.audio.data.abs().max().item()
        ceiling = 10 ** (-1.0 / 20)
        assert peak <= ceiling + 0.01, f"Peak {peak} exceeds ceiling {ceiling}"

    def test_preserves_quiet_signal(self):
        sr = 44100
        sig = torch.ones(1, sr) * 0.1
        ctx = _run(LookaheadLimiter(EngineConfig()), _ctx(sig, sr))
        assert ctx.audio.data.abs().max().item() <= sig.abs().max().item() + 0.01


# ── Output quality tests ─────────────────────────────────────────


class TestOutputQuality:
    def test_no_clipping_after_compression(self):
        sr = 44100
        sig = _synth_speech(sr, 3.0) * 0.9
        ctx = _run(Compressor(EngineConfig()), _ctx(sig, sr))
        peak = ctx.audio.data.abs().max().item()
        assert peak <= 1.0, f"Clipping detected: peak={peak}"

    def test_lufs_normalization_target(self):
        sr = 44100
        sig = _synth_speech(sr, 5.0) * 0.3
        ctx = _run(LoudnessNormalizer(EngineConfig()), _ctx(sig, sr))
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        lufs = meter.integrated_loudness(ctx.audio.data.cpu().squeeze(0).numpy().astype(np.float64))
        if np.isfinite(lufs):
            assert abs(lufs - (-16.0)) < 3.0, f"LUFS {lufs} too far from target -16"


# ── Quality metrics tests ────────────────────────────────────────


class TestQualityMetrics:
    def test_snr_positive(self):
        sr = 44100
        sig = _synth_speech(sr, 2.0)
        clean = AudioBuffer(data=sig, sample_rate=sr)
        noisy = AudioBuffer(data=sig + 0.01 * torch.randn_like(sig), sample_rate=sr)
        snr = QualityMetrics.compute_snr(clean, noisy)
        assert snr > 0
