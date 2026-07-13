import asyncio
import os
import tempfile

import numpy as np
import pytest
import torch
import torchaudio

from app.services.enhancer.pipeline import AudioEnhancer
from app.services.enhancer.enhancer import Enhancer
from app.services.enhancer.config import EngineConfig
from app.services.enhancer.models import ProcessingContext, AudioBuffer
from app.services.enhancer.shaping.eq import DCOffsetRemover, SpeechEQ
from app.services.enhancer.shaping.deesser import DeEsser
from app.services.enhancer.dynamics.compressor import Compressor
from app.services.enhancer.dynamics.normalizer import PeakNormalizer, LoudnessNormalizer
from app.services.enhancer.dynamics.limiter import LookaheadLimiter
from app.services.enhancer.reduction.gate import AdaptiveGate
from app.services.enhancer.reduction.transient import TransientSuppressor
from app.services.enhancer.quality.metrics import QualityMetrics
from app.services.enhancer.base import ProcessingStage
from app.core.storage import storage


def _synth_speech(sr: int, duration_s: float, freq: float = 180.0) -> torch.Tensor:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    sig = 0.3 * np.sin(2 * np.pi * freq * t)
    for h in [2, 3, 4, 5]:
        sig += 0.1 / h * np.sin(2 * np.pi * freq * h * t)
    sig += 0.005 * np.random.randn(len(sig))
    sig /= np.max(np.abs(sig)) + 1e-8
    return torch.from_numpy(sig.astype(np.float32)).unsqueeze(0)


def _save_wav(sig: torch.Tensor, sr: int = 44100) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    torchaudio.save(path, sig, sr)
    return path


def _ctx(sig: torch.Tensor, sr: int = 44100) -> ProcessingContext:
    return ProcessingContext(audio=AudioBuffer(data=sig, sample_rate=sr), raw=AudioBuffer(data=sig.clone(), sample_rate=sr))


def _run(stage: ProcessingStage, ctx: ProcessingContext) -> ProcessingContext:
    stage.load()
    return asyncio.run(stage.process(ctx))


# ── Full Pipeline Integration Tests ────────────────────────────────

class TestAudioEnhancerPipeline:
    def test_enhance_returns_dict_with_all_keys(self, monkeypatch):
        monkeypatch.setattr(storage, "upload_file", lambda *a, **kw: "https://fake-b2/hear-enhanced/test/123.mp3")
        sig = _synth_speech(44100, 2.0)
        path = _save_wav(sig)
        try:
            enhancer = AudioEnhancer(EngineConfig())
            enhancer.load()
            result = asyncio.run(enhancer.enhance(
                input_path=path,
                track_id="test-track",
                job_id="test-job",
                ai_job_id="ai-job-123",
                ai_run_id="run-456",
            ))
            assert isinstance(result, dict)
            expected_keys = {
                "b2_key", "enhanced_url", "local_path",
                "quality_score", "snr_db", "peak_db", "lufs",
                "clipping_detected", "mode_used", "stage_times",
            }
            assert expected_keys.issubset(result.keys()), f"Missing keys: {expected_keys - result.keys()}"
            assert isinstance(result["quality_score"], float)
            assert isinstance(result["snr_db"], float)
            assert isinstance(result["stage_times"], dict)
            assert os.path.isfile(result["local_path"])
        finally:
            if os.path.isfile(path):
                os.unlink(path)

    def test_enhance_without_ai_ids(self, monkeypatch):
        monkeypatch.setattr(storage, "upload_file", lambda *a, **kw: "https://fake-b2/hear-enhanced/test/123.mp3")
        sig = _synth_speech(44100, 1.0)
        path = _save_wav(sig)
        try:
            enhancer = AudioEnhancer(EngineConfig())
            enhancer.load()
            result = asyncio.run(enhancer.enhance(
                input_path=path,
                track_id="test-track",
                job_id="test-job",
            ))
            assert "enhanced_url" in result
            assert result["mode_used"] == "speech"
        finally:
            if os.path.isfile(path):
                os.unlink(path)

    def test_enhance_preserves_audio_content(self, monkeypatch):
        monkeypatch.setattr(storage, "upload_file", lambda *a, **kw: "https://fake-b2/hear-enhanced/test/123.mp3")
        sig = _synth_speech(44100, 1.5)
        path = _save_wav(sig)
        try:
            enhancer = AudioEnhancer(EngineConfig())
            enhancer.load()
            result = asyncio.run(enhancer.enhance(
                input_path=path,
                track_id="test-track",
                job_id="test-job",
            ))
            assert result["quality_score"] > 0.1
            assert result["snr_db"] > -25
        finally:
            if os.path.isfile(path):
                os.unlink(path)

    def test_enhance_empty_audio_fails(self, monkeypatch):
        monkeypatch.setattr(storage, "upload_file", lambda *a, **kw: "https://fake-b2/")
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            torchaudio.save(path, torch.zeros(1, 0), 44100)
            enhancer = AudioEnhancer(EngineConfig())
            enhancer.load()
            with pytest.raises((ValueError, RuntimeError)):
                asyncio.run(enhancer.enhance(
                    input_path=path,
                    track_id="test-track",
                    job_id="test-job",
                ))
        finally:
            if os.path.isfile(path):
                os.unlink(path)


# ── Advanced Enhancer (unwired) Tests ──────────────────────────────

class TestAdvancedEnhancer:
    def test_enhancer_class_returns_enhancement_result(self):
        enhancer = Enhancer(EngineConfig())
        assert enhancer._stages == []
        assert enhancer.is_loaded == False
        enhancer.load()
        assert enhancer.is_loaded == True
        assert len(enhancer._stages) > 0

    def test_enhancer_has_advanced_stages(self):
        enhancer = Enhancer(EngineConfig())
        enhancer.load()
        stage_names = [s.name for s in enhancer._stages]
        advanced = {"bs_roformer", "yamnet", "apply_suppression", "impulse_suppress"}
        assert advanced.issubset(stage_names), f"Missing advanced stages: {advanced - set(stage_names)}"
        assert "vad_consensus" in stage_names
        assert "spectral_reduce" in stage_names
        assert "silence_strip" in stage_names
        assert len(stage_names) >= 19

    def test_enhance_method_called(self, monkeypatch):
        monkeypatch.setattr(storage, "upload_file", lambda *a, **kw: "https://fake-b2/")
        sig = _synth_speech(44100, 1.0)
        path = _save_wav(sig)
        try:
            enhancer = Enhancer(EngineConfig())
            enhancer.load()
            for stage in enhancer._stages:
                if stage.name in ("bs_roformer", "yamnet", "deepfilter",
                                   "clearvoice", "mpsenet", "vad_consensus"):
                    stage._ready = False
            result = asyncio.run(enhancer.enhance(
                input_path=path,
                track_id="test-track",
                job_id="test-job",
            ))
            assert isinstance(result, dict)
            assert "quality_score" in result
            assert "stage_times" in result
            assert "dnsmos_scores" in result
            assert isinstance(result["quality_score"], float)
            assert len(result["dnsmos_scores"]) > 0
        finally:
            if os.path.isfile(path):
                os.unlink(path)


# ── Untested Stage Tests ──────────────────────────────────────────

class TestDCOffsetRemover:
    def test_removes_dc_offset(self):
        sr = 44100
        sig = torch.ones(1, sr) * 0.5
        ctx = _run(DCOffsetRemover(), _ctx(sig, sr))
        mean = ctx.audio.data.mean().item()
        assert abs(mean) < 0.01, f"DC offset remains: mean={mean}"

    def test_preserves_speech_content(self):
        sr = 44100
        sig = _synth_speech(sr, 1.0)
        ctx = _run(DCOffsetRemover(), _ctx(sig, sr))
        diff = (sig - ctx.audio.data).abs().mean().item()
        assert diff < 0.1


class TestPeakNormalizer:
    def test_normalizes_to_ceiling(self):
        sr = 44100
        sig = torch.randn(1, sr) * 0.5
        ctx = _run(PeakNormalizer(EngineConfig()), _ctx(sig, sr))
        ceiling = 10 ** (-1.0 / 20)
        peak = ctx.audio.data.abs().max().item()
        assert peak <= ceiling + 0.01

    def test_quiet_signal_stays_quiet(self):
        sr = 44100
        sig = torch.ones(1, sr) * 0.001
        ctx = _run(PeakNormalizer(EngineConfig()), _ctx(sig, sr))
        peak = ctx.audio.data.abs().max().item()
        assert peak <= 1.0


class TestTransientSuppressor:
    def test_suppresses_impulse(self):
        sr = 44100
        sig = torch.zeros(1, sr)
        sig[0, sr // 2] = 1.0
        ctx = _run(TransientSuppressor(EngineConfig()), _ctx(sig, sr))
        peak = ctx.audio.data.abs().max().item()
        assert peak < 0.5, f"Transient not suppressed: peak={peak}"

    def test_preserves_steady_state(self):
        sr = 44100
        sig = _synth_speech(sr, 1.0, freq=200)
        orig_rms = sig.pow(2).mean().sqrt().item()
        ctx = _run(TransientSuppressor(EngineConfig()), _ctx(sig, sr))
        out_rms = ctx.audio.data.pow(2).mean().sqrt().item()
        assert out_rms > orig_rms * 0.5


class TestCompressorStage:
    def test_reduces_dynamic_range(self):
        sr = 44100
        sig = torch.cat([
            torch.ones(1, sr // 2) * 0.1,
            torch.ones(1, sr // 2) * 1.0,
        ], dim=1)
        ctx = _run(Compressor(EngineConfig()), _ctx(sig, sr))
        segments = ctx.audio.data[:, :sr // 2], ctx.audio.data[:, sr // 2:]
        ratio = segments[1].abs().mean().item() / (segments[0].abs().mean().item() + 1e-8)
        assert ratio < 8.0, f"Dynamic range not reduced: ratio={ratio}"

    def test_no_clipping_output(self):
        sr = 44100
        sig = _synth_speech(sr, 2.0) * 0.9
        ctx = _run(Compressor(EngineConfig()), _ctx(sig, sr))
        peak = ctx.audio.data.abs().max().item()
        assert peak <= 1.0, f"Clipping: peak={peak}"


class TestLoudnessNormalizer:
    def test_normalizes_loudness(self):
        sr = 44100
        sig = _synth_speech(sr, 3.0) * 0.3
        ctx = _run(LoudnessNormalizer(EngineConfig()), _ctx(sig, sr))
        import pyloudnorm as pyln
        meter = pyln.Meter(sr)
        lufs = meter.integrated_loudness(ctx.audio.data.cpu().squeeze(0).numpy().astype(np.float64))
        if np.isfinite(lufs) and -70 < lufs < 0:
            assert abs(lufs - (-16.0)) < 3.0, f"LUFS {lufs} too far from target -16"

    def test_short_audio_fallback(self):
        sr = 44100
        sig = _synth_speech(sr, 0.2)
        ctx = _run(LoudnessNormalizer(EngineConfig()), _ctx(sig, sr))
        assert ctx.audio.data.abs().max().item() <= 1.0


class TestAdaptiveGateNoiseProfile:
    def test_gate_with_noise_profile(self):
        sr = 44100
        sig = torch.cat([
            torch.zeros(1, sr),
            torch.ones(1, sr * 2) * 0.5,
        ], dim=1)
        ctx = _ctx(sig, sr)
        ctx.noise_profile = {"noise_estimate": np.array([0.01])}
        ctx = _run(AdaptiveGate(EngineConfig()), ctx)
        first_half_rms = ctx.audio.data[:, :sr].pow(2).mean().sqrt().item()
        assert first_half_rms < 0.3


class TestMissingStageResilience:
    def test_pipeline_continues_after_stage_failure(self, monkeypatch):
        monkeypatch.setattr(storage, "upload_file", lambda *a, **kw: "https://fake-b2/")
        sig = _synth_speech(44100, 1.0)
        path = _save_wav(sig)
        try:
            enhancer = AudioEnhancer(EngineConfig())
            enhancer.load()
            orig_stages = list(enhancer._stages)
            class BrokenStage(ProcessingStage):
                name = "broken"
                async def process(self, ctx):
                    raise RuntimeError("stage failure")
            bs = BrokenStage()
            bs.load()
            enhancer._stages.insert(1, bs)
            result = asyncio.run(enhancer.enhance(
                input_path=path,
                track_id="test-track",
                job_id="test-job",
            ))
            assert result["mode_used"] == "speech"
        finally:
            if os.path.isfile(path):
                os.unlink(path)

    def test_stage_skip_if_not_ready(self):
        c = EngineConfig()
        eq = SpeechEQ(c)
        assert eq._ready == False
        eq.load()
        assert eq._ready == True
