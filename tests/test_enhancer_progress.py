import asyncio
import os
import tempfile

import numpy as np
import pytest
import torch
import torchaudio

from app.services.enhancer.enhancer import Enhancer
from app.services.enhancer.config import EngineConfig
from app.services.enhancer.models import ProcessingContext, AudioBuffer
from app.services.enhancer.base import ProcessingStage
from app.core.storage import storage


def _synth_speech(sr: int, duration_s: float, freq: float = 180.0) -> torch.Tensor:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    sig = 0.3 * np.sin(2 * np.pi * freq * t)
    for h in [2, 3, 4, 5]:
        sig += 0.1 / h * np.sin(2 * np.pi * freq * h * t)
    sig += 0.005 * np.random.RandomState(42).randn(len(sig))
    sig /= np.max(np.abs(sig)) + 1e-8
    return torch.from_numpy(sig.astype(np.float32)).unsqueeze(0)


def _save_wav(sig: torch.Tensor, sr: int = 44100) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    torchaudio.save(path, sig, sr)
    return path


class PassThroughStage(ProcessingStage):
    def __init__(self, name: str):
        self.name = name

    def load(self):
        self._ready = True

    async def process(self, ctx: ProcessingContext) -> ProcessingContext:
        return ctx


class TestStageProgress:

    def test_on_progress_fires_per_stage(self, monkeypatch):
        monkeypatch.setattr(storage, "upload_file", lambda *a, **kw: "https://fake-b2/test.mp3")
        sr = 48000
        sig = _synth_speech(sr, 5)

        enhancer = Enhancer(EngineConfig())
        stages = [PassThroughStage(f"s{i}") for i in range(4)]
        for s in stages:
            s.load()
        enhancer._stages = stages
        enhancer._dnsmos_stages = set()

        path = _save_wav(sig, sr)
        progress_events = []

        def on_progress(stage_name: str, stage_idx: int, total_stages: int, pct: float):
            progress_events.append((stage_name, stage_idx, total_stages, pct))

        try:
            result = asyncio.run(enhancer.enhance(
                input_path=path, track_id="test", job_id="test-job",
                on_progress=on_progress,
            ))
            assert result["mode_used"] == "speech"

            stage_events = [e for e in progress_events if e[0].startswith("s")]
            assert len(stage_events) == 4, f"Expected 4 stage events, got {len(stage_events)}"

            for i, (name, idx, total, pct) in enumerate(stage_events):
                assert name == f"s{i}", f"Expected stage s{i}, got {name}"
                assert idx == i, f"Expected idx {i}, got {idx}"
                assert total == 4, f"Expected total 4, got {total}"
                expected_pct = ((i + 1) / 4) * 100.0
                assert pct == expected_pct, f"Expected pct {expected_pct}, got {pct}"
        finally:
            if os.path.isfile(path):
                os.unlink(path)

    def test_progress_percentage_monotonic(self, monkeypatch):
        monkeypatch.setattr(storage, "upload_file", lambda *a, **kw: "https://fake-b2/test.mp3")
        sr = 48000
        sig = _synth_speech(sr, 5)

        enhancer = Enhancer(EngineConfig())
        stages = [PassThroughStage(f"s{i}") for i in range(3)]
        for s in stages:
            s.load()
        enhancer._stages = stages
        enhancer._dnsmos_stages = set()

        path = _save_wav(sig, sr)
        pcts = []

        def on_progress(_stage: str, _si: int, _ts: int, pct: float):
            if _stage != "heartbeat":
                pcts.append(pct)

        try:
            asyncio.run(enhancer.enhance(
                input_path=path, track_id="pct-test", job_id="pct-test-job",
                on_progress=on_progress,
            ))
            assert len(pcts) == 3
            assert pcts == [33.3, 66.7, 100.0]
        finally:
            if os.path.isfile(path):
                os.unlink(path)

    def test_stage_times_recorded(self, monkeypatch):
        monkeypatch.setattr(storage, "upload_file", lambda *a, **kw: "https://fake-b2/test.mp3")
        sr = 48000
        sig = _synth_speech(sr, 3)

        enhancer = Enhancer(EngineConfig())
        stages = [PassThroughStage(f"s{i}") for i in range(2)]
        for s in stages:
            s.load()
        enhancer._stages = stages
        enhancer._dnsmos_stages = set()

        path = _save_wav(sig, sr)
        try:
            result = asyncio.run(enhancer.enhance(
                input_path=path, track_id="st-test", job_id="st-test-job",
            ))
            for s in stages:
                assert s.name in result["stage_times"]
                assert result["stage_times"][s.name] >= 0
        finally:
            if os.path.isfile(path):
                os.unlink(path)

    def test_pass_through_preserves_audio(self, monkeypatch):
        monkeypatch.setattr(storage, "upload_file", lambda *a, **kw: "https://fake-b2/test.mp3")
        sr = 48000
        sig = _synth_speech(sr, 4)

        enhancer = Enhancer(EngineConfig())
        pt = PassThroughStage("pass")
        pt.load()
        enhancer._stages = [pt]
        enhancer._dnsmos_stages = set()

        path = _save_wav(sig, sr)
        try:
            result = asyncio.run(enhancer.enhance(
                input_path=path, track_id="eq-test", job_id="eq-test-job",
            ))
            out_wav, out_sr = torchaudio.load(result["local_path"])
            assert out_sr == sr
            min_len = min(sig.shape[1], out_wav.shape[1])
            mse = (sig[0, :min_len] - out_wav[0, :min_len]).pow(2).mean().item()
            assert mse < 0.001, f"MSE too high: {mse}"
        finally:
            if os.path.isfile(path):
                os.unlink(path)

    def test_heartbeat_events_only_with_long_stages(self, monkeypatch):
        monkeypatch.setattr(storage, "upload_file", lambda *a, **kw: "https://fake-b2/test.mp3")
        sr = 48000
        sig = _synth_speech(sr, 3)

        enhancer = Enhancer(EngineConfig())
        pt = PassThroughStage("fast")
        pt.load()
        enhancer._stages = [pt]
        enhancer._dnsmos_stages = set()

        path = _save_wav(sig, sr)
        events = []

        def on_progress(stage_name: str, _si: int, _ts: int, _pct: float):
            events.append(stage_name)

        try:
            asyncio.run(enhancer.enhance(
                input_path=path, track_id="hb-test", job_id="hb-test-job",
                on_progress=on_progress,
            ))
            heartbeats = [s for s in events if s == "heartbeat"]
            assert len(heartbeats) == 0
        finally:
            if os.path.isfile(path):
                os.unlink(path)
