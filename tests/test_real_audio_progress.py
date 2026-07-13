import asyncio
import os

import pytest
import torch
import torchaudio

from app.services.enhancer.enhancer import Enhancer
from app.services.enhancer.config import EngineConfig
from app.services.enhancer.base import ProcessingStage
from app.services.enhancer.models import ProcessingContext
from app.services.enhancer.shaping.eq import DCOffsetRemover, SpeechEQ
from app.services.enhancer.shaping.deesser import DeEsser
from app.services.enhancer.dynamics.compressor import Compressor
from app.services.enhancer.dynamics.limiter import LookaheadLimiter
from app.core.storage import storage

AUDIO_PATH = os.path.join(os.path.dirname(__file__), "..", "HRA Elects New Leaders to Challenge Council.wav")


class PassThroughStage(ProcessingStage):
    def __init__(self, name: str):
        self.name = name

    def load(self):
        self._ready = True

    async def process(self, ctx: ProcessingContext) -> ProcessingContext:
        return ctx


@pytest.fixture
def audio_path():
    path = os.path.abspath(AUDIO_PATH)
    assert os.path.isfile(path), f"Audio file not found: {path}"
    return path


def test_file_exists_and_is_valid(audio_path):
    wav, sr = torchaudio.load(audio_path)
    assert wav.shape[0] == 1
    assert sr == 44100
    assert wav.shape[1] == 4285647
    dur = wav.shape[1] / sr
    assert 95 < dur < 100, f"Expected ~97s, got {dur:.1f}s"


class TestRealAudioProgress:

    def test_stage_progress_fires_for_each_stage(self, audio_path, monkeypatch):
        monkeypatch.setattr(storage, "upload_file", lambda *a, **kw: "https://fake-b2/test.mp3")

        enhancer = Enhancer(EngineConfig())
        stages = [PassThroughStage(f"s{i}") for i in range(5)]
        for s in stages:
            s.load()
        enhancer._stages = stages
        enhancer._dnsmos_stages = set()

        progress_log = []

        def on_progress(stage_name: str, stage_idx: int, total_stages: int, pct: float):
            progress_log.append((stage_name, stage_idx, total_stages, pct))

        result = asyncio.run(enhancer.enhance(
            input_path=audio_path,
            track_id="progress-test",
            job_id="progress-test-job",
            on_progress=on_progress,
        ))

        assert result["mode_used"] == "speech"

        stage_events = [e for e in progress_log if e[0].startswith("s")]
        assert len(stage_events) == 5
        for i, (name, idx, total, pct) in enumerate(stage_events):
            assert name == f"s{i}", f"Expected s{i}, got {name}"
            assert idx == i
            assert total == 5

        pcts = [e[3] for e in progress_log if e[0] != "heartbeat"]
        for i in range(1, len(pcts)):
            assert pcts[i] >= pcts[i - 1], f"Progress regressed: {pcts[i-1]} -> {pcts[i]}"
        assert pcts[-1] == 100.0

    def test_pass_through_preserves_audio(self, audio_path, monkeypatch):
        monkeypatch.setattr(storage, "upload_file", lambda *a, **kw: "https://fake-b2/test.mp3")

        enhancer = Enhancer(EngineConfig())
        pt = PassThroughStage("pass")
        pt.load()
        enhancer._stages = [pt]
        enhancer._dnsmos_stages = set()

        progress_log = []

        def on_progress(stage_name: str, stage_idx: int, total_stages: int, pct: float):
            progress_log.append((stage_name, stage_idx, total_stages, pct))

        result = asyncio.run(enhancer.enhance(
            input_path=audio_path,
            track_id="pass-test",
            job_id="pass-test-job",
            on_progress=on_progress,
        ))

        out_path = result["local_path"]
        assert os.path.isfile(out_path)
        out_wav, out_sr = torchaudio.load(out_path)
        assert out_sr == 48000

        in_wav, in_sr = torchaudio.load(audio_path)
        if in_sr != out_sr:
            in_wav = torchaudio.functional.resample(in_wav, in_sr, out_sr)
        min_len = min(in_wav.shape[1], out_wav.shape[1])
        mse = (in_wav[0, :min_len] - out_wav[0, :min_len]).pow(2).mean().item()
        assert mse < 0.001, f"Pass-through changed audio: MSE={mse}"

        assert len(progress_log) >= 1

    def test_real_dsp_stages_no_artifacts(self, audio_path, monkeypatch):
        monkeypatch.setattr(storage, "upload_file", lambda *a, **kw: "https://fake-b2/test.mp3")

        enhancer = Enhancer(EngineConfig())
        stages = [
            DCOffsetRemover(),
            SpeechEQ(EngineConfig()),
            DeEsser(EngineConfig()),
            Compressor(EngineConfig()),
            LookaheadLimiter(EngineConfig()),
        ]
        for s in stages:
            s.load()
        enhancer._stages = stages
        enhancer._dnsmos_stages = set()

        progress_log = []

        def on_progress(stage_name: str, stage_idx: int, total_stages: int, pct: float):
            progress_log.append((stage_name, stage_idx, total_stages, pct))

        result = asyncio.run(enhancer.enhance(
            input_path=audio_path,
            track_id="dsp-test",
            job_id="dsp-test-job",
            on_progress=on_progress,
        ))

        out_wav, out_sr = torchaudio.load(result["local_path"])
        assert out_sr == 48000
        assert out_wav.shape[1] > 0

        in_wav, in_sr = torchaudio.load(audio_path)
        in_dur = in_wav.shape[1] / in_sr
        out_dur = out_wav.shape[1] / out_sr
        dur_diff = abs(out_dur - in_dur)
        assert dur_diff < 0.5, f"Duration changed: in={in_dur:.2f}s out={out_dur:.2f}s"

        assert out_wav.abs().max().item() > 0.01, "Output is silent"

        stage_events = [e for e in progress_log if e[0] != "heartbeat"]
        assert len(stage_events) == len(stages)

    def test_dnsmos_scores_present(self, audio_path, monkeypatch):
        monkeypatch.setattr(storage, "upload_file", lambda *a, **kw: "https://fake-b2/test.mp3")

        enhancer = Enhancer(EngineConfig())
        pt = PassThroughStage("dummy")
        pt.load()
        enhancer._stages = [pt]
        enhancer._dnsmos_stages = set()

        result = asyncio.run(enhancer.enhance(
            input_path=audio_path,
            track_id="dnsmos-test",
            job_id="dnsmos-test-job",
        ))

        assert "dnsmos_scores" in result
        final_entry = [d for d in result["dnsmos_scores"] if d["stage"] == "final"]
        assert len(final_entry) == 1
        assert final_entry[0]["before"] is None
        assert isinstance(final_entry[0]["after"], float)

    def test_stage_times_accumulated(self, audio_path, monkeypatch):
        monkeypatch.setattr(storage, "upload_file", lambda *a, **kw: "https://fake-b2/test.mp3")

        enhancer = Enhancer(EngineConfig())
        stages = [PassThroughStage(f"s{n}") for n in range(2)]
        for s in stages:
            s.load()
        enhancer._stages = stages
        enhancer._dnsmos_stages = set()

        result = asyncio.run(enhancer.enhance(
            input_path=audio_path,
            track_id="st-test",
            job_id="st-test-job",
        ))

        for s in stages:
            assert s.name in result["stage_times"]
            assert result["stage_times"][s.name] >= 0
