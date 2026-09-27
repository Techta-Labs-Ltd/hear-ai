import hashlib
import importlib.metadata
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.sam_loader import PinnedSamAssets, PinnedSamFactory
from hear.runtime.cleaner.sam_official import SamOfficialPipeline
from hear.services.magic_clean.contracts import (
    CleanExecutionError,
    CleanPlan,
    ErrorCode,
    RuntimeIdentity,
)
from hear.services.magic_clean.engines.sam_audio import SamEngine


def _guard(path):
    return ResourceGuard(
        ResourceBudget(128_000_000, 32_000_000, 100_000),
        path,
        time.monotonic() + 30,
        threading.Event(),
    )


def test_factory_uses_official_sam_audio_api_and_drops_video_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(PinnedSamAssets, "verify", lambda self, guard: None)
    config = tmp_path / "sam" / "config.json"
    config.parent.mkdir()
    config.write_text("{}")
    checkpoint = config.with_name("checkpoint.pt")
    checkpoint.write_bytes(b"checkpoint")
    text = tmp_path / "t5"
    text.mkdir()
    ranker = tmp_path / "clap.pt"
    span = tmp_path / "pe"
    cache = tmp_path / "cache"
    span.mkdir()
    cache.mkdir()
    calls = []

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.vision_encoder = SimpleNamespace(dim=13)

    class FakeSAMAudio:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("model", model_id, kwargs))
            return FakeModel()

    class FakeProcessor:
        audio_sampling_rate = 48_000

        @classmethod
        def from_pretrained(cls, model_id):
            calls.append(("processor", model_id))
            return cls()

    monkeypatch.setitem(
        __import__("sys").modules,
        "sam_audio",
        SimpleNamespace(SAMAudio=FakeSAMAudio, SAMAudioProcessor=FakeProcessor),
    )
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: PinnedSamFactory.PACKAGES[name],
    )
    factory = PinnedSamFactory(
        PinnedSamAssets(
            config,
            checkpoint,
            text,
            tuple(
                (name, "a" * 64)
                for name in (
                    "config.json",
                    "tokenizer.json",
                    "spiece.model",
                    "model.safetensors",
                )
            ),
            ranker,
            "b" * 64,
            span,
            tuple(
                (name, "c" * 64)
                for name in (
                    "config.json",
                    "model.safetensors",
                    "preprocessor_config.json",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                )
            ),
            cache,
        ),
        device="cpu",
        text_encoder_identity="a" * 64,
    )
    backend = factory.open(_guard(tmp_path))

    assert calls[0][0] == "model"
    assert calls[0][1] == str(config.parent)
    assert calls[0][2]["local_files_only"] is True
    assert calls[0][2]["text_encoder"] == {"name": str(text)}
    assert calls[0][2]["visual_ranker"] is None
    assert calls[0][2]["text_ranker"] == {
        "kind": "clap",
        "checkpoint": str(ranker),
    }
    assert calls[0][2]["span_predictor"] == str(span)
    assert calls[1] == ("processor", str(config.parent))
    assert not hasattr(backend.pipeline.model, "vision_encoder")
    audio_features = torch.ones(2, 5, 9)
    video_features = backend.pipeline.model._get_video_features(None, audio_features)
    assert video_features.shape == (2, 13, 5)
    assert torch.count_nonzero(video_features) == 0
    backend.close()
    factory.close()


def test_official_pipeline_overlap_keeps_every_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(SamOfficialPipeline, "CHUNK_SECONDS", 4)
    monkeypatch.setattr(SamOfficialPipeline, "OVERLAP_SECONDS", 1)

    class FakeBatch:
        def to(self, device):
            return self

    class FakeProcessor:
        audio_sampling_rate = 48_000
        audio_hop_length = 997

        def __call__(self, *, audios, descriptions):
            assert descriptions == ["background noise"]
            self.waveform = audios[0]
            return FakeBatch()

    class FakeModel:
        device = torch.device("cpu")

        def parameters(self):
            return iter((torch.nn.Parameter(torch.ones(())),))

        def __init__(self, processor):
            self.processor = processor
            self.calls = 0

        def separate(self, batch, *, predict_spans, reranking_candidates):
            assert predict_spans is False
            assert reranking_candidates == 2
            self.calls += 1
            wave = self.processor.waveform[0]
            padded = (
                (wave.shape[-1] + self.processor.audio_hop_length - 1)
                // self.processor.audio_hop_length
            ) * self.processor.audio_hop_length
            wave = torch.nn.functional.pad(wave, (0, padded - wave.shape[-1]))
            return SimpleNamespace(
                target=[wave + self.calls],
                residual=[wave - self.calls],
            )

    sample_rate = 48_000
    values = np.linspace(-0.1, 0.1, sample_rate * 7, dtype=np.float32)
    source = tmp_path / "source.wav"
    destination = tmp_path / "separated.wav"
    sf.write(source, values, sample_rate, subtype="FLOAT")
    processor = FakeProcessor()
    model = FakeModel(processor)
    pipeline = SamOfficialPipeline(model, processor)
    pipeline._separate_48k(
        source,
        destination,
        description="background noise",
        stream="residual",
        seed=5,
        predict_spans=False,
        guard=_guard(tmp_path),
    )

    output, rate = sf.read(destination, dtype="float32")
    assert rate == sample_rate
    assert output.shape == values.shape
    assert model.calls == 3
    assert np.isfinite(output).all()


def test_sam_residual_rejects_contiguous_active_audio_collapse():
    source = np.full(48_000 * 4, 0.25, dtype=np.float32)
    residual = source.copy()
    residual[48_000 : 48_000 * 3] = 0

    with pytest.raises(CleanExecutionError, match="residual collapsed") as error:
        SamOfficialPipeline._validate_residual(source, residual)

    assert error.value.code == ErrorCode.INVALID_AUDIO


def test_sam_residual_accepts_bounded_attenuation():
    source = np.full(48_000 * 4, 0.25, dtype=np.float32)
    residual = source * 0.25

    SamOfficialPipeline._validate_residual(source, residual)


def test_sam_target_rejects_inaudible_output():
    source = np.full(48_000 * 3, 0.1, dtype=np.float32)
    target = np.full_like(source, 1e-4)

    assert not SamOfficialPipeline._target_detected(source, target)


def test_sam_target_accepts_audible_event():
    source = np.full(48_000 * 3, 0.1, dtype=np.float32)
    target = np.zeros_like(source)
    target[48_000:96_000] = 0.01

    assert SamOfficialPipeline._target_detected(source, target)


def test_sam_target_rejects_generated_sound_on_silent_source():
    source = np.zeros(48_000, dtype=np.float32)
    target = np.full_like(source, 0.01)

    assert not SamOfficialPipeline._target_detected(source, target)


def test_sam_engine_passes_user_prompt_and_releases_session(tmp_path):
    identity = RuntimeIdentity(
        engine="sam_audio_base",
        runtime_sha256="1" * 64,
        checkpoint_sha256="2" * 64,
        precision_policy_sha256="3" * 64,
        longform_policy_sha256="4" * 64,
    )
    description = "background noise"
    plan = CleanPlan(
        profile="sam_audio",
        profile_version="v1",
        catalogue_sha256="5" * 64,
        runtime=identity,
        attenuation_limit_db=None,
        prompt_sha256=hashlib.sha256(description.encode()).hexdigest(),
        prompt_text=description,
        channel_policy="mono",
        mono_acknowledged=True,
        adjust_loudness=False,
        match_comparison_loudness=True,
        shorten_pauses=False,
        seed=7,
    )
    guard = _guard(tmp_path)
    calls, closed = [], []

    class Backend:
        pipeline = SimpleNamespace(separate_plan=lambda *args, **kwargs: calls.append(kwargs))

        def close(self):
            closed.append(True)

    class Factory:
        def validate_identity(self, supplied):
            assert supplied == identity

        def open(self, _guard):
            assert _guard is guard
            return Backend()

        def close(self):
            pass

    engine = SamEngine(identity, Factory())
    session = engine.open_session(plan, guard)
    assert plan.prompt_action == "remove"
    with pytest.raises(CleanExecutionError) as occupied:
        engine.open_session(plan, guard)
    assert occupied.value.code == ErrorCode.RESOURCE_EXHAUSTED
    session.process(tmp_path / "source.wav", tmp_path / "target.wav", plan, guard)
    assert calls[0]["plan"].prompt_text == description
    assert calls[0]["plan"].prompt_action == "remove"
    assert calls[0]["expected_runtime"] == identity
    session.close()
    assert closed == [True]
    engine.close()
