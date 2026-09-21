import hashlib
import threading
import time
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf
import torch

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.sam_pipeline import SamSeparationPipeline
from hear.runtime.cleaner.sam_prompt_cache import SamPromptCache, SamPromptIdentity
from hear.services.magic_clean.contracts import CleanExecutionError, CleanPlan, ErrorCode


@pytest.fixture
def binding(tmp_path):
    text = torch.ones(1, 2, 768)
    mask = torch.ones(1, 2, dtype=torch.bool)
    prompt = SamPromptIdentity(
        "a" * 64,
        "b" * 64,
        "c" * 64,
        hashlib.sha256(text.numpy().tobytes()).hexdigest(),
        hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
        2,
    )
    plan = CleanPlan(
        profile="voice_focus",
        profile_version="v1",
        catalogue_sha256="d" * 64,
        runtime=dict(
            engine="sam_audio_small",
            runtime_sha256="e" * 64,
            checkpoint_sha256="f" * 64,
            precision_policy_sha256="1" * 64,
            longform_policy_sha256="2" * 64,
        ),
        attenuation_limit_db=None,
        noise_reduction_db=None,
        noise_reference=None,
        prompt_sha256=prompt.prompt_sha256,
        channel_policy="mono",
        mono_acknowledged=True,
        adjust_loudness=True,
        match_comparison_loudness=True,
        shorten_pauses=False,
        seed=42,
    )
    guard = ResourceGuard(
        ResourceBudget(1000000, 1000000, 10000), tmp_path, time.monotonic() + 30, threading.Event()
    )
    return plan, prompt, SamPromptCache(((prompt, text, mask),)), guard


def test_plan_uses_seed_and_verified_cache_snapshot(binding, monkeypatch):
    plan, prompt, cache, guard = binding
    pipeline = SamSeparationPipeline(None, None)
    seen = []
    source = guard.workspace / "input"
    sf.write(source, np.zeros(3841), 48000, format="WAV", subtype="FLOAT")

    def process(source, destination, **kwargs):
        seen.append(kwargs)
        assert kwargs["seed"] == 42
        assert kwargs["text"].all() and kwargs["text_mask"].all()
        kwargs["text"].zero_()
        return "noise-identity"

    monkeypatch.setattr(pipeline, "separate_file", process)
    for _ in range(2):
        assert (
            pipeline.separate_plan(
                guard.workspace / "input",
                guard.workspace / "output",
                plan=plan,
                expected_runtime=plan.runtime,
                prompt=prompt,
                cache=cache,
                guard=guard,
            )
            == "noise-identity"
        )
    assert len(seen) == 2 and set(guard.workspace.iterdir()) == {source}


def test_preflight_reserves_codec_padding_and_minimum_live_features(binding):
    _, _, _, guard = binding
    guard.budget = replace(guard.budget, max_frames=578000, scratch_bytes=2000000000)
    assert SamSeparationPipeline.preflight(576001, 48000, guard) == 577920 * 1536
    assert SamSeparationPipeline.preflight(288001, 24000, guard) == 577920 * 1536
    guard.budget = replace(guard.budget, max_frames=576001)
    with pytest.raises(CleanExecutionError, match="padding exceeds"):
        SamSeparationPipeline.preflight(576001, 48000, guard)
    guard.budget = replace(guard.budget, max_frames=578000, scratch_bytes=512000000)
    with pytest.raises(CleanExecutionError, match="minimum codec scratch"):
        SamSeparationPipeline.preflight(576001, 48000, guard)
    assert not list(guard.workspace.iterdir())


def test_preflight_counts_existing_scratch(binding):
    _, _, _, guard = binding
    minimum = 5760 * 1536
    guard.budget = replace(guard.budget, scratch_bytes=minimum)
    assert SamSeparationPipeline.preflight(3841, 48000, guard) == minimum
    (guard.workspace / "occupied").write_bytes(b"x")
    with pytest.raises(CleanExecutionError, match="minimum codec scratch"):
        SamSeparationPipeline.preflight(3841, 48000, guard)


@pytest.mark.parametrize("frames, rate", [(0, 48000), (True, 48000), (10, 0), (10, 96001)])
def test_preflight_rejects_invalid_geometry(binding, frames, rate):
    _, _, _, guard = binding
    with pytest.raises(CleanExecutionError) as error:
        SamSeparationPipeline.preflight(frames, rate, guard)
    assert error.value.code == ErrorCode.INVALID_AUDIO


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000])
def test_plan_rate_roundtrip_restores_exact_length(binding, monkeypatch, rate):
    plan, prompt, cache, guard = binding
    source, destination = guard.workspace / "input.wav", guard.workspace / "output.wav"
    frames = rate // 20 + 1
    values = (0.1 * np.sin(np.arange(frames) * 2 * np.pi * 440 / rate)).astype(np.float32)
    sf.write(source, values, rate, subtype="FLOAT")
    original = source.read_bytes()
    pipeline = SamSeparationPipeline(None, None)

    def process(prepared, output, **kwargs):
        with sf.SoundFile(prepared) as audio:
            assert audio.channels == 1 and audio.samplerate == 48000
            assert audio.frames == (frames * 48000 + rate - 1) // rate
            samples = audio.read(dtype="float32")
        sf.write(output, samples, 48000, format="RF64", subtype="FLOAT")
        return "fixture-identity"

    monkeypatch.setattr(pipeline, "separate_file", process)
    assert (
        pipeline.separate_plan(
            source,
            destination,
            plan=plan,
            expected_runtime=plan.runtime,
            prompt=prompt,
            cache=cache,
            guard=guard,
        )
        == "fixture-identity"
    )
    with sf.SoundFile(destination) as output:
        assert (output.samplerate, output.channels, output.frames) == (rate, 1, frames)
        assert np.isfinite(output.read()).all()
    assert source.read_bytes() == original
    assert set(guard.workspace.iterdir()) == {source, destination}


def test_resampled_inference_failure_cleans_intermediates(binding, monkeypatch):
    plan, prompt, cache, guard = binding
    source, destination = guard.workspace / "input.wav", guard.workspace / "output.wav"
    sf.write(source, np.zeros(2206), 44100, subtype="FLOAT")
    pipeline = SamSeparationPipeline(None, None)

    def fail(*args, **kwargs):
        raise RuntimeError("fixture model failure")

    monkeypatch.setattr(pipeline, "separate_file", fail)
    with pytest.raises(RuntimeError, match="fixture model failure"):
        pipeline.separate_plan(
            source,
            destination,
            plan=plan,
            expected_runtime=plan.runtime,
            prompt=prompt,
            cache=cache,
            guard=guard,
        )
    assert set(guard.workspace.iterdir()) == {source}


@pytest.mark.parametrize("mismatch", ["runtime", "prompt", "cache_model", "closed", "channel"])
def test_mismatch_rejected_before_audio_work(binding, monkeypatch, mismatch):
    plan, prompt, cache, guard = binding
    expected = plan.runtime
    if mismatch == "runtime":
        expected = expected.model_copy(update={"runtime_sha256": "3" * 64})
    elif mismatch == "prompt":
        plan = plan.model_copy(update={"prompt_sha256": "4" * 64})
    elif mismatch == "cache_model":
        prompt = replace(prompt, model_sha256="5" * 64)
    elif mismatch == "closed":
        cache.close()
    else:
        plan = plan.model_copy(update={"channel_policy": "validated_dual_mono"})
    pipeline = SamSeparationPipeline(None, None)

    def forbidden(*args, **kwargs):
        pytest.fail("mismatched plan reached audio processing")

    monkeypatch.setattr(pipeline, "separate_file", forbidden)
    with pytest.raises(CleanExecutionError) as error:
        pipeline.separate_plan(
            guard.workspace / "absent",
            guard.workspace / "output",
            plan=plan,
            expected_runtime=expected,
            prompt=prompt,
            cache=cache,
            guard=guard,
        )
    assert error.value.code == ErrorCode.ENGINE_UNAVAILABLE
    assert not list(guard.workspace.iterdir())
