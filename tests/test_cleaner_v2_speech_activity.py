import hashlib
import os
import threading
import time
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from hear.runtime.cleaner.resampling import AudioResampler
from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.speech_activity import CpuSpeechActivity, SpeechActivityPolicy
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode, SourceIdentity


class FakeSession:
    def __init__(self, *args, providers, sess_options):
        assert providers == ["CPUExecutionProvider"]
        assert sess_options.intra_op_num_threads == sess_options.inter_op_num_threads == 1
        self.calls = []
        self.bad = None

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def get_inputs(self):
        return [
            SimpleNamespace(name=k, type=v)
            for k, v in (
                ("input", "tensor(float)"),
                ("state", "tensor(float)"),
                ("sr", "tensor(int64)"),
            )
        ]

    def get_outputs(self):
        return [SimpleNamespace(name="output"), SimpleNamespace(name="stateN")]

    def run(self, names, inputs):
        self.calls.append({key: value.copy() for key, value in inputs.items()})
        probability = (np.max(np.abs(inputs["input"][:, 64:]), axis=1) > 0.05).astype(np.float32)
        probability = (probability * 0.8)[:, None]
        state = inputs["state"] + 1
        if self.bad == "nan":
            state[:] = np.nan
        if self.bad == "probability":
            probability[:] = 2
        if self.bad == "shape":
            state = state[:, :, :-1]
        return probability, state


@pytest.fixture
def setup(tmp_path, monkeypatch):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"test model")
    policy = SpeechActivityPolicy(
        hashlib.sha256(model.read_bytes()).hexdigest(), version("onnxruntime"), version("numpy")
    )
    guard = ResourceGuard(
        ResourceBudget(20_000_000, 10_000_000, 1_000_000),
        tmp_path,
        time.monotonic() + 60,
        threading.Event(),
    )
    monkeypatch.setattr("hear.runtime.cleaner.speech_activity.ort.InferenceSession", FakeSession)
    return model, policy, guard


def fixture_source(tmp_path, samples, rate=16000):
    source = tmp_path / "source.wav"
    sf.write(source, samples, rate, subtype="FLOAT")
    data = source.read_bytes()
    identity = SourceIdentity(
        revision_id="revision",
        media_id="media",
        object_key="source.wav",
        object_version="v1",
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        sample_rate=rate,
        channels=samples.shape[1],
        frames=len(samples),
    )
    return source, identity


def test_channel_states_tail_and_repeat_scan_are_independent(tmp_path, setup):
    model, policy, guard = setup
    scanner = CpuSpeechActivity(model, policy, AudioResampler(CancellableProcessRunner()), guard)
    source, expected = fixture_source(tmp_path, np.full((1027, 2), [0.1, -0.1], dtype=np.float32))
    first = scanner._scan(source, expected, guard)
    second = scanner._scan(source, expected, guard)
    assert first == second
    assert first.active_frames == (1027, 1027)
    assert [(v.channel, v.start_frame, v.end_frame) for v in first.intervals] == [
        (0, 0, 1027),
        (1, 0, 1027),
    ]
    assert first.policy_sha256 == policy.digest
    assert first.source_sha256 == expected.sha256
    for index in (0, 3):
        assert not scanner.session.calls[index]["state"].any()
        assert not scanner.session.calls[index]["input"][:, :64].any()
    assert scanner.session.calls[1]["state"].min() == 1
    assert scanner.session.calls[2]["input"].shape == (2, 576)
    assert not scanner.session.calls[2]["input"][:, 67:].any()


def test_fragmented_speech_evidence_is_bounded_but_totals_remain_complete(tmp_path, setup):
    model, policy, guard = setup
    scanner = CpuSpeechActivity(model, policy, AudioResampler(CancellableProcessRunner()), guard)
    samples = np.zeros((512 * 300, 2), dtype=np.float32)
    for i in range(0, 300, 2):
        samples[i * 512 : (i + 1) * 512] = 0.1
    source, expected = fixture_source(tmp_path, samples)
    report = scanner._scan(source, expected, guard)
    assert len(report.intervals) == 128
    assert report.intervals_truncated
    assert report.active_frames == (512 * 150, 512 * 150)


@pytest.mark.parametrize("bad", ["nan", "shape", "probability"])
def test_invalid_model_outputs_fail_typed(tmp_path, setup, bad):
    model, policy, guard = setup
    scanner = CpuSpeechActivity(model, policy, AudioResampler(CancellableProcessRunner()), guard)
    scanner.session.bad = bad
    source, expected = fixture_source(tmp_path, np.zeros((512, 1), dtype=np.float32))
    with pytest.raises(CleanExecutionError) as error:
        scanner._scan(source, expected, guard)
    assert error.value.code == ErrorCode.INVALID_AUDIO


def test_model_digest_and_runtime_versions_are_enforced(setup):
    model, policy, guard = setup
    for invalid in (replace(policy, model_sha256="0" * 64), replace(policy, numpy_version="0")):
        with pytest.raises(CleanExecutionError) as error:
            CpuSpeechActivity(model, invalid, AudioResampler(CancellableProcessRunner()), guard)
        assert error.value.code == ErrorCode.ENGINE_UNAVAILABLE
    assert replace(policy, threshold=0.6).digest != policy.digest


def test_cancellation_stops_scan(tmp_path, setup):
    model, policy, guard = setup
    scanner = CpuSpeechActivity(model, policy, AudioResampler(CancellableProcessRunner()), guard)
    source, expected = fixture_source(tmp_path, np.zeros((512, 1), dtype=np.float32))
    guard.cancelled.set()
    with pytest.raises(CleanExecutionError) as error:
        scanner._scan(source, expected, guard)
    assert error.value.code == ErrorCode.CANCELLED
    assert not scanner.session.calls


@pytest.mark.parametrize("rate", [16000, 44100])
def test_scan_resamples_without_changing_source_grid(tmp_path, setup, rate):
    model, policy, guard = setup
    scanner = CpuSpeechActivity(model, policy, AudioResampler(CancellableProcessRunner()), guard)
    source, expected = fixture_source(tmp_path, np.full((rate + 7, 2), [0.1, -0.1]), rate)
    report = scanner.scan(source, expected, guard)
    assert report.frames == rate + 7
    assert report.active_frames == (rate + 7, rate + 7)
    assert all(interval.end_frame == rate + 7 for interval in report.intervals)
    assert not list(tmp_path.glob("speech-analysis-*"))


@pytest.mark.skipif(
    not os.environ.get("HEAR_TEST_SILERO_ONNX"), reason="explicit real model path required"
)
def test_real_silero_cpu_streaming_matches_upstream_state_context(tmp_path):
    # This checks adapter parity, not speech accuracy or listening certification.
    import torch
    from silero_vad.utils_vad import OnnxWrapper

    model = Path(os.environ["HEAR_TEST_SILERO_ONNX"])
    policy = SpeechActivityPolicy(
        "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
        version("onnxruntime"),
        version("numpy"),
    )
    guard = ResourceGuard(
        ResourceBudget(20_000_000, 10_000_000, 1_000_000),
        tmp_path,
        time.monotonic() + 60,
        threading.Event(),
    )
    scanner = CpuSpeechActivity(model, policy, AudioResampler(CancellableProcessRunner()), guard)
    samples = np.random.default_rng(42).normal(0, 0.05, (1543, 2)).astype(np.float32)
    source, expected = fixture_source(tmp_path, samples)
    actual = []
    session = scanner.session

    class RecordingSession:
        def run(self, names, inputs):
            result = session.run(names, inputs)
            actual.append(result[0].copy())
            return result

    scanner.session = RecordingSession()
    report = scanner.scan(source, expected, guard)
    upstream = OnnxWrapper(str(model), force_onnx_cpu=True)
    for index, offset in enumerate(range(0, len(samples), 512)):
        data = np.zeros((2, 512), dtype=np.float32)
        chunk = samples[offset : offset + 512]
        data[:, : len(chunk)] = chunk.T
        reference = upstream(torch.from_numpy(data), 16000).numpy()
        np.testing.assert_allclose(actual[index], reference, rtol=0, atol=1e-7)
    assert report.frames == 1543 and report.channels == 2
