import json
import threading
import time

import numpy as np
import pytest
import soundfile as sf

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.services.magic_clean.contracts import AttemptTicket, CleanExecutionError, ErrorCode
from hear.services.magic_clean.engines.deepfilter import ContextualPolicy, DeepFilterEngine
from tests.test_cleaner_v2_contracts import ticket as ticket_fixture

ticket = ticket_fixture


class Backend:
    def __init__(self):
        self.lengths = []
        self.closed = False
        self.corrupt = False

    def enhance(self, samples, attenuation_limit_db):
        assert attenuation_limit_db == 18
        self.lengths.append(samples.shape[1])
        result = samples.copy()
        # Simulate boundary transients to verify contextual margins are cropped.
        if len(self.lengths) > 1:
            result[:, :10] = 0
        if self.corrupt:
            result[:, -1] = np.nan
        return result

    def close(self):
        self.closed = True


class Factory:
    def __init__(self, backend):
        self.backend = backend

    def open(self, guard):
        return self.backend

    def validate_identity(self, identity):
        assert identity.engine == "deepfilternet3"


@pytest.fixture
def runtime(tmp_path, ticket):
    policy = ContextualPolicy(48000, 4800)
    ticket["plan"]["runtime"]["longform_policy_sha256"] = policy.digest
    plan = AttemptTicket.model_validate_json(json.dumps(ticket)).plan
    backend = Backend()
    engine = DeepFilterEngine(plan.runtime, Factory(backend), policy)
    guard = ResourceGuard(
        ResourceBudget(20_000_000, 5_000_000, 300000),
        tmp_path,
        time.monotonic() + 20,
        threading.Event(),
    )
    return engine, backend, plan, guard


@pytest.mark.parametrize("frames", [1, 47999, 48000, 48001, 100003])
def test_contextual_blocks_keep_exact_stereo_samples_and_tail(tmp_path, runtime, frames):
    engine, backend, plan, guard = runtime
    rng = np.random.default_rng(32)
    wave = rng.uniform(-0.3, 0.3, (frames, 2)).astype(np.float32)
    source, destination = tmp_path / "source.wav", tmp_path / "output.wav"
    sf.write(source, wave, 48000, subtype="FLOAT")
    session = engine.open_session(plan, guard)
    try:
        session.process(source, destination, plan, guard)
    finally:
        session.close()
    output, _ = sf.read(destination, dtype="float32", always_2d=True)
    assert sf.info(destination).format == "RF64"
    np.testing.assert_array_equal(output, wave)
    assert max(backend.lengths) <= 57600
    assert backend.closed


def test_invalid_model_output_removes_partial_file_and_releases_session(tmp_path, runtime):
    engine, backend, plan, guard = runtime
    backend.corrupt = True
    source, destination = tmp_path / "source.wav", tmp_path / "output.wav"
    sf.write(source, np.ones((48001, 2)) * 0.1, 48000, subtype="FLOAT")
    session = engine.open_session(plan, guard)
    try:
        with pytest.raises(CleanExecutionError):
            session.process(source, destination, plan, guard)
    finally:
        session.close()
    assert not destination.exists()
    assert backend.closed
    engine.open_session(plan, guard).close()


def test_concurrent_sessions_rejected(runtime):
    engine, _, plan, guard = runtime
    session = engine.open_session(plan, guard)
    try:
        with pytest.raises(CleanExecutionError):
            engine.open_session(plan, guard)
    finally:
        session.close()


def test_context_change_changes_policy_digest():
    assert ContextualPolicy(48000, 4800).digest != ContextualPolicy(48000, 9600).digest


@pytest.mark.parametrize("rate", [8000, 44100, 96000])
def test_non_model_rate_preserves_source_timing_and_cleans_scratch(tmp_path, runtime, rate):
    engine, backend, plan, guard = runtime
    source, destination = tmp_path / "source.wav", tmp_path / "output.wav"
    frames = rate + 17
    wave = np.zeros((frames, 2), dtype=np.float32)
    wave[rate // 2, 0] = 0.5
    sf.write(source, wave, rate, subtype="FLOAT")
    session = engine.open_session(plan, guard)
    try:
        session.process(source, destination, plan, guard)
    finally:
        session.close()
    output, actual_rate = sf.read(destination, dtype="float32", always_2d=True)
    assert actual_rate == rate
    assert output.shape == wave.shape
    assert abs(np.argmax(np.abs(output[:, 0])) - rate // 2) <= 1
    assert np.max(np.abs(output[:, 1])) == 0
    assert max(backend.lengths) <= 57600
    assert not list(tmp_path.glob("df3-resample-*"))


def test_resampled_model_failure_cleans_temporary_pcm(tmp_path, runtime):
    engine, backend, plan, guard = runtime
    backend.corrupt = True
    source, destination = tmp_path / "source.wav", tmp_path / "output.wav"
    sf.write(source, np.full((44117, 2), 0.1), 44100, subtype="FLOAT")
    session = engine.open_session(plan, guard)
    try:
        with pytest.raises(CleanExecutionError):
            session.process(source, destination, plan, guard)
    finally:
        session.close()
    assert source.exists()
    assert not destination.exists()
    assert not list(tmp_path.glob("df3-resample-*"))
    assert backend.closed


@pytest.mark.parametrize("rate", [48000, 44100])
@pytest.mark.parametrize("fail", [False, True])
def test_racing_output_preserved_through_session_cleanup(
    tmp_path, runtime, monkeypatch, rate, fail
):
    engine, backend, plan, guard = runtime
    source, destination = tmp_path / "source.wav", tmp_path / "output.wav"
    sf.write(source, np.zeros(4801), rate, subtype="FLOAT")
    original = backend.enhance

    def enhance(samples, attenuation):
        destination.write_bytes(b"concurrent owner")
        if fail:
            raise RuntimeError("fixture model failure")
        return original(samples, attenuation)

    monkeypatch.setattr(backend, "enhance", enhance)
    session = engine.open_session(plan, guard)
    try:
        with pytest.raises(CleanExecutionError) as error:
            session.process(source, destination, plan, guard)
        assert error.value.code == (
            ErrorCode.PROCESS_FAILED if fail else ErrorCode.ARTIFACT_CONFLICT
        )
    finally:
        session.close()
    assert destination.read_bytes() == b"concurrent owner"
    assert set(tmp_path.iterdir()) == {source, destination}
    assert backend.closed
