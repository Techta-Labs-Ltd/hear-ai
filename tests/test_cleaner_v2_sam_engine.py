from types import SimpleNamespace

import pytest
import torch

from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode
from hear.services.magic_clean.engines.sam_audio import SamEngine
from tests.test_cleaner_v2_sam_plan import binding as binding_fixture

binding = binding_fixture


class Factory:
    def __init__(self, backend):
        self.backend = backend
        self.validations = 0

    def validate_identity(self, identity):
        self.validations += 1
        assert identity.engine == "sam_audio_small"

    def open(self, guard):
        return self.backend


@pytest.mark.parametrize("fail", [False, True])
def test_one_shot_session_binds_plan_closes_and_releases(binding, fail):
    plan, prompt, cache, guard = binding
    calls, closed = [], []

    def process(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["plan"] is plan and kwargs["guard"] is guard
        assert kwargs["expected_runtime"] == plan.runtime
        assert kwargs["prompt"] is prompt and kwargs["cache"] is cache
        if fail:
            raise RuntimeError("fixture inference failure")

    backend = SimpleNamespace(
        pipeline=SimpleNamespace(separate_plan=process),
        prompt=prompt,
        cache=cache,
        close=lambda: closed.append(True),
    )
    factory = Factory(backend)
    engine = SamEngine(plan.runtime, factory)
    session = engine.open_session(plan, guard)
    with pytest.raises(CleanExecutionError) as error:
        engine.open_session(plan, guard)
    assert error.value.code == ErrorCode.RESOURCE_EXHAUSTED
    source, output = guard.workspace / "source", guard.workspace / "output"
    try:
        if fail:
            with pytest.raises(RuntimeError):
                session.process(source, output, plan, guard)
        else:
            session.process(source, output, plan, guard)
        with pytest.raises(CleanExecutionError):
            session.process(source, output, plan, guard)
        assert len(calls) == 1
    finally:
        session.close()
        session.close()
    assert len(closed) == 1
    if fail:
        with pytest.raises(CleanExecutionError) as error:
            engine.open_session(plan, guard)
        assert error.value.worker_restart_required
        assert factory.validations == 3
    else:
        engine.open_session(plan, guard).close()
        assert factory.validations == 4


def test_close_failure_quarantines_engine(binding):
    plan, prompt, cache, guard = binding

    def fail():
        raise RuntimeError("fixture unload failure")

    backend = SimpleNamespace(prompt=prompt, cache=cache, close=fail)
    engine = SamEngine(plan.runtime, Factory(backend))
    session = engine.open_session(plan, guard)
    with pytest.raises(RuntimeError):
        session.close()
    with pytest.raises(CleanExecutionError) as error:
        engine.open_session(plan, guard)
    assert error.value.code == ErrorCode.ENGINE_UNAVAILABLE


def test_rejected_prompt_closes_loaded_backend_and_releases_lease(binding):
    plan, prompt, cache, guard = binding
    closed = []
    backend = SimpleNamespace(prompt=prompt, cache=cache, close=lambda: closed.append(True))
    engine = SamEngine(plan.runtime, Factory(backend))
    wrong = plan.model_copy(update={"prompt_sha256": "9" * 64})
    with pytest.raises(CleanExecutionError):
        engine.open_session(wrong, guard)
    assert closed == [True]
    engine.open_session(plan, guard).close()


def test_active_inference_cannot_be_closed(binding):
    plan, prompt, cache, guard = binding
    closed = []
    backend = SimpleNamespace(prompt=prompt, cache=cache, close=lambda: closed.append(True))
    engine = SamEngine(plan.runtime, Factory(backend))
    session = engine.open_session(plan, guard)

    def process(*args, **kwargs):
        with pytest.raises(CleanExecutionError) as error:
            session.close()
        assert error.value.code == ErrorCode.RESOURCE_EXHAUSTED
        assert not closed

    backend.pipeline = SimpleNamespace(separate_plan=process)
    session.process(guard.workspace / "source", guard.workspace / "output", plan, guard)
    session.close()
    assert closed == [True]


def test_runtime_drift_after_open_is_rejected_before_pipeline(binding, monkeypatch):
    plan, prompt, cache, guard = binding
    called = []
    backend = SimpleNamespace(
        prompt=prompt,
        cache=cache,
        close=lambda: None,
        pipeline=SimpleNamespace(separate_plan=lambda *args, **kwargs: called.append(True)),
    )
    factory = Factory(backend)
    engine = SamEngine(plan.runtime, factory)
    session = engine.open_session(plan, guard)

    # The bound validator checks its factory's state again, not a cached readiness result.
    def reject(identity):
        raise CleanExecutionError(ErrorCode.ENGINE_UNAVAILABLE, "runtime changed")

    monkeypatch.setattr(session, "_validate_runtime", reject)
    try:
        with pytest.raises(CleanExecutionError, match="runtime changed"):
            session.process(guard.workspace / "absent", guard.workspace / "output", plan, guard)
        assert not called and not list(guard.workspace.iterdir())
    finally:
        session.close()


@pytest.mark.parametrize("stage", ["load", "process"])
@pytest.mark.parametrize("kind", ["memory", "cuda_oom", "native", "typed_restart", "cancel"])
def test_native_faults_are_typed_and_quarantined(binding, monkeypatch, stage, kind):
    plan, prompt, cache, guard = binding
    closed = []

    def fail(*args, **kwargs):
        if kind == "memory":
            raise MemoryError("private native details")
        if kind == "cuda_oom":
            raise torch.cuda.OutOfMemoryError("private native details")
        if kind == "native":
            raise RuntimeError("private native details")
        raise CleanExecutionError(
            ErrorCode.CANCELLED if kind == "cancel" else ErrorCode.ENGINE_UNAVAILABLE,
            "typed failure",
            worker_restart_required=kind == "typed_restart",
        )

    backend = SimpleNamespace(
        prompt=prompt,
        cache=cache,
        pipeline=SimpleNamespace(separate_plan=fail),
        close=lambda: closed.append(True),
    )
    factory = Factory(backend)
    engine = SamEngine(plan.runtime, factory)
    if stage == "load":
        monkeypatch.setattr(factory, "open", fail)
        with pytest.raises(CleanExecutionError) as error:
            engine.open_session(plan, guard)
        monkeypatch.setattr(factory, "open", lambda guard: backend)
        assert not closed
    else:
        session = engine.open_session(plan, guard)
        try:
            with pytest.raises(CleanExecutionError) as error:
                session.process(guard.workspace / "source", guard.workspace / "out", plan, guard)
        finally:
            session.close()
        assert closed == [True]
    expected = (
        ErrorCode.RESOURCE_EXHAUSTED
        if kind in ("memory", "cuda_oom")
        else ErrorCode.CANCELLED
        if kind == "cancel"
        else ErrorCode.ENGINE_UNAVAILABLE
    )
    assert error.value.code == expected
    assert error.value.worker_restart_required == (kind != "cancel")
    assert "private native details" not in str(error.value)
    assert error.value.__context__ is None
    assert not engine._lease.locked()
    if kind == "cancel":
        engine.open_session(plan, guard).close()
    else:
        with pytest.raises(CleanExecutionError) as refused:
            engine.open_session(plan, guard)
        assert refused.value.worker_restart_required
