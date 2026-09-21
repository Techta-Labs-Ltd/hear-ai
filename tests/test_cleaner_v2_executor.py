import hashlib
import json
import pickle
import sys
import threading
import time
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
import soundfile as sf

from hear.runtime.cleaner.executor import ExecutionContext, PublishedExecutionError
from hear.runtime.cleaner.factory import CleanerWorkerFactory
from hear.runtime.cleaner.model_registry import CertifiedRuntime
from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.s3_verification import S3SourceStager
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.services.magic_clean.contracts import (
    AttemptTicket,
    CleanExecutionError,
    CleanPlan,
    ErrorCode,
    SpeechRiskEvidence,
)
from hear.services.magic_clean.quality import AudioQualityGate
from tests.test_cleaner_v2_artifacts import MemoryStore
from tests.test_cleaner_v2_contracts import ticket as ticket_fixture
from tests.test_cleaner_v2_source_staging import SourceClient

ticket = ticket_fixture


def test_executor_publishes_speech_evidence_in_manifest_and_report(execution):
    executor, context, _, store = execution

    class RiskAnalyser:
        def evaluate(self, source, processed, expected, guard):
            assert expected == context.ticket.input
            return SpeechRiskEvidence(
                source_sha256=expected.sha256,
                output_sha256=hashlib.sha256(processed.read_bytes()).hexdigest(),
                analysis_sha256="a" * 64,
                comparison_sha256="b" * 64,
                source_active_frames=(1000, 1000),
                output_active_frames=(1000, 1000),
                evidence_truncated=False,
            )

    executor.quality = AudioQualityGate(RiskAnalyser())
    result = executor.execute(context.ticket.plan, context)
    evidence = result.manifest.validation.speech_activity
    assert evidence is not None
    assert evidence.source_sha256 == context.ticket.input.sha256
    artifact = next(v for v in result.manifest.artifacts if v.role == "validation_report")
    report = json.loads(store.objects[artifact.object_key])
    assert report["validation"]["speech_activity"] == evidence.model_dump(mode="json")
    assert result.manifest.validation.wanted_content == "review_required"


def test_restart_required_fault_publishes_no_candidate_and_blocks_next_admission(execution):
    executor, context, engine, store = execution

    def oom(source, destination, plan, guard):
        raise CleanExecutionError(
            ErrorCode.RESOURCE_EXHAUSTED,
            "injected allocation failure",
            worker_restart_required=True,
        )

    engine.process = oom
    with pytest.raises(PublishedExecutionError) as error:
        executor.execute(context.ticket.plan, context)
    assert error.value.code == ErrorCode.RESOURCE_EXHAUSTED
    assert error.value.worker_restart_required
    transported = pickle.loads(pickle.dumps(error.value))
    assert isinstance(transported, PublishedExecutionError)
    assert transported.worker_restart_required
    assert transported.bundle == error.value.bundle
    assert error.value.bundle.manifest.outcome == "failed"
    assert not error.value.bundle.manifest.artifacts
    assert store.calls == [context.ticket.manifest_key]
    assert engine.closed
    with pytest.raises(CleanExecutionError) as next_error:
        executor.execute(context.ticket.plan, context)
    assert next_error.value.code == ErrorCode.ENGINE_UNAVAILABLE
    assert next_error.value.worker_restart_required
    assert store.calls == [context.ticket.manifest_key]


@pytest.mark.parametrize(
    "primary_code",
    [None, ErrorCode.CANCELLED, ErrorCode.DEADLINE_EXCEEDED, ErrorCode.RESOURCE_EXHAUSTED],
)
def test_session_cleanup_failure_retires_worker_and_preserves_primary_code(execution, primary_code):
    executor, context, engine, store = execution
    if primary_code is not None:

        def fail_processing(*args):
            raise CleanExecutionError(primary_code, "injected primary failure")

        engine.process = fail_processing

    def fail_close():
        raise RuntimeError("private-model-path-and-native-diagnostic")

    engine.close = fail_close
    with pytest.raises(CleanExecutionError) as error:
        executor.execute(context.ticket.plan, context)
    assert error.value.code == (primary_code or ErrorCode.PROCESS_FAILED)
    assert error.value.worker_restart_required
    assert "private-model-path" not in str(error.value)
    if primary_code in (ErrorCode.CANCELLED, ErrorCode.DEADLINE_EXCEEDED):
        assert not isinstance(error.value, PublishedExecutionError)
        assert store.calls == []
    else:
        assert isinstance(error.value, PublishedExecutionError)
        assert error.value.bundle.manifest.outcome == "failed"
        assert not error.value.bundle.manifest.artifacts
        assert store.calls == [context.ticket.manifest_key]
    assert "validating" not in context.progress.stages
    timings = context.timings.snapshot()
    assert {"inspection", "loading", "inference", "cleanup"} <= timings.keys()
    assert "validation" not in timings
    assert "mastering" not in timings
    assert ("upload" in timings) == (
        primary_code not in (ErrorCode.CANCELLED, ErrorCode.DEADLINE_EXCEEDED)
    )
    with pytest.raises(CleanExecutionError) as retired:
        executor.worker_lease.assert_owned("gpu")
    assert retired.value.worker_restart_required


class Authorizer:
    def __init__(self):
        self.calls = 0
        self.reject = False

    def verify(self, ticket):
        self.calls += 1
        if self.reject:
            raise CleanExecutionError(ErrorCode.ARTIFACT_CONFLICT, "attempt authorization rejected")


class Progress:
    def __init__(self):
        self.stages = []

    def transition(self, ticket, stage):
        self.stages.append(stage)


class FakeEngine:
    """Test-only engine; never a registered production capability."""

    def __init__(self, identity):
        self.identity = identity
        self.closed = False
        self.fail = False
        self.erase_channel = False

    def open_session(self, plan, guard):
        return self

    def process(self, source, destination, plan, guard):
        if self.fail:
            raise CleanExecutionError(ErrorCode.PROCESS_FAILED, "injected inference failure")
        with (
            sf.SoundFile(source) as audio,
            sf.SoundFile(
                destination,
                "w",
                samplerate=audio.samplerate,
                channels=audio.channels,
                format="WAV",
                subtype="FLOAT",
            ) as output,
        ):
            for block in audio.blocks(blocksize=4096, always_2d=True):
                if self.erase_channel:
                    block[:, 0] = 0
                output.write(block)

    def close(self):
        self.closed = True


@pytest.fixture
def execution(tmp_path, ticket):
    source = tmp_path / "source.wav"
    wave = 0.1 * np.sin(np.arange(48000) * 0.1)
    sf.write(source, np.column_stack((wave, wave)), 48000, subtype="FLOAT")
    data = source.read_bytes()
    ticket["input"].update(sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
    ticket["deadline"] = (datetime.now(UTC) + timedelta(minutes=2)).isoformat()
    parsed = AttemptTicket.model_validate_json(json.dumps(ticket))
    engine = FakeEngine(parsed.plan.runtime)
    certified = CertifiedRuntime(parsed.plan.runtime, "c" * 64, 48000, len(data), (48000,), (2,))
    store = MemoryStore()
    worker = CleanerWorkerFactory.build(
        lock_directory=tmp_path,
        lane="gpu",
        runtimes=(certified,),
        loaders={"deepfilternet3": lambda: engine},
        readiness={"deepfilternet3": lambda identity: identity == engine.identity},
        store=store,
    )
    guard = ResourceGuard(
        ResourceBudget(20_000_000, 5_000_000, 100000),
        tmp_path,
        time.monotonic() + 120,
        threading.Event(),
    )
    context = ExecutionContext(parsed, source, guard, Authorizer(), Progress())
    try:
        yield worker.executor, context, engine, store
    finally:
        worker.close()


def test_full_attempt_runs_real_inspection_mastering_and_manifest(execution):
    executor, context, engine, store = execution
    result = executor.execute(context.ticket.plan, context)
    assert engine.closed
    assert context.authorizer.calls == 2
    assert result.manifest.validation.wanted_content == "review_required"
    assert result.manifest.fence == context.ticket.fence
    assert store.calls[-1] == context.ticket.manifest_key
    report_artifact = next(a for a in result.manifest.artifacts if a.role == "validation_report")
    report = json.loads(store.objects[report_artifact.object_key])
    timings = report["stage_seconds_before_publication"]
    assert set(timings) == {
        "inspection",
        "loading",
        "inference",
        "cleanup",
        "validation",
        "mastering",
    }
    assert all(value >= 0 for value in timings.values())
    assert context.timings.snapshot()["upload"] >= 0
    assert "upload" not in timings
    assert context.progress.stages == [
        "inspecting",
        "processing",
        "validating",
        "mastering",
        "uploading",
    ]
    report = json.loads(store.objects[context.ticket.artifact_prefix + "/validation_report.json"])
    assert report["master"]["frames"] == 48000
    assert "master-0" not in json.dumps(report)


def test_sam_scratch_preflight_precedes_registry_and_model_loading(execution, monkeypatch):
    executor, context, engine, _ = execution
    values = context.ticket.plan.model_dump()
    values.update(
        profile="voice_focus",
        attenuation_limit_db=None,
        prompt_sha256="a" * 64,
        channel_policy="mono",
        mono_acknowledged=True,
    )
    values["runtime"]["engine"] = "sam_audio_small"
    plan = CleanPlan.model_validate(values)

    def forbidden(*args, **kwargs):
        pytest.fail("registry loaded before impossible SAM scratch reservation was rejected")

    monkeypatch.setattr(executor.registry, "load", forbidden)
    with pytest.raises(CleanExecutionError, match="minimum codec scratch") as error:
        executor._execute_verified(plan, context)
    assert error.value.code == ErrorCode.RESOURCE_EXHAUSTED
    assert not engine.closed
    assert context.progress.stages == ["inspecting"]


def test_sample_manifest_and_audio_have_exact_selected_length(execution):
    executor, context, engine, store = execution
    raw = context.ticket.model_dump(mode="json")
    raw.update(purpose="sample_preview", sample={"start_frame": 1000, "end_frame": 22001})
    ticket = AttemptTicket.model_validate_json(json.dumps(raw))
    context = ExecutionContext(
        ticket, context.source, context.guard, context.authorizer, context.progress
    )
    result = executor.execute(ticket.plan, context)
    assert result.manifest.purpose == "sample_preview"
    assert sf.info(context.guard.workspace / "sample.wav").format == "RF64"
    report = json.loads(store.objects[ticket.artifact_prefix + "/validation_report.json"])
    assert report["master"]["frames"] == 21001
    assert "can_apply" not in result.manifest.model_dump()


def test_engine_failure_closes_session_and_publishes_only_failure(execution):
    executor, context, engine, store = execution
    engine.fail = True
    with pytest.raises(PublishedExecutionError) as error:
        executor.execute(context.ticket.plan, context)
    assert engine.closed
    assert list(store.objects) == [context.ticket.manifest_key]
    assert error.value.code == ErrorCode.PROCESS_FAILED
    assert error.value.bundle.manifest.outcome == "failed"
    assert error.value.bundle.manifest.artifacts == ()


def test_lost_channel_is_rejected_before_mastering(execution):
    executor, context, engine, store = execution
    engine.erase_channel = True
    with pytest.raises(CleanExecutionError) as error:
        executor.execute(context.ticket.plan, context)
    assert error.value.code == ErrorCode.INVALID_AUDIO
    assert engine.closed
    assert list(store.objects) == [context.ticket.manifest_key]
    assert json.loads(store.objects[context.ticket.manifest_key])["outcome"] == "failed"
    assert "mastering" not in context.progress.stages


def test_authorization_rejection_precedes_processing(execution):
    executor, context, engine, store = execution
    context.authorizer.reject = True
    with pytest.raises(CleanExecutionError):
        executor.execute(context.ticket.plan, context)
    assert not context.progress.stages
    assert not store.objects


def test_revoked_authorization_prevents_upload_after_inference(execution):
    executor, context, engine, store = execution

    class RevokedAuthorizer(Authorizer):
        def verify(self, ticket):
            self.reject = self.calls > 0
            super().verify(ticket)

    context = ExecutionContext(
        context.ticket, context.source, context.guard, RevokedAuthorizer(), context.progress
    )
    with pytest.raises(CleanExecutionError):
        executor.execute(context.ticket.plan, context)
    assert engine.closed
    assert not store.objects


def test_cancellation_after_inference_closes_session_without_upload(execution):
    executor, context, engine, store = execution
    original_process = engine.process

    def cancel_after_process(*args):
        original_process(*args)
        context.guard.cancelled.set()

    engine.process = cancel_after_process
    with pytest.raises(CleanExecutionError) as error:
        executor.execute(context.ticket.plan, context)
    assert error.value.code == ErrorCode.CANCELLED
    assert engine.closed
    assert not store.objects


def test_failure_marker_storage_error_preserves_processing_error(execution):
    executor, context, engine, store = execution
    engine.fail = True
    store.fail_at = 1
    with pytest.raises(CleanExecutionError) as error:
        executor.execute(context.ticket.plan, context)
    assert error.value.code == ErrorCode.PROCESS_FAILED
    assert not isinstance(error.value, PublishedExecutionError)
    assert not store.objects
    assert engine.closed


def test_failure_reauthorizes_before_terminal_marker(execution):
    executor, context, engine, store = execution
    engine.fail = True

    class RevokedAuthorizer(Authorizer):
        def verify(self, ticket):
            self.reject = self.calls > 0
            super().verify(ticket)

    context = ExecutionContext(
        context.ticket, context.source, context.guard, RevokedAuthorizer(), context.progress
    )
    with pytest.raises(CleanExecutionError) as error:
        executor.execute(context.ticket.plan, context)
    assert error.value.code == ErrorCode.PROCESS_FAILED
    assert not isinstance(error.value, PublishedExecutionError)
    assert not store.objects
    assert context.authorizer.calls == 2


def test_upload_uncertainty_does_not_attempt_failure_marker(execution):
    executor, context, engine, store = execution
    store.fail_at = 4
    with pytest.raises(CleanExecutionError) as error:
        executor.execute(context.ticket.plan, context)
    assert error.value.code == ErrorCode.STORAGE_FAILED
    assert len(store.calls) == 4
    assert context.ticket.manifest_key not in store.objects
    assert not isinstance(error.value, PublishedExecutionError)


def test_missing_worker_ownership_prevents_execution(execution):
    executor, context, engine, store = execution
    executor.worker_lease.close()
    with pytest.raises(CleanExecutionError) as error:
        executor.execute(context.ticket.plan, context)
    assert error.value.code == ErrorCode.RESOURCE_EXHAUSTED
    assert not engine.closed
    assert not context.progress.stages
    assert not store.objects


def test_authenticated_deadline_bounds_child_work_not_just_stage_transitions(
    execution, monkeypatch
):
    executor, context, engine, store = execution
    # Isolate this test's model-child deadline from the separately supervised
    # decoder startup; inspection cancellation has its own regression tests.
    inspected = executor.inspector.inspect(context.source, context.ticket.input, context.guard)
    monkeypatch.setattr(executor.inspector, "inspect", lambda *args: inspected)
    raw = context.ticket.model_dump(mode="json")
    raw["deadline"] = (datetime.now(UTC) + timedelta(seconds=0.3)).isoformat()
    short_ticket = AttemptTicket.model_validate_json(json.dumps(raw))
    context = ExecutionContext(
        short_ticket, context.source, context.guard, context.authorizer, context.progress
    )
    initial_worker_deadline = context.guard.deadline

    def slow_process(source, destination, plan, guard):
        CancellableProcessRunner().run([sys.executable, "-c", "import time; time.sleep(30)"], guard)

    engine.process = slow_process
    started = time.monotonic()
    with pytest.raises(CleanExecutionError) as error:
        executor.execute(short_ticket.plan, context)
    assert error.value.code == ErrorCode.DEADLINE_EXCEEDED
    assert time.monotonic() - started < 2
    assert context.guard.deadline < initial_worker_deadline
    assert engine.closed
    assert not store.objects


def test_busy_worker_rejects_before_inspection_or_failure_publication(execution):
    executor, context, engine, store = execution
    with executor.worker_lease.attempt("gpu"):
        with pytest.raises(CleanExecutionError) as error:
            executor.execute(context.ticket.plan, context)
    assert error.value.code == ErrorCode.RESOURCE_EXHAUSTED
    assert not isinstance(error.value, PublishedExecutionError)
    assert not context.progress.stages
    assert not store.objects
    assert not engine.closed


def test_remote_source_runs_through_inspection_mastering_and_publication(execution):
    executor, context, engine, store = execution
    data = context.source.read_bytes()
    context.source.unlink()
    client = SourceClient(context.ticket, data, context.source)
    result = executor.execute(
        context.ticket.plan, context, stager=S3SourceStager(client, "scoped-bucket")
    )
    assert context.progress.stages[0:2] == ["downloading", "inspecting"]
    assert context.source.read_bytes() == data
    assert engine.closed
    assert result.manifest.outcome == "succeeded"
    assert client.calls[0]["VersionId"] == context.ticket.input.object_version
    assert store.calls[-1] == context.ticket.manifest_key


def test_unauthorized_attempt_never_downloads_remote_source(execution):
    executor, context, _, _ = execution
    context.authorizer.reject = True
    client = SourceClient(context.ticket, b"unused", context.source)
    with pytest.raises(CleanExecutionError):
        executor.execute(context.ticket.plan, context, stager=S3SourceStager(client, "bucket"))
    assert not client.calls
