import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from hear.contracts.jobs import AttemptEnvelope
from hear.execution.native import NativeExecutor
from hear.runtime.cleaner.resource_guard import ResourceBudget
from hear.services.magic_clean.artifacts import PublishedBundle, StoredObject
from hear.services.magic_clean.contracts import (
    CleanExecutionError,
    CleanResultManifest,
    ErrorCode,
)
from hear.workflows.available_magic_clean import AvailableMagicCleanWorkflow
from hear.workflows.magic_clean import MagicCleanWorkflow


class Executor:
    def execute(self, plan, context, *, stager=None, artifacts=None):
        context.progress.transition(context.ticket, "inspecting")
        context.progress.transition(context.ticket, "processing")
        manifest = CleanResultManifest(
            contract_version="hear.cleaner.result.v2",
            backend_id=context.ticket.backend_id,
            tenant_scope=context.ticket.tenant_scope,
            job_id=context.ticket.job_id,
            attempt_id=context.ticket.attempt_id,
            fence=context.ticket.fence,
            purpose=context.ticket.purpose,
            source=context.ticket.input,
            expected_active_audio_revision=context.ticket.expected_active_audio_revision,
            plan=context.ticket.plan,
            plan_sha256="b" * 64,
            sample=None,
            deadline=context.ticket.deadline,
            completed_at=datetime.now(UTC),
            outcome="succeeded",
            error_code=None,
            validation={
                "hard_integrity": "passed",
                "wanted_content": "passed",
                "warning_codes": (),
            },
            artifacts=(
                {
                    "role": "cleaned_master",
                    "object_key": f"{context.ticket.artifact_prefix}/cleaned_master.flac",
                    "object_version": "v1",
                    "sha256": "c" * 64,
                    "size_bytes": 100,
                    "content_type": "audio/flac",
                },
                {
                    "role": "delivery_audio",
                    "object_key": f"{context.ticket.artifact_prefix}/delivery_audio.mp3",
                    "object_version": "v2",
                    "sha256": "d" * 64,
                    "size_bytes": 80,
                    "content_type": "audio/mpeg",
                },
                {
                    "role": "validation_report",
                    "object_key": f"{context.ticket.artifact_prefix}/validation_report.json",
                    "object_version": "v3",
                    "sha256": "e" * 64,
                    "size_bytes": 50,
                    "content_type": "application/json",
                },
            ),
            timing="identity",
            correlation_id=context.ticket.correlation_id,
        )
        return PublishedBundle(
            manifest,
            StoredObject(context.ticket.manifest_key, "mv1", "f" * 64, 200),
        )


def envelope():
    deadline = datetime.now(UTC) + timedelta(minutes=5)
    prefix = "tenant/jobs/backend-a/job-1/attempt-1"
    ticket = {
        "contract_version": "hear.cleaner.v2",
        "backend_id": "backend-a",
        "tenant_scope": "tenant",
        "job_id": "job-1",
        "attempt_id": "attempt-1",
        "fence": 1,
        "provider": "pod",
        "purpose": "full_candidate",
        "input": {
            "revision_id": "revision-1",
            "media_id": "media-1",
            "object_key": "tenant/source.flac",
            "object_version": "source-v1",
            "sha256": "a" * 64,
            "size_bytes": 100,
            "sample_rate": 48000,
            "channels": 2,
            "frames": 48000,
        },
        "expected_active_audio_revision": "revision-1",
        "plan": {
            "profile": "natural",
            "profile_version": "v1",
            "catalogue_sha256": "a" * 64,
            "runtime": {
                "engine": "deepfilternet3",
                "runtime_sha256": "a" * 64,
                "checkpoint_sha256": "a" * 64,
                "precision_policy_sha256": "a" * 64,
                "longform_policy_sha256": "a" * 64,
            },
            "attenuation_limit_db": 18,
            "prompt_sha256": None,
            "channel_policy": "preserve",
            "mono_acknowledged": False,
            "adjust_loudness": True,
            "match_comparison_loudness": True,
            "shorten_pauses": False,
            "seed": 42,
        },
        "sample": None,
        "artifact_prefix": prefix,
        "manifest_key": f"{prefix}/manifest.json",
        "deadline": deadline.isoformat(),
        "heartbeat_seconds": 10,
        "lease_seconds": 60,
        "correlation_id": "corr-1",
    }
    return AttemptEnvelope.model_validate(
        {
            "job_id": "job-1",
            "run_id": "run-1",
            "attempt_id": "attempt-1",
            "job_type": "magic_clean",
            "track_id": "track-1",
            "user_id": "user-1",
            "source": {
                "url": "https://media.example/source.flac",
                "revision": 1,
                "file_sha256": "a" * 64,
            },
            "storage": {
                "endpoint_url": "https://s3.example",
                "bucket_name": "bucket",
                "key_id": "key",
                "application_key": "secret",
                "folder_prefix": "tenant/jobs/",
                "public_base_url": "https://cdn.example",
                "expires_at": (deadline + timedelta(minutes=5)).isoformat(),
            },
            "options": {"profile": "natural", "cleaner_ticket": ticket},
            "artifact_prefix": prefix,
            "deadline": deadline.isoformat(),
            "reporting_grant": "grant",
            "backend_base_url": "https://api.example/api/v1",
        }
    )


def test_magic_clean_workflow_maps_cleaner_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "hear.workflows.magic_clean.B2ImmutableArtifactStore",
        lambda storage: SimpleNamespace(client=object(), bucket_name="bucket"),
    )
    monkeypatch.setattr(
        "hear.workflows.magic_clean.S3SourceStager",
        lambda client, bucket: object(),
    )
    native = NativeExecutor("magic-clean-test")
    workflow = MagicCleanWorkflow(
        SimpleNamespace(executor=Executor()),
        native,
        workspace_root=tmp_path,
        resource_budget=ResourceBudget(
            10_000_000,
            1_000_000,
            1_000_000,
        ),
    )

    async def run():
        result = [item async for item in workflow.stream(envelope())]
        await native.close()
        return result

    events = asyncio.run(run())
    outcome = events[-1].data["outcome"]
    assert outcome["status"] == "completed"
    assert outcome["result"]["requires_approval"] is True
    assert outcome["result"]["profile"] == "natural"
    assert outcome["result"]["delivery_audio"]["content_type"] == "audio/mpeg"
    assert [item.event.value for item in events] == [
        "started",
        "stage",
        "stage",
        "outcome",
    ]


def test_available_magic_clean_returns_processing_failure_outcome(tmp_path):
    class Audio:
        async def download_source(self, url, workspace):
            source = workspace.file("source.wav")
            source.write_bytes(b"audio")
            return source

    class Cleaner:
        profile = "natural"

        def clean(self, *args, **kwargs):
            raise CleanExecutionError(
                ErrorCode.PROCESS_FAILED,
                "DeepFilterNet processing failed",
            )

    native = NativeExecutor("available-magic-clean-test")
    workflow = AvailableMagicCleanWorkflow(
        Audio(),
        SimpleNamespace(),
        native,
        workspace_root=tmp_path,
        timeout_seconds=30,
        model_cleaner=Cleaner(),
    )

    async def run():
        request = envelope()
        request = request.model_copy(update={"source": request.source.model_copy(
            update={"file_sha256": hashlib.sha256(b"audio").hexdigest()}
        )})
        result = [item async for item in workflow.stream(request)]
        await native.close()
        return result

    events = asyncio.run(run())
    outcome = events[-1].data["outcome"]
    assert outcome["status"] == "failed"
    assert outcome["error_code"] == "process_failed"
    assert outcome["artifacts"] == []
    assert [item.event.value for item in events] == ["started", "stage", "outcome"]
