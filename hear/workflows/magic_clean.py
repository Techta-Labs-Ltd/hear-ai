from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from hear.audio.workspace import AudioWorkspace
from hear.contracts.events import ExecutionEvent, ExecutionEventType
from hear.contracts.jobs import AttemptEnvelope
from hear.contracts.outcomes import ArtifactManifest, ExecutionOutcome
from hear.execution.native import NativeExecutor
from hear.runtime.cleaner.executor import ExecutionContext, PublishedExecutionError
from hear.runtime.cleaner.factory import CleanerWorker
from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.s3_verification import S3SourceStager
from hear.services.magic_clean.artifacts import ArtifactWriter, PublishedBundle
from hear.services.magic_clean.contracts import AttemptTicket
from hear.storage.magic_clean import B2ImmutableArtifactStore


class AttemptAuthorizer:
    def __init__(self, ticket: AttemptTicket) -> None:
        self._ticket = ticket

    def verify(self, ticket: AttemptTicket) -> None:
        if ticket != self._ticket:
            raise RuntimeError("cleaner_attempt_mismatch")
        if datetime.now(UTC) >= ticket.deadline:
            raise RuntimeError("cleaner_attempt_expired")


class ProgressBridge:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=16)

    def transition(self, ticket: AttemptTicket, stage: str) -> None:
        self._loop.call_soon_threadsafe(self._put, stage)

    def _put(self, stage: str) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(stage)

    async def next(self) -> str:
        return await self._queue.get()


class MagicCleanWorkflow:
    STAGE_PROGRESS = {
        "downloading": 5.0,
        "inspecting": 12.0,
        "processing": 30.0,
        "validating": 70.0,
        "mastering": 82.0,
        "uploading": 92.0,
    }

    def __init__(
        self,
        worker: CleanerWorker,
        native: NativeExecutor,
        *,
        workspace_root: Path,
        resource_budget: ResourceBudget,
    ) -> None:
        self._worker = worker
        self._native = native
        self._workspace_root = workspace_root
        self._resource_budget = resource_budget

    async def stream(self, envelope: AttemptEnvelope):
        raw_ticket = envelope.options.get("cleaner_ticket")
        if not isinstance(raw_ticket, dict):
            raise ValueError("cleaner_ticket_required")
        ticket = AttemptTicket.model_validate_json(json.dumps(raw_ticket))
        self._validate_ticket(envelope, ticket)
        workspace = AudioWorkspace(
            self._workspace_root,
            envelope.job_id,
            envelope.attempt_id,
        )
        cancelled = threading.Event()
        loop = asyncio.get_running_loop()
        progress = ProgressBridge(loop)
        guard = ResourceGuard(
            self._resource_budget,
            workspace.path,
            time.monotonic() + max(1.0, (ticket.deadline - datetime.now(UTC)).total_seconds()),
            cancelled,
        )
        source = workspace.file("source.audio")
        store = B2ImmutableArtifactStore(envelope.storage)
        stager = S3SourceStager(store.client, store.bucket_name)
        context = ExecutionContext(
            ticket,
            source,
            guard,
            AttemptAuthorizer(ticket),
            progress,
        )
        writer = ArtifactWriter(store)
        task = asyncio.create_task(
            self._native.run(
                self._worker.executor.execute,
                ticket.plan,
                context,
                stager=stager,
                artifacts=writer,
            )
        )
        sequence = 1
        yield self._event(envelope, sequence, "accepted", 0.0, ExecutionEventType.STARTED)
        sequence += 1
        try:
            while not task.done():
                try:
                    stage = await asyncio.wait_for(progress.next(), timeout=1.0)
                except TimeoutError:
                    continue
                yield self._event(
                    envelope,
                    sequence,
                    stage,
                    self.STAGE_PROGRESS.get(stage, 50.0),
                    ExecutionEventType.STAGE,
                )
                sequence += 1
            bundle = await task
            outcome = self._outcome(envelope, bundle)
            yield self._event(
                envelope,
                sequence,
                "completed",
                100.0,
                ExecutionEventType.OUTCOME,
                {"outcome": outcome.model_dump(mode="json")},
            )
        except asyncio.CancelledError:
            cancelled.set()
            if not task.done():
                task.cancel()
            raise
        except PublishedExecutionError as exc:
            outcome = self._failed_outcome(envelope, exc)
            yield self._event(
                envelope,
                sequence,
                "failed",
                100.0,
                ExecutionEventType.OUTCOME,
                {"outcome": outcome.model_dump(mode="json")},
            )
        finally:
            if not task.done():
                cancelled.set()
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            workspace.cleanup()

    @staticmethod
    def _validate_ticket(envelope: AttemptEnvelope, ticket: AttemptTicket) -> None:
        if ticket.job_id != envelope.job_id or ticket.attempt_id != envelope.attempt_id:
            raise ValueError("cleaner_ticket_identity_mismatch")
        if ticket.deadline != envelope.deadline:
            raise ValueError("cleaner_ticket_deadline_mismatch")
        if not ticket.artifact_prefix.startswith(envelope.storage.folder_prefix):
            raise ValueError("cleaner_ticket_storage_mismatch")
        if envelope.source.file_sha256 and ticket.input.sha256 != envelope.source.file_sha256:
            raise ValueError("cleaner_ticket_source_mismatch")
        profile = str(envelope.options.get("profile") or "")
        if ticket.plan.profile != profile:
            raise ValueError("cleaner_ticket_profile_mismatch")

    @staticmethod
    def _outcome(envelope: AttemptEnvelope, bundle: PublishedBundle) -> ExecutionOutcome:
        artifacts = tuple(
            ArtifactManifest(
                bucket_name=envelope.storage.bucket_name,
                object_key=item.object_key,
                size_bytes=item.size_bytes,
                sha256=item.sha256,
                content_type=item.content_type,
                audio_url=MagicCleanWorkflow._url(envelope, item.object_key),
            )
            for item in bundle.manifest.artifacts
        )
        delivery = next(
            (item for item in artifacts if item.content_type == "audio/mpeg"),
            None,
        )
        return ExecutionOutcome(
            job_id=envelope.job_id,
            attempt_id=envelope.attempt_id,
            track_id=envelope.track_id,
            job_type=envelope.job_type,
            source_revision=envelope.source.revision,
            status="completed",
            artifacts=artifacts,
            result={
                "requires_approval": True,
                "profile": bundle.manifest.plan.profile,
                "manifest": bundle.manifest.model_dump(mode="json"),
                "manifest_reference": {
                    "object_key": bundle.reference.key,
                    "object_version": bundle.reference.version,
                    "sha256": bundle.reference.sha256,
                    "size_bytes": bundle.reference.size_bytes,
                },
                "delivery_audio": delivery.model_dump(mode="json") if delivery else None,
            },
        )

    @staticmethod
    def _failed_outcome(
        envelope: AttemptEnvelope,
        error: PublishedExecutionError,
    ) -> ExecutionOutcome:
        return ExecutionOutcome(
            job_id=envelope.job_id,
            attempt_id=envelope.attempt_id,
            track_id=envelope.track_id,
            job_type=envelope.job_type,
            source_revision=envelope.source.revision,
            status="failed",
            error_code=error.code.value,
            result={
                "manifest_reference": {
                    "object_key": error.bundle.reference.key,
                    "object_version": error.bundle.reference.version,
                    "sha256": error.bundle.reference.sha256,
                    "size_bytes": error.bundle.reference.size_bytes,
                }
            },
        )

    @staticmethod
    def _url(envelope: AttemptEnvelope, object_key: str) -> str:
        return f"{str(envelope.storage.public_base_url).rstrip('/')}/{object_key}"

    @staticmethod
    def _event(
        envelope: AttemptEnvelope,
        sequence: int,
        stage: str,
        progress: float,
        event_type: ExecutionEventType,
        data: dict | None = None,
    ) -> ExecutionEvent:
        return ExecutionEvent(
            event_id=str(uuid.uuid4()),
            job_id=envelope.job_id,
            attempt_id=envelope.attempt_id,
            track_id=envelope.track_id,
            job_type=envelope.job_type,
            source_revision=envelope.source.revision,
            sequence=sequence,
            event=event_type,
            stage=stage,
            progress_pct=progress,
            data=data or {},
        )
