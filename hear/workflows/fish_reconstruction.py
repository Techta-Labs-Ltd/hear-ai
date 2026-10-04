"""One Fish TTS workflow for Pod and Serverless; result ownership stays in the backend."""

from __future__ import annotations

import asyncio
import hashlib
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from hear.audio.workspace import AudioWorkspace
from hear.contracts.events import ExecutionEvent, ExecutionEventType
from hear.contracts.outcomes import ExecutionOutcome
from hear.contracts.reconstruction import ReconstructionOptions
from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode
from hear.services.reconstruction.fish_renderer import FishReconstructionRenderer


class FishReconstructionWorkflow:
    def __init__(
        self,
        renderer: FishReconstructionRenderer,
        audio,
        storage_factory,
        native,
        *,
        workspace_root: Path,
        timeout_seconds: float = 1800,
        budget: ResourceBudget | None = None,
    ):
        self.renderer = renderer
        self.audio = audio
        self.storage_factory = storage_factory
        self.native = native
        self.workspace_root = workspace_root
        self.timeout_seconds = timeout_seconds
        self.budget = budget or ResourceBudget(8 * 1024**3, 2 * 1024**3, 48000 * 3600)

    @staticmethod
    def digest(path: Path) -> str:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    async def stream(self, envelope):
        operation = envelope.operation.value
        options = ReconstructionOptions.validate_operation(operation, envelope.options)
        remaining = (envelope.deadline - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            raise CleanExecutionError(ErrorCode.DEADLINE_EXCEEDED, "attempt_deadline_exceeded")
        if envelope.storage.expires_at <= datetime.now(UTC):
            raise ValueError("storage_grant_expired")
        workspace = AudioWorkspace(
            self.workspace_root / envelope.workspace_namespace, envelope.job_id, envelope.attempt_id
        )
        cancelled = threading.Event()
        guard = ResourceGuard(
            self.budget,
            workspace.path,
            time.monotonic() + min(remaining, self.timeout_seconds),
            cancelled,
        )
        guard.bind_deadline(envelope.deadline)
        task = None
        sequence = 1
        try:
            guard.check_scratch()
            yield self.event(envelope, sequence, "preparing", 0, ExecutionEventType.STARTED)
            sequence += 1
            source = await self.audio.download_source(str(envelope.source.url), workspace)
            if source.stat().st_size > self.budget.max_input_bytes:
                raise ValueError("reconstruction_source_too_large")
            digest = await self.native.run(self.digest, source)
            if envelope.source.file_sha256 and digest != envelope.source.file_sha256:
                raise CleanExecutionError(
                    ErrorCode.SOURCE_MISMATCH, "reconstruction_source_digest_mismatch"
                )
            progress = asyncio.Queue(maxsize=4)

            def report(value):
                if progress.full():
                    progress.get_nowait()
                progress.put_nowait(value)

            yield self.event(envelope, sequence, "generating_speech", 15)
            sequence += 1
            task = asyncio.create_task(
                self.renderer.render(
                    source,
                    operation,
                    options,
                    guard,
                    f"{envelope.job_id}:{digest}",
                    progress=report,
                )
            )
            while not task.done():
                try:
                    value = await asyncio.wait_for(progress.get(), timeout=0.25)
                except TimeoutError:
                    continue
                yield self.event(envelope, sequence, "generating_speech", value)
                sequence += 1
            result = await task
            guard.check()
            storage = self.storage_factory.create(envelope.storage)
            artifacts = []

            async def upload(path, name, mime):
                guard.check()
                sha = await self.native.run(self.digest, path)
                item = await self.native.run(
                    storage.upload_file,
                    path,
                    storage.key("jobs", envelope.job_id, envelope.attempt_id, name),
                    sha256=sha,
                    content_type=mime,
                )
                artifacts.append(item)
                return item

            yield self.event(envelope, sequence, "uploading", 85)
            sequence += 1
            delivery = await upload(result["delivery"], "reconstructed.mp3", "audio/mpeg")
            rows = []
            for index, (segment, row) in enumerate(
                zip(result["segments"], result["timeline"], strict=True)
            ):
                row = dict(row)
                if segment.delivery is not None:
                    segment_artifact = await upload(
                        segment.delivery, f"segments/segment-{index:03d}.mp3", "audio/mpeg"
                    )
                    row.update(
                        b2_key=segment_artifact.object_key,
                        audio_url=segment_artifact.audio_url,
                        bucket_name=segment_artifact.bucket_name,
                        sha256=segment_artifact.sha256,
                    )
                rows.append(row)
            payload = {
                key: result[key]
                for key in (
                    "sample_rate",
                    "source_frames",
                    "output_frames",
                    "channels",
                    "duration",
                    "duration_delta_seconds",
                    "timeline_policy",
                    "delivery_measurement",
                    "final_gain_db",
                )
                if key in result
            }
            payload.update(
                {
                    "source_sha256": digest,
                    "delivery": {**delivery.model_dump(mode="json"), "duration_seconds": result.get("duration")},
                    "segments": rows,
                }
            )
            outcome = ExecutionOutcome(
                job_id=envelope.job_id,
                attempt_id=envelope.attempt_id,
                track_id=envelope.track_id,
                job_type=envelope.job_type,
                backend_id=envelope.backend_id,
                source_revision=envelope.source.revision,
                status="completed",
                artifacts=tuple(artifacts),
                result={
                    "operation": operation,
                    "engine": "fish_speech_s2_pro",
                    "requires_approval": True,
                    "reconstructed_audio": payload,
                },
            )
            yield self.event(
                envelope,
                sequence,
                "completed",
                100,
                ExecutionEventType.OUTCOME,
                {"outcome": outcome.model_dump(mode="json")},
            )
        finally:
            cancelled.set()
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            workspace.cleanup()

    @staticmethod
    def event(envelope, sequence, stage, percent, kind=ExecutionEventType.STAGE, data=None):
        return ExecutionEvent(
            event_id=str(uuid.uuid4()),
            job_id=envelope.job_id,
            attempt_id=envelope.attempt_id,
            track_id=envelope.track_id,
            job_type=envelope.job_type,
            backend_id=envelope.backend_id,
            source_revision=envelope.source.revision,
            sequence=sequence,
            event=kind,
            stage=stage,
            progress_pct=percent,
            data=data or {},
        )
