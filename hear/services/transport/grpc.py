from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from datetime import date, datetime
from typing import Any

import grpc
from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Struct
from pydantic import BaseModel
from ray.serve.handle import DeploymentHandle

from hear.config import settings
from hear.core.backend_registry import BackendRegistry
from hear.core.blocking import BoundedDatabaseWork
from hear.models.database import AiJob, DatabaseRuntime
from hear.models.schemas import StorageContext
from hear.proto import pipeline_pb2
from hear.services.transport.operations import Operations

logger = logging.getLogger(__name__)


class ProtobufValues:
    @staticmethod
    def _json_safe(value: Any) -> Any:
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        if isinstance(value, Mapping):
            return {str(key): ProtobufValues._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [ProtobufValues._json_safe(item) for item in value]
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    @staticmethod
    def _struct(value: Any = None) -> Struct:
        result = Struct()
        safe = ProtobufValues._json_safe(value or {})
        if isinstance(safe, dict):
            result.update(safe)
        else:
            result.update({"value": safe})
        return result


JOB_TYPE_PAYLOAD_MAP: dict[str, tuple[str, type]] = {
    "pipeline": ("pipeline", pipeline_pb2.PipelinePayload),
    "categorization": ("pipeline", pipeline_pb2.PipelinePayload),
    "rebuild": ("pipeline", pipeline_pb2.PipelinePayload),
    "transcription": ("transcription", pipeline_pb2.TranscriptionPayload),
    "audio_tag": ("audio_tag", pipeline_pb2.AudioTagPayload),
    "magic_clean": ("magic_clean", pipeline_pb2.MagicCleanPayload),
    "reconstruct": ("reconstruct", pipeline_pb2.ReconstructPayload),
    "edit_transcript": ("reconstruct", pipeline_pb2.ReconstructPayload),
    "discovery": ("pipeline", pipeline_pb2.PipelinePayload),
}


class PipelineGrpcService:
    def __init__(self, orchestrator: DeploymentHandle, operations: Operations) -> None:
        self._orchestrator = orchestrator
        self._operations = operations

    @staticmethod
    def _backend_id(context) -> str | None:
        metadata = dict(context.invocation_metadata() or ()) if context else {}
        if metadata.get("application", "") != settings.GRPC_APPLICATION_NAME:
            backend_id = None
        else:
            backend_id = BackendRegistry.service_key_backend(metadata.get("x-api-key", ""))
        if not backend_id and context:
            context.set_code(grpc.StatusCode.UNAUTHENTICATED)
            context.set_details("invalid backend credentials")
        return backend_id

    @classmethod
    def _authenticated(cls, context) -> bool:
        return cls._backend_id(context) is not None

    @staticmethod
    def _storage_context(message) -> StorageContext:
        return StorageContext.model_validate(
            {
                "endpoint_url": message.endpoint_url,
                "bucket_name": message.bucket_name,
                "key_id": message.key_id,
                "application_key": message.application_key,
                "folder_prefix": message.folder_prefix,
                "public_base_url": message.public_base_url,
                "expires_at": message.expires_at,
            }
        )

    @staticmethod
    def _set_error(context, code: grpc.StatusCode, details: str) -> None:
        if context:
            context.set_code(code)
            context.set_details(details)

    async def _call(self, context, fn: Callable[[], Awaitable[dict]], msg_class: type) -> Any:
        if not self._authenticated(context):
            return msg_class()
        try:
            result = await fn()
            return ParseDict(result, msg_class(), ignore_unknown_fields=True)
        except Exception as exc:
            status_code = getattr(exc, "status_code", 500)
            code = {
                400: grpc.StatusCode.INVALID_ARGUMENT,
                404: grpc.StatusCode.NOT_FOUND,
                409: grpc.StatusCode.ALREADY_EXISTS,
                422: grpc.StatusCode.INVALID_ARGUMENT,
                503: grpc.StatusCode.UNAVAILABLE,
            }.get(status_code, grpc.StatusCode.INTERNAL)
            detail = getattr(exc, "detail", "operation failed")
            self._set_error(context, code, str(detail))
            logger.exception("gRPC operation failed")
            return msg_class()

    async def Subscribe(self, request, context=None):
        backend_id = self._backend_id(context)
        if not backend_id:
            return
        def owned_job() -> bool:
            db = DatabaseRuntime.SessionLocal()
            try:
                return bool(
                    db.query(AiJob.id)
                    .filter(AiJob.id == request.job_id, AiJob.backend_id == backend_id)
                    .first()
                )
            finally:
                db.close()

        owned = await BoundedDatabaseWork.run(owned_job)
        if not owned:
            self._set_error(context, grpc.StatusCode.NOT_FOUND, "job not found")
            return
        stream = self._orchestrator.subscribe.options(stream=True).remote(request.job_id)
        async for raw in stream:
            event = pipeline_pb2.PipelineEvent(
                event=raw.get("event") or "",
                job_id=raw.get("job_id") or "",
                run_id=raw.get("run_id") or "",
                track_id=raw.get("track_id") or "",
                job_type=raw.get("job_type") or "",
                status=raw.get("status") or "",
                current_stage=raw.get("current_stage") or "",
                label=raw.get("label") or "",
                description=raw.get("description") or "",
                error=raw.get("error") or "",
                backend_id=raw.get("backend_id") or "",
            )
            for field in ("progress_pct", "elapsed_seconds", "estimated_remaining"):
                if raw.get(field) is not None:
                    setattr(event, field, raw[field])
            if raw.get("result"):
                event.result.CopyFrom(ProtobufValues._struct(raw["result"]))
            error_report = raw.get("error_report")
            if error_report is None and raw.get("event") == "job_failed":
                error_report = (raw.get("result") or {}).get("report")
            if error_report:
                event.error_report.CopyFrom(ProtobufValues._struct(error_report))
            yield event
            if raw.get("event") in {"job_completed", "job_failed", "job_cancelled"}:
                break

    async def GetResult(self, request, context=None):
        backend_id = self._backend_id(context)
        if not backend_id:
            return pipeline_pb2.JobResult()
        def read_job() -> dict | None:
            db = DatabaseRuntime.SessionLocal()
            try:
                job = (
                    db.query(AiJob)
                    .filter(AiJob.id == request.job_id, AiJob.backend_id == backend_id)
                    .first()
                )
                if job is None:
                    return None
                return {
                    "job_id": job.id or "", "run_id": job.run_id or "",
                    "track_id": job.track_id or "", "job_type": job.job_type or "",
                    "status": job.status or "", "current_stage": job.current_stage or "",
                    "error": job.error or "", "backend_id": job.backend_id or "",
                    "result_json": job.result_json or {},
                }
            finally:
                db.close()

        job = await BoundedDatabaseWork.run(read_job)
        if job is None:
            self._set_error(context, grpc.StatusCode.NOT_FOUND, "job not found")
            return pipeline_pb2.JobResult()
        response = pipeline_pb2.JobResult(
            **{
                key: job[key]
                for key in (
                    "job_id",
                    "run_id",
                    "track_id",
                    "job_type",
                    "status",
                    "current_stage",
                    "error",
                    "backend_id",
                )
            }
        )
        terminal_error = job["result_json"].get("_terminal_error")
        if terminal_error:
            response.error_report.CopyFrom(ProtobufValues._struct(terminal_error))
        oneof_field, msg_class = JOB_TYPE_PAYLOAD_MAP.get(job["job_type"], (None, None))
        if oneof_field and job["result_json"]:
            payload = ParseDict(job["result_json"], msg_class(), ignore_unknown_fields=True)
            getattr(response, oneof_field).CopyFrom(payload)
        return response

    async def CancelJob(self, request, context=None):
        backend_id = self._backend_id(context)
        if not backend_id:
            return pipeline_pb2.JobResult()
        def owned_job() -> bool:
            db = DatabaseRuntime.SessionLocal()
            try:
                return bool(
                    db.query(AiJob.id)
                    .filter(AiJob.id == request.job_id, AiJob.backend_id == backend_id)
                    .first()
                )
            finally:
                db.close()

        owned = await BoundedDatabaseWork.run(owned_job)
        if not owned:
            self._set_error(context, grpc.StatusCode.NOT_FOUND, "job not found")
            return pipeline_pb2.JobResult()
        found = await self._orchestrator.cancel.remote(request.job_id)
        if not found:
            self._set_error(context, grpc.StatusCode.NOT_FOUND, "job not found")
            return pipeline_pb2.JobResult()
        return await self.GetResult(request, context)

    async def GetQueueStats(self, request, context=None):
        return await self._call(
            context, lambda: self._orchestrator.get_stats.remote(), pipeline_pb2.QueueStatsReply
        )

    async def Moderate(self, request, context=None):
        return await self._call(
            context, lambda: self._operations.moderate(request.text), pipeline_pb2.ModerationReply
        )

    async def Categorize(self, request, context=None):
        return await self._call(
            context,
            lambda: self._operations.categorize(
                request.text, list(request.custom_tags), request.max_tags or 8
            ),
            pipeline_pb2.CategorizationReply,
        )

    async def CreatePreview(self, request, context=None):
        backend_id = self._backend_id(context)
        if not backend_id or backend_id != request.backend_id:
            return pipeline_pb2.CreatePreviewReply()
        changes = [
            {
                "segment_start": item.segment_start,
                "segment_end": item.segment_end,
                "new_text": item.new_text,
                "original_text": item.original_text if item.HasField("original_text") else None,
            }
            for item in request.changes
        ]
        return await self._call(
            context,
            lambda: self._operations.create_preview(
                audio_url=request.audio_url,
                track_id=request.track_id,
                changes=changes,
                segment_start=request.segment_start if request.HasField("segment_start") else None,
                segment_end=request.segment_end if request.HasField("segment_end") else None,
                new_text=request.new_text if request.HasField("new_text") else None,
                same_speaker=request.same_speaker if request.HasField("same_speaker") else True,
                backend_id=backend_id,
                storage_context=self._storage_context(request.storage),
            ),
            pipeline_pb2.CreatePreviewReply,
        )

    async def ConfirmPreview(self, request, context=None):
        backend_id = self._backend_id(context)
        if not backend_id:
            return pipeline_pb2.ConfirmPreviewReply()
        return await self._call(
            context,
            lambda: self._operations.confirm_preview(
                request.preview_id,
                request.track_id if request.HasField("track_id") else None,
                request.user_id if request.HasField("user_id") else None,
                backend_id,
            ),
            pipeline_pb2.ConfirmPreviewReply,
        )

    async def RemoveSegment(self, request, context=None):
        backend_id = self._backend_id(context)
        if not backend_id:
            return pipeline_pb2.RemoveSegmentReply()
        return await self._call(
            context,
            lambda: self._operations.remove_segment(
                track_id=request.track_id,
                audio_url=request.audio_url,
                segment_start=request.segment_start,
                segment_end=request.segment_end,
                user_id=request.user_id if request.HasField("user_id") else None,
                backend_id=backend_id,
                storage_context=self._storage_context(request.storage),
            ),
            pipeline_pb2.RemoveSegmentReply,
        )

    async def RollbackPreview(self, request, context=None):
        backend_id = self._backend_id(context)
        if not backend_id:
            return pipeline_pb2.RollbackPreviewReply()
        return await self._call(
            context,
            lambda: self._operations.rollback_preview(request.preview_id, backend_id),
            pipeline_pb2.RollbackPreviewReply,
        )

    async def GetPreview(self, request, context=None):
        backend_id = self._backend_id(context)
        if not backend_id:
            return pipeline_pb2.Preview()
        return await self._call(
            context,
            lambda: self._operations.get_preview(request.preview_id, backend_id),
            pipeline_pb2.Preview,
        )

    async def ListDiscovery(self, request, context=None):
        return await self._call(
            context,
            lambda: self._operations.list_discovery(
                request.sort or "latest", request.limit or 50, request.offset
            ),
            pipeline_pb2.ListDiscoveryReply,
        )

    async def health_data(self) -> dict:
        try:
            async with asyncio.timeout(2.0):
                queue = await self._orchestrator.get_stats.remote()
        except Exception:
            result = await self._operations.health({})
            result.update(
                status="unavailable", control_ready=False, error="orchestrator_unavailable"
            )
            return result
        return await self._operations.health(queue)

    async def Health(self, request, context=None):
        return await self._call(context, self.health_data, pipeline_pb2.HealthReply)
