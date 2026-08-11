import logging

import grpc
from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import JSONResponse
from ray import serve
from ray.serve.handle import DeploymentHandle

from hear.config import settings
from hear.core.backend_registry import authenticate_backend, service_key_backend
from hear.core.category_loader import category_loader
from hear.core.discovery_taxonomy import discovery_taxonomy_loader
from hear.core.keyword_loader import auto_tag_keyword_loader, harm_keyword_loader
from hear.models.database import init_db
from hear.models.schemas import DiscoveryProcessRequest, PipelineRequest, ProcessResponse
from hear.proto import pipeline_pb2
from hear.services.jobs.submission import (
    JobSubmissionService,
    SubmissionConflictError,
    SubmissionUnavailableError,
)
from hear.services.model_client import RayModelClient, set_model_client
from hear.services.transport.grpc import PipelineGrpcService

logger = logging.getLogger(__name__)

http_app = FastAPI(
    title="Hear AI",
    description="Ray Serve ingress for the Hear AI application.",
    version="5.0.0",
    docs_url="/docs" if settings.ENABLE_DOCS else None,
    redoc_url="/redoc" if settings.ENABLE_DOCS else None,
    openapi_url="/openapi.json" if settings.ENABLE_DOCS else None,
)


@serve.deployment(
    name="grpc_gateway",
    ray_actor_options={"num_gpus": 0.0, "num_cpus": 0.5},
    num_replicas=settings.GATEWAY_REPLICA_COUNT,
    max_ongoing_requests=100,
    health_check_period_s=10,
    health_check_timeout_s=30,
)
@serve.ingress(http_app)
class GrpcGateway:
    def __init__(
        self,
        orchestrator: DeploymentHandle,
        audio_cleanup: DeploymentHandle,
        transcription: DeploymentHandle,
        fish_speech: DeploymentHandle,
        small_models: DeploymentHandle,
        llm: DeploymentHandle,
    ) -> None:
        init_db()
        category_loader.load()
        discovery_taxonomy_loader.load()
        harm_keyword_loader.load()
        auto_tag_keyword_loader.load()

        model_client = RayModelClient(
            {
                "transcription": transcription,
                "fish_speech": fish_speech,
                "small_models": small_models,
                "llm": llm,
            }
        )
        set_model_client(model_client)
        self._pipeline = PipelineGrpcService(orchestrator)
        self._submission = JobSubmissionService(orchestrator)
        self._audio_cleanup = audio_cleanup
        orchestrator.recover_jobs.remote()
        logger.info("FastAPI and gRPC gateway initialized")

    @http_app.get("/", tags=["system"])
    async def root(self) -> dict:
        return {
            "service": "hear-ai",
            "status": "running",
            "application": settings.GRPC_APPLICATION_NAME,
            "transports": ["http", "grpc"],
        }

    @http_app.get("/health", tags=["system"])
    async def http_health(self) -> dict:
        pipeline = await self._pipeline.health_data()
        return pipeline

    @http_app.get("/ready", tags=["system"])
    async def http_ready(self):
        pipeline = await self._pipeline.health_data()
        ready = pipeline.get("status") in {"healthy", "ready", "running"}
        payload = {"status": "ready" if ready else "loading"}
        return JSONResponse(payload, status_code=200 if ready else 503)

    @http_app.post(
        "/process",
        response_model=ProcessResponse,
        status_code=202,
        responses={
            200: {"model": ProcessResponse, "description": "Idempotent replay"},
            401: {"description": "Invalid service key"},
            409: {"description": "job_id payload conflict"},
            422: {"description": "Invalid job request"},
            503: {"description": "Job saved but dispatch unavailable"},
        },
        tags=["jobs"],
    )
    async def process_job(
        self,
        body: PipelineRequest,
        response: Response,
        service_key: str | None = Header(default=None, alias="X-Service-Key"),
    ) -> ProcessResponse:
        if not authenticate_backend(body.backend_id, service_key):
            raise HTTPException(status_code=401, detail="invalid service key")
        try:
            result = await self._submission.submit(body)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SubmissionConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except SubmissionUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("REST job submission failed")
            raise HTTPException(status_code=500, detail="job submission failed") from exc
        response.status_code = 200 if result.replayed else 202
        return ProcessResponse(**result.__dict__)

    @http_app.post(
        "/discovery",
        response_model=ProcessResponse,
        status_code=202,
        tags=["jobs"],
    )
    async def process_discovery(
        self,
        body: DiscoveryProcessRequest,
        response: Response,
        service_key: str | None = Header(default=None, alias="X-Service-Key"),
    ) -> ProcessResponse:
        if not authenticate_backend(body.backend_id, service_key):
            raise HTTPException(status_code=401, detail="invalid service key")
        request = PipelineRequest(
            backend_id=body.backend_id,
            storage=body.storage,
            job_id=body.job_id,
            track_id=body.track_id,
            job_type="discovery",
            audio_url=body.audio_url,
            user_id=body.user_id,
            source=body.source,
        )
        try:
            result = await self._submission.submit(request)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except SubmissionConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except SubmissionUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        response.status_code = 200 if result.replayed else 202
        return ProcessResponse(**result.__dict__)

    @staticmethod
    def _authenticated(context, backend_id: str | None = None) -> bool:
        metadata = dict(context.invocation_metadata() or ()) if context else {}
        supplied = metadata.get("x-api-key", "")
        application = metadata.get("application", "")
        authenticated_backend = service_key_backend(supplied)
        authenticated = (
            application == settings.GRPC_APPLICATION_NAME
            and authenticated_backend is not None
            and (backend_id is None or authenticated_backend == backend_id)
        )
        if not authenticated and context:
            context.set_code(grpc.StatusCode.UNAUTHENTICATED)
            context.set_details("invalid backend credentials")
        return authenticated

    async def SubmitJob(self, request, grpc_context=None):
        if not self._authenticated(grpc_context, request.backend_id):
            return pipeline_pb2.SubmitJobResponse(status="rejected", error="unauthenticated")
        try:
            body = PipelineRequest(
                backend_id=request.backend_id,
                storage={
                    "endpoint_url": request.storage.endpoint_url,
                    "bucket_name": request.storage.bucket_name,
                    "key_id": request.storage.key_id,
                    "application_key": request.storage.application_key,
                    "folder_prefix": request.storage.folder_prefix,
                    "public_base_url": request.storage.public_base_url,
                    "expires_at": request.storage.expires_at,
                },
                job_id=request.job_id,
                track_id=request.track_id,
                job_type=request.job_type or "pipeline",
                max_tags=request.max_tags or 8,
                audio_url=request.audio_url if request.HasField("audio_url") else None,
                edited_transcript=(
                    request.edited_transcript
                    if request.HasField("edited_transcript") else None
                ),
                changes=[
                    {
                        "segment_start": item.segment_start,
                        "segment_end": item.segment_end,
                        "new_text": item.new_text,
                        "original_text": (
                            item.original_text if item.HasField("original_text") else None
                        ),
                    }
                    for item in request.changes
                ],
                same_speaker=request.same_speaker,
                grouped=request.grouped,
                group_id=request.group_id if request.HasField("group_id") else None,
                kind=request.kind or "track",
                source=request.source if request.HasField("source") else None,
                track_count=request.track_count or 1,
                speed_multipliers=list(request.speed_multipliers),
                playback_instruction=(
                    request.playback_instruction
                    if request.HasField("playback_instruction") else None
                ),
                user_id=request.user_id if request.HasField("user_id") else "",
                speech=request.speech if request.HasField("speech") else None,
                music=request.music if request.HasField("music") else None,
                background=request.background if request.HasField("background") else None,
                cut_silence=(
                    request.cut_silence if request.HasField("cut_silence") else False
                ),
                type=request.type if request.HasField("type") else None,
                media_file_id=(
                    request.media_file_id if request.HasField("media_file_id") else None
                ),
            )
            result = await self._submission.submit(body)
            return pipeline_pb2.SubmitJobResponse(
                job_id=result.job_id,
                run_id=result.run_id,
                status=result.status,
                backend_id=result.backend_id,
            )
        except SubmissionConflictError as exc:
            code = grpc.StatusCode.ALREADY_EXISTS
            detail = str(exc)
        except SubmissionUnavailableError as exc:
            code = grpc.StatusCode.UNAVAILABLE
            detail = str(exc)
        except ValueError as exc:
            code = grpc.StatusCode.INVALID_ARGUMENT
            detail = str(exc)
        except Exception:
            logger.exception("gRPC job submission failed")
            code = grpc.StatusCode.INTERNAL
            detail = "job submission failed"
        if grpc_context:
            grpc_context.set_code(code)
            grpc_context.set_details(detail)
        return pipeline_pb2.SubmitJobResponse(
            job_id=request.job_id, status="rejected", error=detail
        )

    async def Subscribe(self, request, grpc_context=None):
        async for event in self._pipeline.Subscribe(request, grpc_context):
            yield event

    async def GetResult(self, request, grpc_context=None):
        return await self._pipeline.GetResult(request, grpc_context)

    async def CancelJob(self, request, grpc_context=None):
        return await self._pipeline.CancelJob(request, grpc_context)

    async def GetQueueStats(self, request, grpc_context=None):
        return await self._pipeline.GetQueueStats(request, grpc_context)

    async def Moderate(self, request, grpc_context=None):
        return await self._pipeline.Moderate(request, grpc_context)

    async def Categorize(self, request, grpc_context=None):
        return await self._pipeline.Categorize(request, grpc_context)

    async def CreatePreview(self, request, grpc_context=None):
        return await self._pipeline.CreatePreview(request, grpc_context)

    async def ConfirmPreview(self, request, grpc_context=None):
        return await self._pipeline.ConfirmPreview(request, grpc_context)

    async def RemoveSegment(self, request, grpc_context=None):
        return await self._pipeline.RemoveSegment(request, grpc_context)

    async def RollbackPreview(self, request, grpc_context=None):
        return await self._pipeline.RollbackPreview(request, grpc_context)

    async def GetPreview(self, request, grpc_context=None):
        return await self._pipeline.GetPreview(request, grpc_context)

    async def ListDiscovery(self, request, grpc_context=None):
        return await self._pipeline.ListDiscovery(request, grpc_context)

    async def TrainCategorizer(self, request, grpc_context=None):
        return await self._pipeline.TrainCategorizer(request, grpc_context)

    async def IngestCategoryEvent(self, request, grpc_context=None):
        return await self._pipeline.IngestCategoryEvent(request, grpc_context)

    async def UpdatePlatformSettings(self, request, grpc_context=None):
        return await self._pipeline.UpdatePlatformSettings(request, grpc_context)

    async def Health(self, request, grpc_context=None):
        return await self._pipeline.Health(request, grpc_context)

