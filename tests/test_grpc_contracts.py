import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import grpc
import pytest

from hear.config import settings
from hear.core.backend_registry import backend_registry
from hear.deployments.gateway import GrpcGateway
from hear.models.schemas import PipelineRequest
from hear.proto import pipeline_pb2
from hear.services.transport.grpc import PipelineGrpcService

GatewayClass = GrpcGateway.func_or_class
OriginalGatewayClass = GatewayClass.__wrapped__


def configure_backend_auth(monkeypatch, service_key: str = "secret") -> None:
    digest = hashlib.sha256(service_key.encode()).hexdigest()
    monkeypatch.setattr(
        settings,
        "BACKEND_REGISTRY_JSON",
        json.dumps({
            "backend-a": {
                "service_key_sha256": digest,
                "allowed_endpoint_urls": ["https://s3.example.test"],
                "allowed_buckets": ["bucket-a"],
                "allowed_public_base_urls": ["https://cdn.example.test"],
            }
        }),
    )
    backend_registry.cache_clear()


class FakeContext:
    def __init__(self, metadata):
        self._metadata = metadata
        self.code = None
        self.details = None

    def invocation_metadata(self):
        return self._metadata

    def set_code(self, code):
        self.code = code

    def set_details(self, details):
        self.details = details


def test_pipeline_contract_has_full_grpc_surface():
    methods = {
        method.name
        for method in pipeline_pb2.DESCRIPTOR.services_by_name["Pipeline"].methods
    }
    assert methods == {
        "SubmitJob",
        "Subscribe",
        "GetResult",
        "CancelJob",
        "GetQueueStats",
        "Moderate",
        "Categorize",
        "CreatePreview",
        "ConfirmPreview",
        "RemoveSegment",
        "RollbackPreview",
        "GetPreview",
        "ListDiscovery",
        "TrainCategorizer",
        "IngestCategoryEvent",
        "UpdatePlatformSettings",
        "Health",
    }


def test_every_pipeline_rpc_has_a_gateway_handler():
    methods = pipeline_pb2.DESCRIPTOR.services_by_name["Pipeline"].methods
    assert all(callable(getattr(GatewayClass, method.name, None)) for method in methods)


def test_gateway_does_not_expose_resolver_rpcs():
    assert not any(
        callable(getattr(GatewayClass, method, None))
        for method in ("Resolve", "ResolverHealth", "Rebuild", "Apply")
    )


@pytest.mark.anyio
async def test_submit_job_accepts_track_without_existence_gate(monkeypatch):
    configure_backend_auth(monkeypatch, "secret")
    monkeypatch.setattr(
        OriginalGatewayClass,
        "_authenticated",
        staticmethod(lambda _context, _backend_id=None: True),
    )
    gateway = object.__new__(OriginalGatewayClass)
    gateway._submission = SimpleNamespace(
        submit=AsyncMock(
            return_value=SimpleNamespace(
                backend_id="backend-a", job_id="job-1", run_id="run-1", status="queued"
            )
        )
    )
    context = FakeContext((("x-api-key", "secret"), ("application", "hear")))
    request = pipeline_pb2.SubmitJobRequest(
        job_id="job-1",
        track_id="track-1",
        job_type="magic_clean",
        audio_url="https://example.test/audio.mp3",
        user_id="user-1",
        backend_id="backend-a",
        storage=pipeline_pb2.StorageContext(
            endpoint_url="https://s3.example.test",
            bucket_name="bucket-a",
            key_id="key-id",
            application_key="application-key",
            folder_prefix="users/user/jobs/job",
            public_base_url="https://cdn.example.test",
            expires_at="2099-01-01T00:00:00Z",
        ),
    )

    response = await OriginalGatewayClass.SubmitJob(gateway, request, context)

    assert response.backend_id == "backend-a"
    assert response.job_id == "job-1"
    assert response.run_id == "run-1"
    assert response.status == "queued"
    assert context.code is None
    submitted = gateway._submission.submit.await_args.args[0]
    assert submitted.track_id == "track-1"
    assert submitted.audio_url == "https://example.test/audio.mp3"


def test_track_exists_is_not_part_of_submission_contract():
    assert "track_exists" not in PipelineRequest.model_fields
    assert "track_exists" not in pipeline_pb2.SubmitJobRequest.DESCRIPTOR.fields_by_name


def test_grpc_authentication_rejects_invalid_key(monkeypatch):
    configure_backend_auth(monkeypatch, "expected")
    context = FakeContext((("x-api-key", "wrong"), ("application", "hear")))

    assert not PipelineGrpcService._authenticated(context)
    assert context.code == grpc.StatusCode.UNAUTHENTICATED
    assert context.details == "invalid backend credentials"


def test_grpc_authentication_rejects_missing_application(monkeypatch):
    configure_backend_auth(monkeypatch, "expected")
    context = FakeContext((("x-api-key", "expected"),))

    assert not PipelineGrpcService._authenticated(context)
    assert context.code == grpc.StatusCode.UNAUTHENTICATED



def test_grpc_authentication_accepts_valid_key(monkeypatch):
    configure_backend_auth(monkeypatch, "expected")
    context = FakeContext((("x-api-key", "expected"), ("application", "hear")))

    assert PipelineGrpcService._authenticated(context)
    assert context.code is None
