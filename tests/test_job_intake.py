"""Confirmed, non-streaming job intake and shared execution-scope tests."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hear.api.gateway import PodGateway
from hear.contracts.jobs import AttemptEnvelope
from hear.contracts.scope import ExecutionScope
from hear.queue.topology import RabbitMQTopology
from hear.runtime.gateway import (
    GatewayDeadlineExpired,
    GatewayQueueFull,
    GatewayUnavailable,
    RabbitMQGateway,
)
from hear.runtime.roles import WorkerRole
from tests.test_cleaning_profile_workflow import envelope


@pytest.mark.parametrize(
    "failure,status",
    [
        (None, 202),
        (GatewayQueueFull("full"), 429),
        (GatewayUnavailable("offline"), 503),
        (GatewayDeadlineExpired("expired"), 422),
    ],
)
def test_http_acceptance_requires_confirmed_publish(failure, status):
    request = envelope("natural")
    runtime = SimpleNamespace(submit=AsyncMock(side_effect=failure))
    gateway = PodGateway(runtime, "test-token")
    app = FastAPI()
    app.include_router(gateway.router)
    response = TestClient(app).post(
        "/v1/attempts",
        json=request.model_dump(mode="json"),
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == status
    if status == 202:
        body = response.json()
        assert body["attempt_id"] == request.attempt_id and body["job_id"] == request.job_id
        assert body["result_delivery"] == "owning_backend_callback"
        assert "storage" not in body and "reporting_grant" not in body
    assert runtime.submit.await_count == 1


def test_unauthorised_request_never_reaches_queue():
    runtime = SimpleNamespace(submit=AsyncMock())
    gateway = PodGateway(runtime, "test-token")
    app = FastAPI()
    app.include_router(gateway.router)
    response = TestClient(app).post(
        "/v1/attempts", json=envelope("natural").model_dump(mode="json")
    )
    assert response.status_code == 401 and runtime.submit.await_count == 0


@pytest.mark.parametrize("consumers,raises", [(0, True), (1, False)])
def test_intake_does_not_accept_jobs_for_an_absent_worker(consumers, raises):
    runtime = RabbitMQGateway("amqp://example", {WorkerRole.MAGIC_CLEAN_NATURAL})
    runtime._channel = SimpleNamespace()
    runtime._connection = SimpleNamespace(is_closed=False)
    runtime._status_channel = SimpleNamespace(
        declare_queue=AsyncMock(
            return_value=SimpleNamespace(
                declaration_result=SimpleNamespace(consumer_count=consumers, message_count=0)
            )
        )
    )
    if raises:
        with pytest.raises(GatewayUnavailable, match="worker_not_ready"):
            asyncio.run(runtime._admit(envelope("natural")))
    else:
        assert asyncio.run(runtime._admit(envelope("natural"))).routing_key == "magic_clean.natural"


def test_non_streaming_submission_creates_no_ephemeral_reply_queue():
    runtime = RabbitMQGateway("amqp://example", {WorkerRole.MAGIC_CLEAN_NATURAL})
    binding = RabbitMQTopology().binding(WorkerRole.MAGIC_CLEAN_NATURAL)
    runtime._admit = AsyncMock(return_value=binding)
    runtime._publish = AsyncMock()
    value = envelope("natural")
    asyncio.run(runtime.submit(value))
    runtime._publish.assert_awaited_once_with(value, binding, None)


def test_failed_publish_is_not_reported_as_accepted():
    provider = SimpleNamespace(
        Message=lambda **kwargs: SimpleNamespace(**kwargs),
        DeliveryMode=SimpleNamespace(PERSISTENT=2),
    )
    runtime = RabbitMQGateway("amqp://example", {WorkerRole.MAGIC_CLEAN_NATURAL}, provider)
    runtime._exchange = SimpleNamespace(publish=AsyncMock(side_effect=OSError("disconnected")))
    binding = RabbitMQTopology().binding(WorkerRole.MAGIC_CLEAN_NATURAL)
    with pytest.raises(GatewayUnavailable, match="retry_same_attempt"):
        asyncio.run(runtime._publish(envelope("natural"), binding, None))


def test_submitted_options_are_not_mutated_before_scope_verification():
    raw = envelope("natural").model_dump(mode="json")
    raw["options"] = {"profile": "studio_voice"}
    raw["backend_id"] = "backend-a"
    before = ExecutionScope.digest(raw)
    validated = AttemptEnvelope.model_validate(raw)
    assert validated.options == {"profile": "studio_voice"}
    assert ExecutionScope.digest(validated.model_dump(mode="json")) == before


def test_scope_detects_changed_cleaning_options():
    raw = envelope("natural").model_dump(mode="json")
    before = ExecutionScope.digest(raw)
    raw["options"] = {"profile": "clean_raw"}
    assert ExecutionScope.digest(raw) != before


def test_scope_matches_backend_reference_vector():
    value = {
        "schema_version": 1,
        "backend_id": "backend-a",
        "job_id": "job",
        "run_id": "run",
        "attempt_id": "attempt",
        "track_id": "track",
        "user_id": "user",
        "job_type": "magic_clean",
        "options": {"profile": "natural"},
        "source": {"url": "https://cdn.example.com/a.mp3", "revision": 1, "file_sha256": "a" * 64},
        "storage": {
            "endpoint_url": "https://s3.example.com",
            "bucket_name": "bucket",
            "key_id": "test-id",
            "application_key": "test-value",
            "folder_prefix": "creators/me/audio/jobs/job/",
            "public_base_url": "https://cdn.example.com",
            "expires_at": "2030-01-01T00:00:00Z",
        },
        "artifact_prefix": "creators/me/audio/jobs/job/attempt",
        "backend_base_url": "https://api.example.com/api/v1",
        "deadline": "2030-01-01T00:00:00+00:00",
    }
    assert (
        ExecutionScope.digest(value)
        == "90e6ec34959edae2986abde0cb6f14a7c8621dd347e3e246c1454c191f6fc9c5"
    )
