"""Backend dispatch clients: one receipt type for Pod and Serverless transports."""

import json

import httpx
import pytest

from hear.dispatch import (
    DispatcherFactory,
    DispatchError,
    DispatchReceipt,
    DispatchTransport,
    PodDispatcher,
    ServerlessDispatcher,
)
from hear.runtime.roles import WorkerRole
from tests.test_cleaning_profile_workflow import envelope


def pipeline_envelope():
    return envelope("natural").model_copy(update={"job_type": "pipeline", "options": {}})


class Recorder:
    def __init__(self, responder):
        self.requests = []
        self._responder = responder

    def transport(self):
        async def handle(request):
            self.requests.append(request)
            return self._responder(request)

        return httpx.MockTransport(handle)


@pytest.mark.anyio
async def test_pod_dispatcher_returns_canonical_receipt_on_confirmed_publish():
    request = envelope("natural")

    def respond(http_request):
        assert http_request.headers["authorization"] == "Bearer pod-token"
        assert http_request.url.path == "/v1/attempts"
        body = json.loads(http_request.content)
        assert body["attempt_id"] == request.attempt_id
        assert body["storage"]["application_key"] == "test-secret"
        return httpx.Response(202, json={"attempt_id": request.attempt_id, "status": "accepted"})

    recorder = Recorder(respond)
    async with httpx.AsyncClient(transport=recorder.transport()) as client:
        dispatcher = PodDispatcher("https://pod.example/", "pod-token", client=client)
        receipt = await dispatcher.submit(request)
        assert await dispatcher.cancel(receipt) is False
    assert receipt == DispatchReceipt(
        transport=DispatchTransport.POD,
        job_id=request.job_id,
        run_id=request.run_id,
        attempt_id=request.attempt_id,
        track_id=request.track_id,
        source_revision=request.source.revision,
    )
    assert receipt.result_delivery == "owning_backend_callback"


@pytest.mark.parametrize(
    "status,retryable", [(429, True), (503, True), (401, False), (403, False), (422, False)]
)
@pytest.mark.anyio
async def test_pod_dispatcher_maps_rejections_to_retry_policy(status, retryable):
    recorder = Recorder(lambda _request: httpx.Response(status, json={"detail": "job_queue_full"}))
    async with httpx.AsyncClient(transport=recorder.transport()) as client:
        dispatcher = PodDispatcher("https://pod.example", "pod-token", client=client)
        with pytest.raises(DispatchError) as info:
            await dispatcher.submit(envelope("natural"))
    assert info.value.retryable is retryable
    assert info.value.status_code == status
    assert info.value.code == "job_queue_full"


@pytest.mark.anyio
async def test_serverless_dispatcher_routes_by_role_and_wraps_envelope_in_input():
    request = pipeline_envelope()

    def respond(http_request):
        assert http_request.headers["authorization"] == "Bearer rp-key"
        if http_request.url.path == "/v2/pipe-1/run":
            payload = json.loads(http_request.content)
            assert payload["input"]["attempt_id"] == request.attempt_id
            assert payload["input"]["job_type"] == "pipeline"
            return httpx.Response(200, json={"id": "rp-job-1", "status": "IN_QUEUE"})
        if http_request.url.path == "/v2/pipe-1/status/rp-job-1":
            return httpx.Response(
                200,
                json={"id": "rp-job-1", "status": "IN_PROGRESS", "output": [{"event": "progress"}]},
            )
        if http_request.url.path == "/v2/pipe-1/cancel/rp-job-1":
            return httpx.Response(200, json={"id": "rp-job-1", "status": "CANCELLED"})
        if http_request.url.path == "/v2/clean-1/cancel/rp-job-1":
            return httpx.Response(404, json={"error": "not found"})
        if http_request.url.path.endswith("/health"):
            return httpx.Response(200, json={"workers": {"idle": 1, "running": 0}})
        raise AssertionError(http_request.url.path)

    recorder = Recorder(respond)
    async with httpx.AsyncClient(transport=recorder.transport()) as client:
        dispatcher = ServerlessDispatcher(
            {WorkerRole.PIPELINE: "pipe-1", WorkerRole.MAGIC_CLEAN_NATURAL: "clean-1"},
            "rp-key",
            client=client,
        )
        receipt = await dispatcher.submit(request)
        status = await dispatcher.status(receipt, "pipe-1")
        health = await dispatcher.health()
        cancelled = await dispatcher.cancel(receipt)
    assert receipt.transport == DispatchTransport.SERVERLESS
    assert receipt.provider_job_id == "rp-job-1"
    assert receipt.model_dump(exclude={"transport", "provider_job_id"}) == DispatchReceipt(
        transport=DispatchTransport.POD,
        job_id=request.job_id,
        run_id=request.run_id,
        attempt_id=request.attempt_id,
        track_id=request.track_id,
        source_revision=request.source.revision,
    ).model_dump(exclude={"transport", "provider_job_id"})
    assert status.state == "IN_PROGRESS" and status.events == ({"event": "progress"},)
    assert health.ready and set(health.detail) == {"pipeline", "magic_clean_natural"}
    assert cancelled is True


@pytest.mark.anyio
async def test_serverless_dispatcher_rejects_jobs_without_an_endpoint():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _r: httpx.Response(500))
    ) as client:
        dispatcher = ServerlessDispatcher({WorkerRole.PIPELINE: "pipe-1"}, "rp-key", client=client)
        with pytest.raises(DispatchError, match="serverless_endpoint_not_configured_for_job"):
            await dispatcher.submit(envelope("natural"))


def test_factory_selects_transport_from_backend_environment():
    pod = DispatcherFactory(
        {
            "HEAR_AI_TRANSPORT": "pod",
            "HEAR_POD_BASE_URL": "https://pod.example",
            "HEAR_POD_API_KEY": "k",
        }
    ).build()
    assert pod.transport == DispatchTransport.POD
    serverless = DispatcherFactory(
        {
            "HEAR_AI_TRANSPORT": "serverless",
            "HEAR_RUNPOD_API_KEY": "k",
            "HEAR_RUNPOD_ENDPOINTS_JSON": json.dumps({"pipeline": "a", "magic_clean_natural": "b"}),
        }
    ).build()
    assert serverless.transport == DispatchTransport.SERVERLESS
    with pytest.raises(ValueError, match="missing_dispatch_setting:HEAR_RUNPOD_API_KEY"):
        DispatcherFactory(
            {"HEAR_AI_TRANSPORT": "serverless", "HEAR_RUNPOD_ENDPOINTS_JSON": '{"pipeline": "a"}'}
        ).build()
