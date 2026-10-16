import json

from hear.runtime.ownership import BackendOwnershipPolicy
from scripts.serverless_canary import ServerlessCanary

POLICY = {
    "backend_id": "backend-a",
    "backend_base_urls": ["https://api.example.com/api/v1"],
    "bucket_name": "bucket",
    "storage_endpoint": "https://s3.example.com",
    "public_base_url": "https://cdn.example.com",
    "source_hosts": ["cdn.example.com"],
}


def test_synthetic_canary_envelope_satisfies_deployment_policy():
    for job_type in ("pipeline", "transcription", "magic_clean", "reconstruction"):
        envelope = ServerlessCanary.synthetic_envelope(POLICY, job_type)
        BackendOwnershipPolicy.from_json(json.dumps(POLICY)).validate(envelope)
        assert envelope.job_type.value == job_type
        assert envelope.reporting_grant.startswith("canary-grant")


def test_canary_stops_waiting_when_workers_keep_dying_and_cancels_its_job():
    import asyncio

    import httpx

    calls = []
    health = {
        "workers": {"ready": 0, "running": 0, "idle": 0, "unhealthy": 3},
        "jobs": {"inQueue": 1},
    }

    def handle(request):
        calls.append((request.method, request.url.path))
        path = request.url.path
        if path.endswith("/health"):
            return httpx.Response(200, json=health)
        if path.endswith("/run"):
            return httpx.Response(200, json={"id": "job-1", "status": "IN_QUEUE"})
        if path.endswith("/status/job-1"):
            return httpx.Response(200, json={"status": "IN_QUEUE"})
        if path.endswith("/stream/job-1"):
            return httpx.Response(404)
        if path.endswith("/cancel/job-1"):
            return httpx.Response(200, json={"status": "CANCELLED"})
        raise AssertionError(path)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    canary = ServerlessCanary(
        "rpa_test", base_url="https://run.test/v2", client=client, never_ready_seconds=0
    )
    envelope = ServerlessCanary.synthetic_envelope(POLICY, "pipeline")
    result = asyncio.run(canary.run("ep1", envelope.job_type, envelope, wait_seconds=60))
    assert result["final_state"] == "workers_never_ready" and result["seconds"] < 30
    assert result["health_after"]["workers"]["unhealthy"] == 3
    assert ("POST", "/v2/ep1/cancel/job-1") in calls


def test_canary_keeps_waiting_while_a_worker_is_alive():
    import asyncio

    import httpx

    polls = {"n": 0}

    def handle(request):
        path = request.url.path
        if path.endswith("/health"):
            return httpx.Response(200, json={"workers": {"ready": 1, "unhealthy": 2}})
        if path.endswith("/run"):
            return httpx.Response(200, json={"id": "job-2", "status": "IN_QUEUE"})
        if path.endswith("/status/job-2"):
            polls["n"] += 1
            return httpx.Response(
                200, json={"status": "COMPLETED" if polls["n"] >= 2 else "IN_PROGRESS"}
            )
        if path.endswith("/stream/job-2"):
            return httpx.Response(200, json={"stream": [{"output": {"event": "started"}}]})
        raise AssertionError(path)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    canary = ServerlessCanary(
        "rpa_test", base_url="https://run.test/v2", client=client, never_ready_seconds=0
    )
    envelope = ServerlessCanary.synthetic_envelope(POLICY, "pipeline")
    result = asyncio.run(canary.run("ep1", envelope.job_type, envelope, wait_seconds=60))
    assert result["final_state"] == "COMPLETED" and "started" in result["worker_events"]
