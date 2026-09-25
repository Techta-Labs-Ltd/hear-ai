import asyncio

import httpx

from hear.contracts.jobs import AttemptClaimStatus, AttemptEnvelope, JobType
from hear.execution.reporter import BackendReporter


def envelope() -> AttemptEnvelope:
    return AttemptEnvelope.model_validate(
        {
            "job_id": JobType.PIPELINE.value + "-job",
            "run_id": "run-1",
            "attempt_id": "attempt-1",
            "job_type": "pipeline",
            "track_id": "track-1",
            "user_id": "user-1",
            "source": {"url": "https://media.example/audio.mp3", "revision": 3},
            "storage": {
                "reference": "storage-1",
                "token": "storage-secret",
                "expires_at": "2026-09-26T00:00:00Z",
            },
            "reporting": {
                "backend_base_url": "https://api.example",
                "token": "report-secret",
            },
        }
    )


def test_reporter_claims_attempt_before_execution():
    async def run():
        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/internal/ai/attempts/attempt-1/claim"
            assert request.headers["Authorization"] == "Bearer report-secret"
            return httpx.Response(
                200,
                json={
                    "status": "execute",
                    "lease_seconds": 60,
                    "heartbeat_seconds": 15,
                },
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await BackendReporter(envelope(), client).claim()

    claim = asyncio.run(run())
    assert claim.status == AttemptClaimStatus.EXECUTE
    assert claim.lease_seconds == 60
    assert claim.heartbeat_seconds == 15
