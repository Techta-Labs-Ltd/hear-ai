import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from hear.contracts.jobs import AttemptClaim, AttemptEnvelope, ClaimDecision, JobType
from hear.execution.lease import AttemptLease, AttemptLeaseLost


class FailingBackend:
    async def heartbeat(self, envelope, sequence):
        raise RuntimeError("backend_unavailable")


def envelope():
    return AttemptEnvelope(
        job_id="job-1",
        run_id="run-1",
        attempt_id="attempt-1",
        job_type=JobType.TRANSCRIPTION,
        track_id="track-1",
        user_id="user-1",
        source={"url": "https://example.com/a.mp3", "revision": 1},
        storage={
            "endpoint_url": "https://s3.example.com",
            "bucket_name": "bucket",
            "key_id": "key",
            "application_key": "secret",
            "folder_prefix": "users/user-1/",
            "public_base_url": "https://cdn.example.com/media",
            "expires_at": datetime.now(UTC) + timedelta(hours=1),
        },
        artifact_prefix="users/user-1/jobs/job-1/attempt-1",
        deadline=datetime.now(UTC) + timedelta(minutes=30),
        reporting_grant="grant",
        backend_base_url="https://api.example.com",
    )


@pytest.mark.anyio
async def test_lease_loss_interrupts_waiting_execution():
    claim = AttemptClaim(
        decision=ClaimDecision.EXECUTE,
        lease_seconds=1.01,
        heartbeat_seconds=0.05,
    )
    lease = AttemptLease(FailingBackend(), envelope(), claim)

    async def iterator():
        await asyncio.sleep(10)
        yield object()

    stream = iterator().__aiter__()
    lease.start()
    with pytest.raises(AttemptLeaseLost):
        await lease.next_event(stream)
    await lease.close()
    await stream.aclose()
