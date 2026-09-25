import httpx

from hear.contracts.events import ExecutionEvent
from hear.contracts.jobs import AttemptClaimResponse, AttemptEnvelope
from hear.contracts.outcomes import JobOutcome


class BackendReporter:
    def __init__(
        self,
        envelope: AttemptEnvelope,
        client: httpx.AsyncClient,
    ) -> None:
        self._envelope = envelope
        self._client = client
        self._base_url = envelope.reporting.backend_base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {envelope.reporting.token.get_secret_value()}",
        }

    async def claim(self) -> AttemptClaimResponse:
        response = await self._client.post(
            f"{self._base_url}/internal/ai/attempts/{self._envelope.attempt_id}/claim",
            json={
                "job_id": self._envelope.job_id,
                "run_id": self._envelope.run_id,
                "attempt_id": self._envelope.attempt_id,
                "job_type": self._envelope.job_type.value,
                "track_id": self._envelope.track_id,
                "source_revision": self._envelope.source.revision,
            },
            headers=self._headers,
        )
        response.raise_for_status()
        return AttemptClaimResponse.model_validate(response.json())

    async def publish_event(self, event: ExecutionEvent) -> None:
        response = await self._client.post(
            f"{self._base_url}/internal/ai/attempts/{self._envelope.attempt_id}/events",
            json=event.model_dump(mode="json"),
            headers=self._headers,
        )
        response.raise_for_status()

    async def publish_outcome(self, outcome: JobOutcome) -> None:
        response = await self._client.post(
            f"{self._base_url}/internal/ai/attempts/{self._envelope.attempt_id}/outcome",
            json=outcome.model_dump(mode="json"),
            headers=self._headers,
        )
        response.raise_for_status()
