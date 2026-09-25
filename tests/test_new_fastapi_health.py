from fastapi.testclient import TestClient

from hear.api.app import ApplicationFactory
from hear.contracts.jobs import WorkerRole
from hear.health.service import HealthService
from hear.runtime.roles import RoleRegistry


def test_health_and_capabilities():
    health = HealthService(
        RoleRegistry().get(WorkerRole.PIPELINE),
        checks={"patch": lambda: True, "models": lambda: True},
    )
    client = TestClient(ApplicationFactory(health).build())
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 200
    payload = client.get("/capabilities").json()
    assert payload["role"] == "pipeline"
    assert payload["job_types"] == ["pipeline", "transcription"]


def test_drain_removes_readiness():
    health = HealthService(RoleRegistry().get(WorkerRole.TRANSCRIPTION))
    client = TestClient(ApplicationFactory(health).build())
    assert client.post("/drain").status_code == 200
    assert client.get("/readyz").status_code == 503
