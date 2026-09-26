import json
from pathlib import Path

from fastapi.testclient import TestClient

from hear.api.app import RuntimeApi
from hear.health.service import RuntimeReadiness
from hear.inference.manifest import ModelManifest
from hear.runtime.roles import WorkerRole


class PatchVerifier:
    def run(self, check: bool = False):
        return {"whisperx": "verified"}


class TestFastApiRuntime:
    def test_health_and_readiness_routes(self, tmp_path: Path):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({"models": []}))
        readiness = RuntimeReadiness(
            WorkerRole.TRANSCRIPTION,
            ModelManifest(manifest_path),
            tmp_path / "models",
            PatchVerifier(),
        )
        readiness.initialize()
        client = TestClient(RuntimeApi(readiness).app)
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 200
        assert client.get("/capabilities").json()["role"] == "transcription"

    def test_drain_route_marks_runtime_not_ready(self, tmp_path: Path):
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({"models": []}))
        readiness = RuntimeReadiness(
            WorkerRole.TRANSCRIPTION,
            ModelManifest(manifest_path),
            tmp_path / "models",
            PatchVerifier(),
        )
        readiness.initialize()
        calls = []

        async def drain():
            calls.append("drain")
            readiness.set_draining(True)

        client = TestClient(RuntimeApi(readiness, drain=drain).app)
        assert client.post("/drain").status_code == 200
        assert calls == ["drain"]
        response = client.get("/readyz")
        assert response.status_code == 503
        assert response.json()["status"] == "draining"
