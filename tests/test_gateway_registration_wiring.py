"""The gateway entrypoint only registers when it knows where it is and whom to tell."""

import logging

from hear.config import RuntimeSettings
from hear.entrypoints.gateway import GatewayEntrypoint
from hear.runtime.roles import WorkerRole


class FakeRuntime:
    async def lane_status(self):
        return {}


def settings(**extra):
    base = {"HEAR_IMAGE_REVISION": "pod-1", "HEAR_WORKER_ROLE": "pipeline"}
    return RuntimeSettings.from_environment({**base, **extra})


def test_registrar_needs_backend_and_address(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    monkeypatch.delenv("HEAR_PUBLIC_GATEWAY_URL", raising=False)
    assert GatewayEntrypoint.registrar(settings(), FakeRuntime(), {WorkerRole.PIPELINE}) is None
    assert "set HEAR_PUBLIC_GATEWAY_URL" in caplog.text
    monkeypatch.setenv("RUNPOD_POD_ID", "pod42")
    assert GatewayEntrypoint.registrar(settings(), FakeRuntime(), {WorkerRole.PIPELINE}) is None
    assert "service key missing" in caplog.text


def test_registrar_is_built_from_settings_and_pod_id(monkeypatch):
    monkeypatch.setenv("RUNPOD_POD_ID", "pod42")
    item = GatewayEntrypoint.registrar(
        settings(
            HEAR_BACKEND_INTERNAL_URL="https://api.hear.test/api/v1",
            HEAR_BACKEND_SERVICE_KEY="svc",
            HEAR_WORKER_GENERATION="g1",
        ),
        FakeRuntime(),
        {WorkerRole.PIPELINE, WorkerRole.RECONSTRUCTION},
    )
    assert item is not None
    assert item._gateway_url == "https://pod42-8000.proxy.runpod.net"
    assert item._identity["roles"] == ["pipeline", "reconstruction"]
    assert item._identity["worker_generation"] == "g1"


def test_zero_interval_disables_registration(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.setenv("RUNPOD_POD_ID", "pod42")
    assert (
        GatewayEntrypoint.registrar(
            settings(HEAR_REGISTRATION_INTERVAL_SECONDS="0"), FakeRuntime(), {WorkerRole.PIPELINE}
        )
        is None
    )
    assert "HEAR_REGISTRATION_INTERVAL_SECONDS=0" in caplog.text
