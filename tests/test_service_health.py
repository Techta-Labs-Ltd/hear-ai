import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hear.core.health import RayHealthSnapshot, ServiceHealth


@pytest.mark.anyio
async def test_health_reports_workers_without_gateway_gpu_metrics():
    reader = Mock(
        return_value={
            "control_ready": True,
            "capabilities": {"asr": {"state": "ready"}, "fish": {"state": "cold"}},
        }
    )
    health = ServiceHealth(reader)
    result = await health.read({"active": 2, "queued": 3})
    assert result["status"] == "healthy"
    assert result["models_loaded"] == ["asr"]
    assert result["capabilities"]["fish"]["state"] == "cold"
    assert result["gpu_metrics_available"] is False
    assert "gpu_available" not in result
    assert result["active_jobs"] == 2
    await health.read({})
    reader.assert_called_once()


@pytest.mark.anyio
async def test_optional_model_failure_does_not_fail_control_readiness():
    reader = Mock(
        return_value={
            "control_ready": True,
            "capabilities": {"fish": {"state": "unavailable"}},
        }
    )
    result = await ServiceHealth(reader).read({})
    assert result["status"] == "degraded"
    assert result["control_ready"] is True


@pytest.mark.anyio
async def test_probe_failure_is_explicit():
    result = await ServiceHealth(Mock(side_effect=RuntimeError("private detail"))).read({})
    assert result["status"] == "unavailable"
    assert result["error"] == "health_probe_failed"
    assert "private detail" not in str(result)


@pytest.mark.anyio
async def test_probe_timeout_reuses_inflight_task(monkeypatch):
    gate = asyncio.Event()
    calls = []

    async def blocking_read(reader):
        calls.append(reader)
        await gate.wait()
        return {"control_ready": True, "capabilities": {}}

    monkeypatch.setattr("hear.core.health.asyncio.to_thread", blocking_read)
    health = ServiceHealth(Mock(), timeout_seconds=0.01, cache_seconds=0)
    assert (await health.read({}))["error"] == "health_probe_timeout"
    assert (await health.read({}))["error"] == "health_probe_timeout"
    assert len(calls) == 1
    gate.set()
    assert (await health.read({}))["status"] == "healthy"


def test_ray_snapshot_distinguishes_cold_failed_warming_and_ready(monkeypatch):
    deployments = {
        "grpc_gateway": SimpleNamespace(status="HEALTHY", replica_states={"RUNNING": 1}),
        "cold": SimpleNamespace(status="HEALTHY", replica_states={}),
        "failed": SimpleNamespace(status="UNHEALTHY", replica_states={}),
        "warming": SimpleNamespace(status="UPDATING", replica_states={"STARTING": 1}),
        "ready": SimpleNamespace(status="HEALTHY", replica_states={"RUNNING": 1}),
    }
    monkeypatch.setattr(
        "hear.core.health.serve.status",
        lambda: SimpleNamespace(applications={"hear": SimpleNamespace(deployments=deployments)}),
    )
    result = RayHealthSnapshot("hear", ("cold", "failed", "warming", "ready", "off"))()
    assert result["control_ready"] is True
    assert {name: value["state"] for name, value in result["capabilities"].items()} == {
        "cold": "cold",
        "failed": "unavailable",
        "warming": "warming",
        "ready": "ready",
        "off": "disabled",
    }
