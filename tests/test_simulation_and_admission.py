import hashlib
import json

import pytest

from hear.queue.topology import RabbitMQTopology
from hear.runtime.host_admission import HostJobAdmission
from hear.runtime.roles import WorkerRole
from hear.runtime.simulation import SimulationBoundary


def test_global_and_role_limits_are_both_enforced(tmp_path):
    path = tmp_path / "admission"
    cleaning_a = HostJobAdmission(path, 2, role="magic_clean_natural")
    cleaning_b = HostJobAdmission(path, 2, role="magic_clean_natural")
    speech = HostJobAdmission(path, 2, role="reconstruction")
    transcription = HostJobAdmission(path, 2, role="transcription")
    first = cleaning_a.try_acquire()
    assert first is not None
    assert cleaning_b.try_acquire() is None
    second = speech.try_acquire()
    assert second is not None
    assert transcription.try_acquire() is None
    first.close()
    third = transcription.try_acquire()
    assert third is not None
    second.close()
    third.close()
    last = cleaning_b.try_acquire()
    assert last is not None
    last.close()


def test_quorum_overflow_is_supported():
    topology = RabbitMQTopology()
    assert (
        topology.queue_arguments(topology.binding(WorkerRole.PIPELINE))["x-overflow"]
        == "reject-publish"
    )


def registry():
    return {
        "backends": [
            {
                "policy": {
                    "backend_id": "simulation-local",
                    "backend_base_urls": ["https://127.0.0.1:18081/api/v1"],
                    "storage_endpoint": "https://127.0.0.1:18081/s3",
                    "bucket_name": "hear-simulation-local",
                    "public_base_url": "https://127.0.0.1:18081/objects",
                    "source_hosts": ["127.0.0.1"],
                },
                "callback_base_url": "https://127.0.0.1:18081/api/v1",
                "ingress_token_sha256": hashlib.sha256(b"test-only").hexdigest(),
            }
        ]
    }


def test_simulation_is_explicit_and_loopback_only(monkeypatch):
    monkeypatch.setenv("HEAR_RUNTIME_MODE", "simulation")
    monkeypatch.delenv("HEAR_BACKEND_POLICY_JSON", raising=False)
    value = registry()
    monkeypatch.setenv("HEAR_BACKEND_REGISTRY_JSON", json.dumps(value))
    assert SimulationBoundary.enabled() is True
    value["backends"][0]["policy"]["source_hosts"] = ["cdn.hear.media"]
    monkeypatch.setenv("HEAR_BACKEND_REGISTRY_JSON", json.dumps(value))
    with pytest.raises(ValueError, match="simulation_source"):
        SimulationBoundary.enabled()


def test_simulation_rejects_a_real_backend_identity(monkeypatch):
    value = registry()
    value["backends"][0]["policy"]["backend_id"] = "backend-a"
    monkeypatch.setenv("HEAR_RUNTIME_MODE", "simulation")
    monkeypatch.delenv("HEAR_BACKEND_POLICY_JSON", raising=False)
    monkeypatch.setenv("HEAR_BACKEND_REGISTRY_JSON", json.dumps(value))
    with pytest.raises(ValueError, match="real_backend_identity"):
        SimulationBoundary.enabled()
