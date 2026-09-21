import json
from types import SimpleNamespace

import pytest

from hear.runtime.cleaner.model_registry import CertifiedRuntime, EngineRegistry
from hear.services.magic_clean.contracts import AttemptTicket, CleanExecutionError, ErrorCode
from tests.test_cleaner_v2_contracts import ticket as ticket_fixture

ticket = ticket_fixture


@pytest.fixture
def certified(ticket):
    plan = AttemptTicket.model_validate_json(json.dumps(ticket)).plan
    return plan, CertifiedRuntime(plan.runtime, "c" * 64, 96000, 1000000, (16000, 48000), (1, 2))


def test_empty_registry_reports_all_profiles_unavailable():
    response = EngineRegistry((), {}).capabilities()
    assert response["contract_version"] == "hear.cleaner.capabilities.v2"
    assert len(response["profiles"]) == 3
    assert all(not item["ready"] for item in response["profiles"])
    assert all(item["reason"] == "not_certified" for item in response["profiles"])


def test_capabilities_do_not_load_models_and_isolate_optional_profiles(certified):
    plan, runtime = certified

    def loader():
        pytest.fail("capabilities must not allocate models")

    registry = EngineRegistry(
        (runtime,),
        {"deepfilternet3": loader},
        {"deepfilternet3": lambda identity: identity == plan.runtime},
    )
    profiles = registry.capabilities()["profiles"]
    assert [item["ready"] for item in profiles] == [True, False, False]
    assert profiles[0]["rate_limits"] == [
        {"sample_rate": 16000, "max_frames": 96000, "max_duration_seconds": 6.0},
        {"sample_rate": 48000, "max_frames": 96000, "max_duration_seconds": 2.0},
    ]
    assert profiles[0]["evidence_sha256"] == runtime.evidence_sha256


def test_missing_probe_fails_closed(certified):
    plan, runtime = certified
    registry = EngineRegistry((runtime,), {"deepfilternet3": lambda: pytest.fail("must not load")})
    assert not registry.capabilities()["profiles"][0]["ready"]
    with pytest.raises(CleanExecutionError) as error:
        registry.load(plan, frames=48000, size_bytes=100, sample_rate=48000, channels=2)
    assert error.value.code == ErrorCode.ENGINE_UNAVAILABLE


def test_probe_failure_is_sanitized_and_dispatch_rechecks(certified):
    plan, runtime = certified
    available = True

    def probe(identity):
        assert identity == runtime.identity
        if not available:
            raise RuntimeError("private-checkpoint-path-and-secret")
        return True

    registry = EngineRegistry(
        (runtime,),
        {"deepfilternet3": lambda: SimpleNamespace(identity=runtime.identity)},
        {"deepfilternet3": probe},
    )
    assert registry.capabilities()["profiles"][0]["ready"]
    available = False
    assert "private-checkpoint" not in json.dumps(registry.capabilities())
    assert not registry.capabilities()["profiles"][0]["ready"]
    with pytest.raises(CleanExecutionError) as error:
        registry.load(plan, frames=48000, size_bytes=100, sample_rate=48000, channels=2)
    assert error.value.code == ErrorCode.ENGINE_UNAVAILABLE


def test_registry_preserves_resource_failure_and_worker_restart_marker(certified):
    plan, runtime = certified

    def loader():
        raise CleanExecutionError(
            ErrorCode.RESOURCE_EXHAUSTED, "allocation failure", worker_restart_required=True
        )

    registry = EngineRegistry(
        (runtime,), {"deepfilternet3": loader}, {"deepfilternet3": lambda _: True}
    )
    with pytest.raises(CleanExecutionError) as error:
        registry.load(plan, frames=48000, size_bytes=100, sample_rate=48000, channels=2)
    assert error.value.code == ErrorCode.RESOURCE_EXHAUSTED
    assert error.value.worker_restart_required
