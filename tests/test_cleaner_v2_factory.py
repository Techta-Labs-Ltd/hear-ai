import json

import pytest

from hear.runtime.cleaner.factory import CleanerWorkerFactory
from hear.runtime.cleaner.model_registry import CertifiedRuntime
from hear.runtime.cleaner.worker_lease import WorkerLease
from hear.services.magic_clean.contracts import AttemptTicket, CleanExecutionError
from tests.test_cleaner_v2_artifacts import MemoryStore
from tests.test_cleaner_v2_contracts import ticket as ticket_fixture

ticket = ticket_fixture


@pytest.fixture
def configuration(tmp_path, ticket):
    runtime = AttemptTicket.model_validate_json(json.dumps(ticket)).plan.runtime
    return {
        "lock_directory": tmp_path,
        "lane": "gpu",
        "runtimes": (CertifiedRuntime(runtime, "c" * 64, 48000, 1000000, (48000,), (2,)),),
        "loaders": {"deepfilternet3": lambda: pytest.fail("must not allocate a model")},
        "readiness": {"deepfilternet3": lambda _: True},
        "store": MemoryStore(),
    }


def test_factory_does_not_probe_load_or_contact_store(configuration):
    configuration["readiness"] = {"deepfilternet3": lambda _: pytest.fail("must not probe")}
    with CleanerWorkerFactory.build(**configuration) as worker:
        assert worker.executor.artifacts.store is configuration["store"]
        assert configuration["store"].calls == []


def test_readiness_tracks_worker_retirement_and_close(configuration):
    worker = CleanerWorkerFactory.build(**configuration)
    try:
        assert [p["ready"] for p in worker.capabilities()["profiles"]] == [True, False, False]
        worker.executor.worker_lease.mark_unhealthy()
        profile = worker.capabilities()["profiles"][0]
        assert not profile["ready"]
        assert profile["reason"] == "worker_unavailable"
        with pytest.raises(CleanExecutionError):
            WorkerLease(configuration["lock_directory"], "gpu")
    finally:
        worker.close()
    assert not worker.capabilities()["profiles"][0]["ready"]
    with WorkerLease(configuration["lock_directory"], "gpu"):
        pass


def test_close_refuses_active_attempt_and_can_retry_after_completion(configuration):
    worker = CleanerWorkerFactory.build(**configuration)
    try:
        with worker.executor.worker_lease.attempt("gpu"):
            with pytest.raises(CleanExecutionError):
                worker.close()
            assert worker.capabilities()["profiles"][0]["ready"]
    finally:
        worker.close()
    worker.close()


def test_mixed_lane_configuration_rejected_before_ownership(configuration):
    configuration["lane"] = "cpu"
    with pytest.raises(ValueError, match="different worker lane"):
        CleanerWorkerFactory.build(**configuration)
    assert not (configuration["lock_directory"] / "cleaner-cpu.lock").exists()


def test_assembly_failure_releases_ownership(configuration, monkeypatch):
    def fail(*args):
        raise RuntimeError("injected assembly failure")

    monkeypatch.setattr("hear.runtime.cleaner.factory.AudioMasteringService", fail)
    with pytest.raises(RuntimeError, match="injected assembly"):
        CleanerWorkerFactory.build(**configuration)
    with WorkerLease(configuration["lock_directory"], "gpu"):
        pass


def test_empty_cpu_worker_does_not_advertise_uncertified_profiles(tmp_path):
    with CleanerWorkerFactory.build(
        lock_directory=tmp_path,
        lane="cpu",
        runtimes=(),
        loaders={},
        readiness={},
        store=MemoryStore(),
    ) as worker:
        assert all(p["reason"] == "not_certified" for p in worker.capabilities()["profiles"])
