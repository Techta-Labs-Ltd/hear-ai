import pytest

import hear.entrypoints.serverless as entrypoint
from hear.runtime.roles import WorkerRole


class FakeReadiness:
    def __init__(self, ready: bool):
        self.ready = ready

    def is_ready(self):
        return self.ready


class FakeResource:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class FakeBootstrap:
    def __init__(self, ready: bool, resource: FakeResource):
        self._readiness = FakeReadiness(ready)
        self._resource = resource

    def executor_for(self, role):
        assert role == WorkerRole.TRANSCRIPTION
        return object(), object(), [self._resource]

    def readiness(self, role):
        assert role == WorkerRole.TRANSCRIPTION
        return self._readiness


@pytest.mark.parametrize("ready", [False, True])
def test_serverless_entrypoint_checks_final_readiness_and_closes_resources(
    monkeypatch,
    ready,
):
    started = []
    resource = FakeResource()
    bootstrap = FakeBootstrap(ready, resource)

    class FakeServerlessRuntime:
        def __init__(self, role, executor, backend, *, readiness, max_concurrent_jobs):
            assert role == WorkerRole.TRANSCRIPTION
            assert readiness is bootstrap._readiness
            assert max_concurrent_jobs == 1

        def start(self):
            started.append(True)

    monkeypatch.setenv("HEAR_WORKER_ROLE", WorkerRole.TRANSCRIPTION.value)
    monkeypatch.setattr(entrypoint, "ServerlessRuntime", FakeServerlessRuntime)
    runtime = entrypoint.ServerlessEntrypoint(bootstrap)

    if ready:
        runtime.run()
        assert started == [True]
    else:
        with pytest.raises(RuntimeError, match="runtime_not_ready"):
            runtime.run()
        assert started == []

    assert resource.closed is True
