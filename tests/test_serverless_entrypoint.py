import asyncio

import pytest

import hear.entrypoints.serverless as entrypoint
from hear.runtime.gpu_idle import AsyncIdleResource
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


@pytest.mark.parametrize("fails", [False, True])
def test_serverless_preloads_before_serving_and_closes_on_failure(monkeypatch, fails):
    calls = []

    class Resource(FakeResource):
        async def warmup(self):
            calls.append("warmup")
            if fails:
                raise RuntimeError("model_load_failed")

    resource = Resource()
    bootstrap = FakeBootstrap(True, resource)

    class Runtime:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            calls.append("serve")

    monkeypatch.setenv("HEAR_WORKER_ROLE", "transcription")
    monkeypatch.setenv("HEAR_SERVERLESS_PRELOAD_MODELS", "true")
    monkeypatch.setenv("HEAR_GPU_IDLE_EVICTION_ENABLED", "false")
    monkeypatch.setattr(entrypoint, "ServerlessRuntime", Runtime)
    runtime = entrypoint.ServerlessEntrypoint(bootstrap)
    if fails:
        with pytest.raises(RuntimeError, match="model_load_failed"):
            runtime.run()
        assert calls == ["warmup"]
    else:
        runtime.run()
        assert calls == ["warmup", "serve"]
    assert resource.closed


def test_serverless_preload_rejects_idle_eviction_before_allocating(monkeypatch):
    class Bootstrap:
        def executor_for(self, role):
            pytest.fail("must reject configuration before model allocation")

    monkeypatch.setenv("HEAR_SERVERLESS_PRELOAD_MODELS", "true")
    monkeypatch.setenv("HEAR_GPU_IDLE_EVICTION_ENABLED", "true")
    with pytest.raises(RuntimeError, match="serverless_preload_requires_idle_eviction_disabled"):
        entrypoint.ServerlessEntrypoint(Bootstrap()).run()


def test_preloaded_model_survives_startup_loop_and_is_reused_by_jobs():
    loads = []

    async def load():
        model = object()
        loads.append(model)
        return model

    async def close(model):
        pass

    resource = AsyncIdleResource(
        "serverless-model", load, close, idle_seconds=1, eviction_enabled=False
    )

    async def borrow():
        model = await resource.acquire()
        await resource.release()
        return model

    preloaded = asyncio.run(borrow())

    async def jobs():
        assert await borrow() is preloaded
        assert await borrow() is preloaded
        assert resource.snapshot["cold_starts"] == 1
        await resource.close()

    asyncio.run(jobs())
    assert loads == [preloaded]
