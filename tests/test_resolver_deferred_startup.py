import asyncio
from unittest.mock import AsyncMock

import httpx

from hear.deployments.resolver import ResolverDeployment
from hear.resolver.index.builder import ShardBuilder
from hear.resolver.index.manager import DoubleBufferManager


def test_manager_startup_defers_taxonomy_load():
    async def scenario():
        manager = DoubleBufferManager()
        attempted = asyncio.Event()

        async def fail_load(_version):
            attempted.set()
            return False

        manager._load = AsyncMock(side_effect=fail_load)

        await manager.startup()

        assert manager._retry_task is not None
        assert not manager._retry_task.done()
        await asyncio.wait_for(attempted.wait(), timeout=1)
        manager._load.assert_awaited_once_with(None)
        manager._retry_task.cancel()
        try:
            await manager._retry_task
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())


def test_missing_version_pointer_does_not_scan_manifests(monkeypatch):
    requested_urls: list[str] = []

    class FakeResponse:
        def raise_for_status(self):
            raise httpx.HTTPStatusError(
                "not found",
                request=httpx.Request("GET", requested_urls[-1]),
                response=httpx.Response(404),
            )

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url):
            requested_urls.append(url)
            return FakeResponse()

    monkeypatch.setattr("hear.resolver.index.builder.httpx.AsyncClient", FakeClient)

    version = asyncio.run(ShardBuilder().discover_latest_version())

    assert version == 0
    assert len(requested_urls) == 1
    assert requested_urls[0].endswith("/version.json")


def test_health_check_allows_background_index_loading(monkeypatch):
    deployment_class = ResolverDeployment.func_or_class
    deployment = deployment_class.__new__(deployment_class)
    deployment._bootstrap_error = None
    monkeypatch.setattr(
        "hear.deployments.resolver.resolver_main.manager.is_ready",
        lambda: False,
    )

    asyncio.run(deployment.check_health())
