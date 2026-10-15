"""The Pod announces itself to the backend; a replaced Pod needs no config edit."""

import asyncio
import json
import logging

import httpx
import pytest

from hear.services.registration import WorkerRegistrar


def lanes_factory(status="ready"):
    async def lane_status():
        return {
            "pipeline": {"status": status, "consumers": 1, "queued": 0},
            "reconstruction": {"status": status, "consumers": 1, "queued": 2},
        }

    return lane_status


def registrar(handler, lanes=None, interval=30.0):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return WorkerRegistrar(
        backend_url="https://api.hear.test/api/v1/",
        service_key="svc-test",
        gateway_url="https://abc123-8000.proxy.runpod.net/",
        worker_id="runpod-abc123-gateway",
        worker_generation="gen-7",
        roles=["reconstruction", "pipeline"],
        image_revision="pod-0646ff8",
        engine_revision="0646ff8",
        lane_status=lanes or lanes_factory(),
        interval_seconds=interval,
        client=client,
    )


def test_gateway_url_comes_from_the_pod_id_unless_set_explicitly():
    assert WorkerRegistrar.derive_gateway_url({"RUNPOD_POD_ID": "6q0tm0g5hfm590"}) == (
        "https://6q0tm0g5hfm590-8000.proxy.runpod.net"
    )
    assert WorkerRegistrar.derive_gateway_url({"RUNPOD_POD_ID": "x", "HTTP_PORT": "9000"}) == (
        "https://x-9000.proxy.runpod.net"
    )
    assert (
        WorkerRegistrar.derive_gateway_url(
            {"RUNPOD_POD_ID": "x", "HEAR_PUBLIC_GATEWAY_URL": "https://gw.example.com/"}
        )
        == "https://gw.example.com"
    )
    assert WorkerRegistrar.derive_gateway_url({}) is None


def test_heartbeat_carries_identity_address_and_lane_status():
    received = []

    def handle(request):
        received.append(
            (request.url, request.headers.get("X-Service-Key"), json.loads(request.content))
        )
        return httpx.Response(204)

    item = registrar(handle)
    assert asyncio.run(item.send()) is True and item.sent == 1
    url, key, body = received[0]
    assert str(url) == "https://api.hear.test/api/v1/internal/ai/runtime/workers/heartbeat"
    assert key == "svc-test"
    assert body["transport"] == "pod" and body["status"] == "ready"
    assert body["gateway_url"] == "https://abc123-8000.proxy.runpod.net"
    assert body["capabilities_url"] == "https://abc123-8000.proxy.runpod.net/capabilities"
    assert body["roles"] == ["pipeline", "reconstruction"]
    assert body["worker_id"] == "runpod-abc123-gateway" and body["worker_generation"] == "gen-7"
    assert (
        body["image_revision"] == "pod-0646ff8" and body["lanes"]["reconstruction"]["queued"] == 2
    )
    assert body["interval_seconds"] == 30.0 and body["sent_at"].endswith("+00:00")


@pytest.mark.parametrize("lane,expected", [("loading", "loading"), ("draining", "draining")])
def test_status_follows_the_lanes(lane, expected):
    item = registrar(lambda request: httpx.Response(204), lanes=lanes_factory(lane))
    assert asyncio.run(item.payload())["status"] == expected


def test_backend_failures_are_counted_and_logged_once_per_state(caplog):
    caplog.set_level(logging.INFO)
    responses = iter([httpx.Response(503), httpx.Response(503), httpx.Response(204)])
    item = registrar(lambda request: next(responses))

    async def go():
        return [await item.send(), await item.send(), await item.send()]

    assert asyncio.run(go()) == [False, False, True]
    assert item.failures == 2 and item.sent == 1
    assert caplog.text.count("outcome=failed detail=http_503") == 1
    assert "outcome=sent detail=ready" in caplog.text


def test_connection_errors_do_not_raise():
    def handle(request):
        raise httpx.ConnectError("refused")

    item = registrar(handle)
    assert asyncio.run(item.send()) is False and item.failures == 1


def test_close_announces_offline_and_stops_the_loop():
    statuses = []

    def handle(request):
        statuses.append(json.loads(request.content)["status"])
        return httpx.Response(204)

    async def go():
        item = registrar(handle, interval=0.01)
        item.start()
        await asyncio.sleep(0.05)
        await item.close()
        return item

    item = asyncio.run(go())
    assert statuses[0] == "ready" and statuses[-1] == "offline" and len(statuses) >= 3
    assert item.sent >= 3


def test_interval_must_be_positive():
    with pytest.raises(ValueError, match="interval"):
        registrar(lambda request: httpx.Response(204), interval=0)
