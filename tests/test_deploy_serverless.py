import json

import httpx
import pytest

from scripts.deploy_serverless import RunPodServerlessDeployer

DIGEST = "ghcr.io/techta-labs-ltd/hear-ai@sha256:" + "a" * 64
PLAN = {
    "template": {
        "name": "hear-ai-pipeline",
        "env": {"HEAR_WORKER_ROLE": "pipeline"},
        "containerDiskInGb": 50,
    },
    "endpoint": {
        "name": "hear-ai-pipeline",
        "computeType": "GPU",
        "gpuTypeIds": ["NVIDIA A40"],
        "workersMax": 1,
    },
}


def deployer(responder, calls):
    def handle(request):
        calls.append(
            (
                request.method,
                request.url.path,
                json.loads(request.content) if request.content else None,
            )
        )
        return responder(request)

    client = httpx.Client(transport=httpx.MockTransport(handle))
    return RunPodServerlessDeployer(
        "rpa_test", rest_base="https://rest.test/v1", run_base="https://run.test/v2", client=client
    )


def test_deploy_updates_template_and_creates_missing_endpoint():
    calls = []

    def respond(request):
        if request.url.path == "/v1/endpoints" and request.method == "GET":
            return httpx.Response(200, json=[])
        if request.url.path == "/v1/endpoints":
            return httpx.Response(200, json={"id": "ep1", "name": "hear-ai-pipeline"})
        if request.url.path == "/v1/templates/tpl1":
            return httpx.Response(200, json={"id": "tpl1", "name": "hear-ai-pipeline"})
        if request.url.path == "/v2/ep1/health":
            return httpx.Response(200, json={"workers": {"idle": 0}})
        raise AssertionError(request.url.path)

    result = deployer(respond, calls).deploy(PLAN, "tpl1", DIGEST)
    assert result["endpoint_id"] == "ep1" and result["health"] == {"workers": {"idle": 0}}
    patch = next(body for method, path, body in calls if method == "PATCH")
    assert patch["imageName"] == DIGEST and patch["env"] == {"HEAR_WORKER_ROLE": "pipeline"}
    create = next(body for method, path, body in calls if method == "POST")
    assert create["templateId"] == "tpl1" and create["computeType"] == "GPU"


def test_existing_endpoint_is_patched_not_duplicated():
    calls = []

    def respond(request):
        if request.url.path == "/v1/endpoints" and request.method == "GET":
            return httpx.Response(200, json=[{"id": "ep9", "name": "hear-ai-pipeline"}])
        if request.url.path == "/v1/endpoints/ep9":
            return httpx.Response(200, json={"id": "ep9", "name": "hear-ai-pipeline"})
        if request.url.path == "/v1/templates/tpl1":
            return httpx.Response(200, json={"id": "tpl1"})
        if request.url.path == "/v2/ep9/health":
            return httpx.Response(200, json={})
        raise AssertionError(request.url.path)

    deployer(respond, calls).deploy(PLAN, "tpl1", DIGEST)
    assert [m for m, _, _ in calls] == ["PATCH", "GET", "PATCH", "GET"]
    assert "computeType" not in calls[2][2]


def test_unpinned_images_are_refused():
    with pytest.raises(ValueError, match="sha256"):
        RunPodServerlessDeployer.require_digest("ghcr.io/techta-labs-ltd/hear-ai:latest")


def test_missing_template_is_created_before_the_endpoint():
    calls = []

    def respond(request):
        if request.url.path == "/v1/templates" and request.method == "POST":
            return httpx.Response(200, json={"id": "tplnew", "name": "hear-ai-pipeline"})
        if request.url.path == "/v1/endpoints" and request.method == "GET":
            return httpx.Response(200, json=[])
        if request.url.path == "/v1/endpoints":
            return httpx.Response(200, json={"id": "ep2", "name": "hear-ai-pipeline"})
        if request.url.path == "/v2/ep2/health":
            return httpx.Response(200, json={})
        raise AssertionError(request.url.path)

    result = deployer(respond, calls).deploy(
        {**PLAN, "template": {**PLAN["template"], "containerRegistryAuthId": "auth1"}}, None, DIGEST
    )
    created = calls[0][2]
    assert calls[0][:2] == ("POST", "/v1/templates")
    assert created["isServerless"] is True and created["containerRegistryAuthId"] == "auth1"
    assert result["template_id"] == "tplnew" and result["endpoint_id"] == "ep2"
