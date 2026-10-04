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
        if request.url.path == "/v1/templates" and request.method == "GET":
            return httpx.Response(200, json=[])
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
    create_call = next(c for c in calls if c[:2] == ("POST", "/v1/templates"))
    created = create_call[2]
    assert calls[0][:2] == ("GET", "/v1/templates")
    assert created["isServerless"] is True and created["containerRegistryAuthId"] == "auth1"
    assert result["template_id"] == "tplnew" and result["endpoint_id"] == "ep2"


def test_secrets_are_injected_from_environment_not_plans():
    plan = RunPodServerlessDeployer.with_environment(
        PLAN, ["HEAR_BACKEND_SERVICE_KEY"], {"HEAR_BACKEND_SERVICE_KEY": "s3cret"}
    )
    assert plan["template"]["env"]["HEAR_BACKEND_SERVICE_KEY"] == "s3cret"
    assert "HEAR_BACKEND_SERVICE_KEY" not in PLAN["template"]["env"]
    with pytest.raises(ValueError, match="missing_deploy_environment:RUNPOD_X"):
        RunPodServerlessDeployer.with_environment(PLAN, ["RUNPOD_X"], {})


def test_committed_plans_hold_no_secrets_and_name_every_role():
    import json
    from pathlib import Path

    plans = {p.stem: json.loads(p.read_text()) for p in Path("deploy/runpod").glob("*.json")}
    assert set(plans) == {"pipeline", "pipeline-llm", "cleaner", "reconstruction"}
    for plan in plans.values():
        env = plan["template"]["env"]
        assert "HEAR_BACKEND_SERVICE_KEY" not in env and "HEAR_BACKEND_POLICY_JSON" not in env
        assert plan["endpoint"]["gpuTypeIds"] and plan["endpoint"]["workersMax"] >= 1


def test_template_is_found_by_name_before_creating():
    calls = []

    def respond(request):
        if request.url.path == "/v1/templates" and request.method == "GET":
            return httpx.Response(200, json=[{"id": "tplx", "name": "hear-ai-pipeline"}])
        if request.url.path == "/v1/templates/tplx":
            return httpx.Response(200, json={"id": "tplx"})
        if request.url.path == "/v1/endpoints" and request.method == "GET":
            return httpx.Response(200, json=[])
        if request.url.path == "/v1/endpoints":
            return httpx.Response(200, json={"id": "ep3", "name": "hear-ai-pipeline"})
        if request.url.path == "/v2/ep3/health":
            return httpx.Response(200, json={})
        raise AssertionError(request.url.path)

    result = deployer(respond, calls).deploy(PLAN, None, DIGEST)
    assert result["template_id"] == "tplx"
    assert not any(m == "POST" and p == "/v1/templates" for m, p, _ in calls)


SECRETS = {
    "HEAR_BACKEND_SERVICE_KEY": "service-key",
    "HEAR_BACKEND_POLICY_JSON": json.dumps(
        {
            "backend_id": "backend-a",
            "backend_base_urls": ["https://api.hear.media/api/v1"],
            "bucket_name": "bucket",
            "storage_endpoint": "https://s3.example.com",
            "public_base_url": "https://cdn.example.com",
            "source_hosts": ["cdn.example.com"],
        }
    ),
}


@pytest.mark.parametrize("role", ["pipeline", "pipeline-llm", "cleaner", "reconstruction"])
def test_every_committed_plan_passes_preflight_once_stamped(role):
    from pathlib import Path

    from scripts.deploy_serverless import PlanPreflight

    plan = json.loads(Path(f"deploy/runpod/{role}.json").read_text())
    plan = RunPodServerlessDeployer.with_environment(plan, list(SECRETS), SECRETS)
    stamped = PlanPreflight.stamp(plan, DIGEST)
    assert stamped["template"]["env"]["HEAR_ENGINE_REVISION"] == "a" * 32
    assert PlanPreflight.check(stamped, DIGEST) == []


def test_preflight_catches_settings_that_would_crash_workers():
    from scripts.deploy_serverless import PlanPreflight

    env = {
        "HEAR_WORKER_ROLE": "reconstruction",
        "HEAR_BACKEND_INTERNAL_URL": "https://other.example/api",
    }
    plan = {"template": {"env": {**env, **SECRETS}}, "endpoint": {}}
    problems = PlanPreflight.check(plan, DIGEST)
    assert "missing_runtime_setting:HEAR_ENGINE_REVISION" in problems
    assert "reconstruction_requires_HEAR_FISH_LICENSE_APPROVED" in problems
    assert any(p.startswith("invalid_backend_policy") for p in problems)
