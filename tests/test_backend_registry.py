"""Trusted environment routing tests; no production credentials or cloud requests."""

import hashlib
import json
from dataclasses import asdict

import pytest

from hear.contracts.jobs import WorkerIdentity
from hear.execution.reporter import BackendAttemptClient
from hear.runtime.ownership import BackendRegistry
from hear.services.pipeline.catalog import PipelineCatalogClient
from tests.test_deployment_boundaries import owned, policy


def registry_json():
    prod = asdict(policy())
    prod["backend_base_urls"] = ["https://api.hear.media", "https://internal.hear.media"]
    dev = {
        **prod,
        "backend_id": "backend-a-dev",
        "backend_base_urls": ["https://api.hear.surf", "https://internal.hear.surf"],
        "bucket_name": "hear-dev-uploads",
        "storage_endpoint": "https://s3.us-east-005.backblazeb2.com",
        "public_base_url": "https://media.hear.surf",
        "source_hosts": ["media.hear.surf"],
    }
    return json.dumps(
        {
            "backends": [
                {
                    "policy": prod,
                    "callback_base_url": "https://internal.hear.media",
                    "ingress_token_sha256": hashlib.sha256(b"test-prod-only").hexdigest(),
                },
                {
                    "policy": dev,
                    "callback_base_url": "https://internal.hear.surf",
                    "ingress_token_sha256": hashlib.sha256(b"test-dev-only").hexdigest(),
                },
            ]
        }
    )


def dev_request():
    prod = owned()
    return prod.model_copy(
        update={
            "backend_id": "backend-a-dev",
            "backend_base_url": "https://api.hear.surf",
            "source": prod.source.model_copy(update={"url": "https://media.hear.surf/input.mp3"}),
            "storage": prod.storage.model_copy(
                update={
                    "bucket_name": "hear-dev-uploads",
                    "endpoint_url": "https://s3.us-east-005.backblazeb2.com",
                    "public_base_url": "https://media.hear.surf",
                }
            ),
        }
    )


def test_both_environments_have_distinct_tokens_buckets_and_callbacks():
    registry = BackendRegistry.from_json(registry_json())
    for envelope, token, origin in (
        (owned(), "test-prod-only", "https://internal.hear.media"),
        (dev_request(), "test-dev-only", "https://internal.hear.surf"),
    ):
        registry.authenticate(envelope, "Bearer " + token)
        registry.validate(envelope)
        assert registry.callback_url(envelope) == origin


def test_dev_token_cannot_submit_a_production_job():
    registry = BackendRegistry.from_json(registry_json())
    with pytest.raises(ValueError, match="authentication"):
        registry.authenticate(owned(), "Bearer test-dev-only")


def test_dev_job_cannot_use_production_bucket():
    registry = BackendRegistry.from_json(registry_json())
    dev = dev_request()
    with pytest.raises(ValueError, match="storage_environment"):
        registry.validate(
            dev.model_copy(
                update={"storage": dev.storage.model_copy(update={"bucket_name": "OldAlexa"})}
            )
        )


def test_reporter_uses_registered_callback_not_fixed_production_default(monkeypatch):
    monkeypatch.setenv("HEAR_BACKEND_REGISTRY_JSON", registry_json())
    monkeypatch.delenv("HEAR_BACKEND_POLICY_JSON", raising=False)
    worker = WorkerIdentity(
        worker_id="worker", generation="g", image_revision="i", engine_revision="e"
    )
    reporter = BackendAttemptClient(worker, "https://api.hear.media", client=object())
    assert reporter._url(dev_request(), "outcome").startswith(
        "https://internal.hear.surf/internal/ai/attempts/"
    )
    assert reporter._url(owned(), "outcome").startswith(
        "https://internal.hear.media/internal/ai/attempts/"
    )


def test_workspace_paths_are_partitioned_by_backend():
    assert owned().workspace_namespace != dev_request().workspace_namespace
    bad = owned().model_dump()
    bad["backend_id"] = "../production"
    with pytest.raises(ValueError):
        type(owned()).model_validate(bad)


def test_pipeline_cannot_use_another_backends_catalogue():
    from hear.contracts.jobs import JobType

    registry = BackendRegistry.from_json(registry_json(), pipeline_backend_id="backend-a")
    registry.validate(owned().model_copy(update={"job_type": JobType.PIPELINE}))
    with pytest.raises(ValueError, match="catalog_owner"):
        registry.validate(dev_request().model_copy(update={"job_type": JobType.PIPELINE}))


def test_pipeline_catalog_fetch_cannot_select_other_registered_backend(monkeypatch):
    monkeypatch.setenv("HEAR_BACKEND_REGISTRY_JSON", registry_json())
    monkeypatch.setenv("HEAR_PIPELINE_CATALOG_BACKEND_ID", "backend-a")
    monkeypatch.delenv("HEAR_BACKEND_POLICY_JSON", raising=False)
    PipelineCatalogClient("https://api.hear.media", "service")
    with pytest.raises(ValueError, match="reporter_configuration_mismatch"):
        PipelineCatalogClient("https://api.hear.surf", "service")


@pytest.mark.parametrize("kind", ["valid_dev", "dev_token_for_prod", "dev_with_prod_storage"])
def test_http_ingress_checks_registry_before_enqueue(kind):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from hear.api.gateway import PodGateway
    from hear.runtime.gateway import GatewayUnavailable

    calls = []

    class Queue:
        async def enqueue(self, envelope):
            calls.append(envelope)
            raise GatewayUnavailable("accepted_registry_test_no_real_queue")

    registry = BackendRegistry.from_json(registry_json())
    gateway = PodGateway(Queue(), "admin-test-token", ownership_policy=registry)
    app = FastAPI()
    app.include_router(gateway.router)
    value = dev_request() if kind != "dev_token_for_prod" else owned()
    if kind == "dev_with_prod_storage":
        value = value.model_copy(
            update={"storage": value.storage.model_copy(update={"bucket_name": "OldAlexa"})}
        )
    response = TestClient(app).post(
        "/v1/attempts/stream",
        json=value.model_dump(mode="json"),
        headers={"Authorization": "Bearer test-dev-only"},
    )
    if kind == "valid_dev":
        assert response.status_code == 503 and len(calls) == 1
    else:
        assert response.status_code == 403 and not calls
