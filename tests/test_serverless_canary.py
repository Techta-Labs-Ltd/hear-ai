import json

from hear.runtime.ownership import BackendOwnershipPolicy
from scripts.serverless_canary import ServerlessCanary

POLICY = {
    "backend_id": "backend-a",
    "backend_base_urls": ["https://api.example.com/api/v1"],
    "bucket_name": "bucket",
    "storage_endpoint": "https://s3.example.com",
    "public_base_url": "https://cdn.example.com",
    "source_hosts": ["cdn.example.com"],
}


def test_synthetic_canary_envelope_satisfies_deployment_policy():
    for job_type in ("pipeline", "magic_clean"):
        envelope = ServerlessCanary.synthetic_envelope(POLICY, job_type)
        BackendOwnershipPolicy.from_json(json.dumps(POLICY)).validate(envelope)
        assert envelope.job_type.value == job_type
        assert envelope.reporting_grant.startswith("canary-grant")
