import hashlib
import json

import pytest

from hear.core.backend_registry import BackendRegistry


def _registration(secret: str, environment: str) -> dict:
    return {
        "environment": environment,
        "service_key_sha256": hashlib.sha256(secret.encode()).hexdigest(),
        "allowed_endpoint_urls": [f"https://{environment}.storage.test"],
        "allowed_buckets": [f"{environment}-bucket"],
        "allowed_public_base_urls": [f"https://{environment}.cdn.test"],
    }


def test_backend_registry_isolated_by_environment():
    raw = json.dumps(
        {
            "dev-backend": _registration("dev-secret", "development"),
            "prod-backend": _registration("prod-secret", "production"),
        }
    )
    development = BackendRegistry.parse_backend_registry(raw, environment="dev")
    production = BackendRegistry.parse_backend_registry(raw, environment="prod")

    assert set(development) == {"dev-backend"}
    assert set(production) == {"prod-backend"}


def test_backend_registry_rejects_an_environment_without_credentials():
    raw = json.dumps({"prod-backend": _registration("prod-secret", "production")})
    with pytest.raises(RuntimeError, match="no backend registrations"):
        BackendRegistry.parse_backend_registry(raw, environment="development")


def test_environment_scoping_rejects_mixed_legacy_registrations():
    raw = json.dumps(
        {
            "dev-backend": _registration("dev-secret", "development"),
            "legacy": {
                "service_key_sha256": hashlib.sha256(b"legacy-secret").hexdigest(),
                "allowed_endpoint_urls": ["https://legacy.storage.test"],
                "allowed_buckets": ["legacy-bucket"],
                "allowed_public_base_urls": ["https://legacy.cdn.test"],
            },
        }
    )
    with pytest.raises(RuntimeError, match="every backend registration"):
        BackendRegistry.parse_backend_registry(raw, environment="development")
