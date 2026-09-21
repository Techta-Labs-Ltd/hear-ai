from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from secrets import compare_digest
from typing import Any

from hear.config import settings
from hear.models.schemas import StorageContext


@dataclass(frozen=True)
class BackendRegistration:
    backend_id: str
    service_key_sha256: str
    environment: str | None
    allowed_endpoint_urls: frozenset[str]
    allowed_buckets: frozenset[str]
    allowed_public_base_urls: frozenset[str]


class BackendRegistry:
    @staticmethod
    def _normalize_environment(value: str | None) -> str:
        normalized = (value or "").strip().lower()
        aliases = {"dev": "development", "prod": "production"}
        return aliases.get(normalized, normalized)

    @staticmethod
    def _normalized_url(value: str) -> str:
        return value.strip().rstrip("/")

    @staticmethod
    def parse_backend_registry(
        raw_json: str, *, environment: str | None = None
    ) -> dict[str, BackendRegistration]:
        try:
            raw: Any = json.loads(raw_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("BACKEND_REGISTRY_JSON must be valid JSON") from exc
        if not isinstance(raw, dict) or not raw:
            raise RuntimeError("BACKEND_REGISTRY_JSON must define at least one backend")
        registrations: dict[str, BackendRegistration] = {}
        seen_hashes: set[str] = set()
        requested_environment = (
            BackendRegistry._normalize_environment(environment) if environment else None
        )
        if requested_environment and requested_environment not in {"development", "production"}:
            raise RuntimeError("ENVIRONMENT must be development or production")
        for raw_id, value in raw.items():
            backend_id = str(raw_id).strip()
            if not backend_id or not isinstance(value, dict):
                raise RuntimeError("backend registrations require a non-empty id and object value")
            digest = str(value.get("service_key_sha256") or "").strip().lower()
            registration_environment = value.get("environment")
            if registration_environment is not None:
                registration_environment = BackendRegistry._normalize_environment(
                    str(registration_environment)
                )
                if registration_environment not in {"development", "production"}:
                    raise RuntimeError(
                        f"backend {backend_id} has an invalid environment"
                    )
            endpoints = frozenset(
                BackendRegistry._normalized_url(str(item))
                for item in value.get("allowed_endpoint_urls") or []
            )
            buckets = frozenset(str(item).strip() for item in value.get("allowed_buckets") or [])
            public_urls = frozenset(
                BackendRegistry._normalized_url(str(item))
                for item in value.get("allowed_public_base_urls") or []
            )
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise RuntimeError(f"backend {backend_id} has an invalid service_key_sha256")
            if digest in seen_hashes:
                raise RuntimeError("backend service key hashes must be unique")
            if not endpoints or not buckets or (not public_urls):
                raise RuntimeError(
                    f"backend {backend_id} requires allowed endpoints, buckets, and public URLs"
                )
            seen_hashes.add(digest)
            registrations[backend_id] = BackendRegistration(
                backend_id=backend_id,
                service_key_sha256=digest,
                environment=registration_environment,
                allowed_endpoint_urls=endpoints,
                allowed_buckets=buckets,
                allowed_public_base_urls=public_urls,
            )
        if requested_environment is None:
            return registrations
        scoped_registrations = [
            registration
            for registration in registrations.values()
            if registration.environment is not None
        ]
        if scoped_registrations and any(
            registration.environment is None for registration in registrations.values()
        ):
            raise RuntimeError(
                "every backend registration must declare environment when "
                "environment scoping is enabled"
            )
        active = {
            backend_id: registration
            for backend_id, registration in registrations.items()
            if registration.environment in {None, requested_environment}
        }
        if not active:
            raise RuntimeError(
                f"BACKEND_REGISTRY_JSON has no backend registrations for {requested_environment}"
            )
        return active

    @staticmethod
    @lru_cache(maxsize=1)
    def backend_registry() -> dict[str, BackendRegistration]:
        return BackendRegistry.parse_backend_registry(
            settings.BACKEND_REGISTRY_JSON, environment=settings.ENVIRONMENT
        )

    @staticmethod
    def service_key_backend(service_key: str | None) -> str | None:
        if not service_key:
            return None
        digest = hashlib.sha256(service_key.encode("utf-8")).hexdigest()
        for backend_id, registration in BackendRegistry.backend_registry().items():
            if compare_digest(digest, registration.service_key_sha256):
                return backend_id
        return None

    @staticmethod
    def authenticate_backend(backend_id: str, service_key: str | None) -> bool:
        return bool(backend_id and BackendRegistry.service_key_backend(service_key) == backend_id)

    @staticmethod
    def validate_storage_for_backend(backend_id: str, storage: StorageContext) -> None:
        registration = BackendRegistry.backend_registry().get(backend_id)
        if registration is None:
            raise ValueError("unknown backend_id")
        if (
            BackendRegistry._normalized_url(storage.endpoint_url)
            not in registration.allowed_endpoint_urls
        ):
            raise ValueError("storage endpoint is not allowed for backend_id")
        if storage.bucket_name not in registration.allowed_buckets:
            raise ValueError("storage bucket is not allowed for backend_id")
        if (
            BackendRegistry._normalized_url(storage.public_base_url)
            not in registration.allowed_public_base_urls
        ):
            raise ValueError("storage public base URL is not allowed for backend_id")
        if storage.expires_at <= datetime.now(UTC):
            raise ValueError("storage credentials have expired")
