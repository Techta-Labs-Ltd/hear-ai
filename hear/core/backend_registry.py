"""Deployment-controlled backend identity and storage allow-list validation."""

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
    allowed_endpoint_urls: frozenset[str]
    allowed_buckets: frozenset[str]
    allowed_public_base_urls: frozenset[str]


def _normalized_url(value: str) -> str:
    return value.strip().rstrip("/")


def parse_backend_registry(raw_json: str) -> dict[str, BackendRegistration]:
    try:
        raw: Any = json.loads(raw_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("BACKEND_REGISTRY_JSON must be valid JSON") from exc
    if not isinstance(raw, dict) or not raw:
        raise RuntimeError("BACKEND_REGISTRY_JSON must define at least one backend")
    registrations: dict[str, BackendRegistration] = {}
    seen_hashes: set[str] = set()
    for raw_id, value in raw.items():
        backend_id = str(raw_id).strip()
        if not backend_id or not isinstance(value, dict):
            raise RuntimeError("backend registrations require a non-empty id and object value")
        digest = str(value.get("service_key_sha256") or "").strip().lower()
        endpoints = frozenset(
            _normalized_url(str(item)) for item in value.get("allowed_endpoint_urls") or []
        )
        buckets = frozenset(str(item).strip() for item in value.get("allowed_buckets") or [])
        public_urls = frozenset(
            _normalized_url(str(item))
            for item in value.get("allowed_public_base_urls") or []
        )
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise RuntimeError(f"backend {backend_id} has an invalid service_key_sha256")
        if digest in seen_hashes:
            raise RuntimeError("backend service key hashes must be unique")
        if not endpoints or not buckets or not public_urls:
            raise RuntimeError(
                f"backend {backend_id} requires allowed endpoints, buckets, and public URLs"
            )
        seen_hashes.add(digest)
        registrations[backend_id] = BackendRegistration(
            backend_id=backend_id,
            service_key_sha256=digest,
            allowed_endpoint_urls=endpoints,
            allowed_buckets=buckets,
            allowed_public_base_urls=public_urls,
        )
    return registrations


@lru_cache(maxsize=1)
def backend_registry() -> dict[str, BackendRegistration]:
    return parse_backend_registry(settings.BACKEND_REGISTRY_JSON)


def service_key_backend(service_key: str | None) -> str | None:
    if not service_key:
        return None
    digest = hashlib.sha256(service_key.encode("utf-8")).hexdigest()
    for backend_id, registration in backend_registry().items():
        if compare_digest(digest, registration.service_key_sha256):
            return backend_id
    return None


def authenticate_backend(backend_id: str, service_key: str | None) -> bool:
    return bool(backend_id and service_key_backend(service_key) == backend_id)


def validate_storage_for_backend(backend_id: str, storage: StorageContext) -> None:
    registration = backend_registry().get(backend_id)
    if registration is None:
        raise ValueError("unknown backend_id")
    if _normalized_url(storage.endpoint_url) not in registration.allowed_endpoint_urls:
        raise ValueError("storage endpoint is not allowed for backend_id")
    if storage.bucket_name not in registration.allowed_buckets:
        raise ValueError("storage bucket is not allowed for backend_id")
    if _normalized_url(storage.public_base_url) not in registration.allowed_public_base_urls:
        raise ValueError("storage public base URL is not allowed for backend_id")
    if storage.expires_at <= datetime.now(UTC):
        raise ValueError("storage credentials have expired")
