"""Deployment-owned environment boundaries; request fields cannot select another backend."""

import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlsplit

from hear.contracts.jobs import AttemptEnvelope


@dataclass(frozen=True)
class BackendOwnershipPolicy:
    backend_id: str
    backend_base_urls: tuple[str, ...]
    bucket_name: str
    storage_endpoint: str
    public_base_url: str
    source_hosts: tuple[str, ...]

    @staticmethod
    def normalized_url(value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("invalid_trusted_url")
        return value.rstrip("/")

    @classmethod
    def from_json(cls, payload: str):
        if not payload.strip():
            return None
        raw = json.loads(payload)
        expected = {
            "backend_id",
            "backend_base_urls",
            "bucket_name",
            "storage_endpoint",
            "public_base_url",
            "source_hosts",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("invalid_backend_ownership_policy")
        for name in ("backend_id", "bucket_name"):
            if not isinstance(raw[name], str) or not raw[name].strip():
                raise ValueError("invalid_backend_ownership_policy")
        for name in ("backend_base_urls", "source_hosts"):
            if (
                not isinstance(raw[name], list)
                or not raw[name]
                or any(not isinstance(x, str) or not x for x in raw[name])
            ):
                raise ValueError("invalid_backend_ownership_policy")
        return cls(
            raw["backend_id"],
            tuple(cls.normalized_url(x) for x in raw["backend_base_urls"]),
            raw["bucket_name"],
            cls.normalized_url(raw["storage_endpoint"]),
            cls.normalized_url(raw["public_base_url"]),
            tuple(raw["source_hosts"]),
        )

    def validate(self, envelope: AttemptEnvelope) -> None:
        if getattr(envelope, "backend_id", None) != self.backend_id:
            raise ValueError("backend_ownership_mismatch")
        if self.normalized_url(str(envelope.backend_base_url)) not in self.backend_base_urls:
            raise ValueError("backend_callback_origin_mismatch")
        storage = envelope.storage
        if (
            storage.bucket_name != self.bucket_name
            or self.normalized_url(str(storage.endpoint_url)) != self.storage_endpoint
        ):
            raise ValueError("storage_environment_mismatch")
        if self.normalized_url(str(storage.public_base_url)) != self.public_base_url:
            raise ValueError("storage_public_origin_mismatch")
        if storage.expires_at.utcoffset() is None or storage.expires_at <= datetime.now(UTC):
            raise ValueError("storage_grant_expired")
        source = urlsplit(str(envelope.source.url))
        if (
            source.scheme != "https"
            or source.hostname not in self.source_hosts
            or source.username
            or source.password
        ):
            raise ValueError("source_origin_not_allowed")
        if not envelope.source.file_sha256:
            raise ValueError("source_digest_required")
        parts = storage.folder_prefix.rstrip("/").split("/")
        if (
            len(parts) != 5
            or parts[0] not in ("localtns", "creators")
            or not parts[1]
            or parts[2:4] != ["audio", "jobs"]
            or parts[4] != envelope.job_id
        ):
            raise ValueError("storage_job_scope_mismatch")
        for value in (envelope.job_id, envelope.attempt_id, envelope.track_id, envelope.user_id):
            if (
                value in (".", "..")
                or any(c in value for c in "/\\")
                or any(ord(c) < 32 for c in value)
            ):
                raise ValueError("invalid_scoped_identifier")
        canonical = storage.folder_prefix.rstrip("/") + "/" + envelope.attempt_id
        if envelope.artifact_prefix.rstrip("/") != canonical:
            raise ValueError("artifact_prefix_mismatch")

    def require_reporting_origin(self, value: str) -> None:
        if self.normalized_url(value) not in self.backend_base_urls:
            raise ValueError("reporter_configuration_mismatch")


class BackendRegistry:
    """One trusted routing entry per environment; never use a request-selected URL."""

    def __init__(self, entries: dict, pipeline_backend_id: str | None = None):
        self.entries = entries
        self.pipeline_backend_id = pipeline_backend_id

    @classmethod
    def from_json(cls, payload: str, pipeline_backend_id: str | None = None):
        if len(payload.encode()) > 65536:
            raise ValueError("backend_registry_too_large")
        raw = json.loads(payload)
        if (
            not isinstance(raw, dict)
            or set(raw) != {"backends"}
            or not isinstance(raw["backends"], list)
            or not 1 <= len(raw["backends"]) <= 8
        ):
            raise ValueError("invalid_backend_registry")
        entries = {}
        for item in raw["backends"]:
            if not isinstance(item, dict) or set(item) != {
                "policy",
                "callback_base_url",
                "ingress_token_sha256",
            }:
                raise ValueError("invalid_backend_registry_entry")
            policy = BackendOwnershipPolicy.from_json(json.dumps(item["policy"]))
            if policy is None or policy.backend_id in entries:
                raise ValueError("duplicate_backend_registry_entry")
            callback = policy.normalized_url(item["callback_base_url"])
            policy.require_reporting_origin(callback)
            digest = item["ingress_token_sha256"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("invalid_backend_ingress_token_digest")
            entries[policy.backend_id] = (policy, callback, digest)
        if len({row[2] for row in entries.values()}) != len(entries):
            raise ValueError("backend_ingress_tokens_must_be_distinct")
        return cls(entries, pipeline_backend_id)

    def entry(self, envelope):
        entry = self.entries.get(envelope.backend_id)
        if entry is None:
            raise ValueError("backend_not_registered")
        return entry

    def authenticate(self, envelope, authorization: str | None):
        _, _, expected = self.entry(envelope)
        prefix = "Bearer "
        if not authorization or not authorization.startswith(prefix):
            raise ValueError("backend_authentication_failed")
        provided = hashlib.sha256(authorization[len(prefix) :].encode()).hexdigest()
        if not hmac.compare_digest(provided, expected):
            raise ValueError("backend_authentication_failed")

    def validate(self, envelope):
        policy, _, _ = self.entry(envelope)
        policy.validate(envelope)
        if (
            envelope.job_type.value == "pipeline"
            and envelope.backend_id != self.pipeline_backend_id
        ):
            raise ValueError("pipeline_catalog_owner_not_configured")

    def callback_url(self, envelope) -> str:
        self.validate(envelope)
        return self.entry(envelope)[1]

    def require_reporting_origin(self, value):
        if not any(
            BackendOwnershipPolicy.normalized_url(value) in row[0].backend_base_urls
            for row in self.entries.values()
        ):
            raise ValueError("reporter_configuration_mismatch")


class DeploymentOwnership:
    @staticmethod
    def load():
        registry = os.environ.get("HEAR_BACKEND_REGISTRY_JSON", "")
        policy = os.environ.get("HEAR_BACKEND_POLICY_JSON", "")
        if registry and policy:
            raise ValueError("configure_registry_or_single_backend_policy_not_both")
        if registry:
            return BackendRegistry.from_json(
                registry, os.environ.get("HEAR_PIPELINE_CATALOG_BACKEND_ID")
            )
        return BackendOwnershipPolicy.from_json(policy)
