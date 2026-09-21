"""Immutable attempt bundles; upload the terminal marker last.

The injected store must enforce create-only writes and scoped object keys. The
existing legacy B2 uploader does not satisfy that contract and is not used here.
The backend owns reconciliation, current-fence checks and candidate approval.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Protocol

from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.services.magic_clean.contracts import (
    ArtifactIdentity,
    AttemptTicket,
    CleanExecutionError,
    CleanResultManifest,
    ErrorCode,
    ValidationSummary,
)


@dataclass(frozen=True)
class StoredObject:
    key: str
    version: str
    sha256: str
    size_bytes: int


class ImmutableArtifactStore(Protocol):
    """All calls must honor guard cancellation and deadline.

    Implementations atomically create, never overwrite, and verify remote bytes.
    An existing object may be returned only if it has exactly the expected
    checksum and size. A metadata-only HEAD is not byte verification.
    """

    def create(
        self,
        key: str,
        source: BinaryIO,
        *,
        size_bytes: int,
        sha256: str,
        content_type: str,
        guard: ResourceGuard,
    ) -> StoredObject: ...


@dataclass(frozen=True)
class LocalArtifact:
    role: str
    path: Path


@dataclass(frozen=True)
class PublishedBundle:
    manifest: CleanResultManifest
    reference: StoredObject


class ArtifactWriter:
    FORMATS = {
        "cleaned_master": ("cleaned_master.flac", "audio/flac"),
        "delivery_audio": ("delivery_audio.mp3", "audio/mpeg"),
        "comparison_source": ("comparison_source.flac", "audio/flac"),
        "validation_report": ("validation_report.json", "application/json"),
    }
    MAX_MANIFEST_BYTES = 256 * 1024

    def __init__(self, store: ImmutableArtifactStore):
        self.store = store

    @staticmethod
    def canonical_bytes(value: dict) -> bytes:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("utf-8")

    @staticmethod
    def fingerprint(source: BinaryIO, guard: ResourceGuard) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        source.seek(0)
        while chunk := source.read(1024 * 1024):
            guard.check()
            digest.update(chunk)
            size += len(chunk)
            if size > guard.budget.scratch_bytes:
                raise CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "artifact exceeds budget")
        source.seek(0)
        return digest.hexdigest(), size

    @staticmethod
    def verify_receipt(receipt: StoredObject, key: str, digest: str, size: int) -> None:
        if (
            receipt.key != key
            or receipt.sha256 != digest
            or receipt.size_bytes != size
            or not receipt.version
        ):
            raise CleanExecutionError(ErrorCode.STORAGE_FAILED, "artifact receipt mismatch")

    def publish(
        self,
        ticket: AttemptTicket,
        artifacts: tuple[LocalArtifact, ...],
        validation: ValidationSummary,
        guard: ResourceGuard,
    ) -> PublishedBundle:
        guard.bind_deadline(ticket.deadline)
        guard.check()
        if datetime.now(UTC) >= ticket.deadline:
            raise CleanExecutionError(ErrorCode.DEADLINE_EXCEEDED, "attempt deadline exceeded")
        roles = [artifact.role for artifact in artifacts]
        required = {"cleaned_master", "delivery_audio", "validation_report"}
        if len(set(roles)) != len(roles) or not required.issubset(roles):
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "invalid artifact role set")
        if any(role not in self.FORMATS for role in roles):
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "unsupported artifact role")
        if validation.hard_integrity != "passed" or validation.wanted_content not in (
            "passed",
            "review_required",
        ):
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "candidate failed validation")
        identities = []
        for artifact in artifacts:
            guard.check()
            if datetime.now(UTC) >= ticket.deadline:
                raise CleanExecutionError(ErrorCode.DEADLINE_EXCEEDED, "attempt deadline exceeded")
            if (
                artifact.path.is_symlink()
                or not artifact.path.is_file()
                or not artifact.path.resolve().is_relative_to(guard.workspace.resolve())
            ):
                raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "artifact outside workspace")
            filename, media_type = self.FORMATS[artifact.role]
            key = f"{ticket.artifact_prefix}/{filename}"
            with artifact.path.open("rb") as source:
                digest, size = self.fingerprint(source, guard)
                if not size:
                    raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "empty artifact")
                receipt = self.store.create(
                    key,
                    source,
                    size_bytes=size,
                    sha256=digest,
                    content_type=media_type,
                    guard=guard,
                )
                self.verify_receipt(receipt, key, digest, size)
                if self.fingerprint(source, guard) != (digest, size):
                    raise CleanExecutionError(
                        ErrorCode.ARTIFACT_CONFLICT, "artifact changed during upload"
                    )
            identities.append(
                ArtifactIdentity(
                    role=artifact.role,
                    object_key=key,
                    object_version=receipt.version,
                    sha256=digest,
                    size_bytes=size,
                    content_type=media_type,
                )
            )
        if datetime.now(UTC) >= ticket.deadline:
            raise CleanExecutionError(ErrorCode.DEADLINE_EXCEEDED, "attempt deadline exceeded")
        manifest = CleanResultManifest(
            contract_version="hear.cleaner.result.v2",
            backend_id=ticket.backend_id,
            tenant_scope=ticket.tenant_scope,
            job_id=ticket.job_id,
            attempt_id=ticket.attempt_id,
            fence=ticket.fence,
            purpose=ticket.purpose,
            source=ticket.input,
            expected_active_audio_revision=ticket.expected_active_audio_revision,
            plan=ticket.plan,
            plan_sha256=hashlib.sha256(
                self.canonical_bytes(ticket.plan.model_dump(mode="json"))
            ).hexdigest(),
            sample=ticket.sample,
            deadline=ticket.deadline,
            completed_at=datetime.now(UTC),
            outcome="succeeded",
            error_code=None,
            validation=validation,
            artifacts=tuple(identities),
            timing="sample_identity" if ticket.sample else "identity",
            correlation_id=ticket.correlation_id,
        )
        return self._publish_manifest(ticket, manifest, guard)

    def publish_failure(
        self,
        ticket: AttemptTicket,
        error_code: ErrorCode,
        guard: ResourceGuard,
    ) -> PublishedBundle:
        """Publish a typed terminal failure under still-live ingress authorization.

        The caller must revalidate authorization/fence first. This method never
        clears cancellation or extends a deadline to force a terminal upload.
        If cancellation/deadline prevents publication, backend reconciliation
        remains responsible for terminal state. No raw exception text is stored.
        """
        guard.bind_deadline(ticket.deadline)
        guard.check()
        if not isinstance(error_code, ErrorCode):
            raise ValueError("failure requires a typed error code")
        manifest = CleanResultManifest(
            contract_version="hear.cleaner.result.v2",
            backend_id=ticket.backend_id,
            tenant_scope=ticket.tenant_scope,
            job_id=ticket.job_id,
            attempt_id=ticket.attempt_id,
            fence=ticket.fence,
            purpose=ticket.purpose,
            source=ticket.input,
            expected_active_audio_revision=ticket.expected_active_audio_revision,
            plan=ticket.plan,
            plan_sha256=hashlib.sha256(
                self.canonical_bytes(ticket.plan.model_dump(mode="json"))
            ).hexdigest(),
            sample=ticket.sample,
            deadline=ticket.deadline,
            completed_at=datetime.now(UTC),
            outcome="cancelled" if error_code == ErrorCode.CANCELLED else "failed",
            error_code=error_code,
            validation=ValidationSummary(
                hard_integrity="not_applicable",
                wanted_content="not_applicable",
                warning_codes=(),
            ),
            artifacts=(),
            timing="sample_identity" if ticket.sample else "identity",
            correlation_id=ticket.correlation_id,
        )
        return self._publish_manifest(ticket, manifest, guard)

    def _publish_manifest(
        self, ticket: AttemptTicket, manifest: CleanResultManifest, guard: ResourceGuard
    ) -> PublishedBundle:
        payload = self.canonical_bytes(manifest.model_dump(mode="json"))
        if len(payload) > self.MAX_MANIFEST_BYTES:
            raise CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "manifest exceeds size limit")
        guard.check()
        if datetime.now(UTC) >= ticket.deadline:
            raise CleanExecutionError(ErrorCode.DEADLINE_EXCEEDED, "attempt deadline exceeded")
        # The bounded terminal document need not touch disk. This also prevents
        # local-file mutation between hashing and streaming the terminal marker.
        digest = hashlib.sha256(payload).hexdigest()
        with BytesIO(payload) as source:
            receipt = self.store.create(
                ticket.manifest_key,
                source,
                size_bytes=len(payload),
                sha256=digest,
                content_type="application/json",
                guard=guard,
            )
        self.verify_receipt(receipt, ticket.manifest_key, digest, len(payload))
        return PublishedBundle(manifest, receipt)


class ManifestVerifier:
    """Bounded semantic verification against an already authenticated attempt.

    The caller must additionally check the current backend fence/review state
    and verify each referenced object's bytes and version before registering it.
    """

    @staticmethod
    def verify(payload: bytes, ticket: AttemptTicket, expected_sha256: str) -> CleanResultManifest:
        if len(payload) > ArtifactWriter.MAX_MANIFEST_BYTES:
            raise CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "manifest exceeds size limit")
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise CleanExecutionError(ErrorCode.ARTIFACT_CONFLICT, "manifest checksum mismatch")
        try:
            manifest = CleanResultManifest.model_validate_json(payload)
        except ValueError as exc:
            raise CleanExecutionError(ErrorCode.INVALID_AUDIO, "invalid result manifest") from exc
        names = (
            "backend_id",
            "tenant_scope",
            "job_id",
            "attempt_id",
            "fence",
            "purpose",
            "expected_active_audio_revision",
            "sample",
            "deadline",
            "correlation_id",
            "plan",
        )
        if any(getattr(manifest, name) != getattr(ticket, name) for name in names):
            raise CleanExecutionError(ErrorCode.ARTIFACT_CONFLICT, "manifest attempt mismatch")
        if manifest.source != ticket.input:
            raise CleanExecutionError(ErrorCode.SOURCE_MISMATCH, "manifest source mismatch")
        plan_digest = hashlib.sha256(
            ArtifactWriter.canonical_bytes(ticket.plan.model_dump(mode="json"))
        ).hexdigest()
        if manifest.plan_sha256 != plan_digest:
            raise CleanExecutionError(
                ErrorCode.ARTIFACT_CONFLICT, "manifest plan checksum mismatch"
            )
        for artifact in manifest.artifacts:
            expected = ArtifactWriter.FORMATS.get(artifact.role)
            if (
                expected is None
                or artifact.object_key != f"{ticket.artifact_prefix}/{expected[0]}"
                or artifact.content_type != expected[1]
            ):
                raise CleanExecutionError(
                    ErrorCode.ARTIFACT_CONFLICT, "artifact destination mismatch"
                )
        return manifest
