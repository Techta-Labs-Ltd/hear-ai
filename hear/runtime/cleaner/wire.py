"""Bounded protobuf/semantic conversion. Decoding never authorizes execution."""

import json
from dataclasses import dataclass

from google.protobuf.json_format import ParseDict
from google.protobuf.message import DecodeError, Message

from hear.proto import cleaner_v2_pb2 as wire
from hear.services.magic_clean.artifacts import ArtifactWriter, ManifestVerifier, PublishedBundle
from hear.services.magic_clean.contracts import (
    AttemptTicket,
    CleanExecutionError,
    ErrorCode,
    StorageGrant,
    TerminalReference,
)


@dataclass(frozen=True)
class DecodedAttempt:
    ticket: AttemptTicket
    source_read_grant: StorageGrant
    artifact_write_grant: StorageGrant


class CleanerWireCodec:
    MAX_REQUEST_BYTES = 256 * 1024
    MAX_REFERENCE_BYTES = 16 * 1024
    NULLABLE = {
        "hear.cleaner.v2.RuntimeIdentity": ("checkpoint_sha256",),
        "hear.cleaner.v2.CleanPlan": (
            "attenuation_limit_db",
            "noise_reduction_db",
            "noise_reference",
            "prompt_sha256",
        ),
        "hear.cleaner.v2.AttemptTicket": ("sample",),
    }

    @classmethod
    def decode(cls, payload: bytes) -> DecodedAttempt:
        if not 0 < len(payload) <= cls.MAX_REQUEST_BYTES:
            raise CleanExecutionError(ErrorCode.RESOURCE_EXHAUSTED, "invalid request size")
        request = wire.ExecuteAttemptRequest()
        try:
            request.ParseFromString(payload)
            known = wire.ExecuteAttemptRequest()
            known.CopyFrom(request)
            known.DiscardUnknownFields()
            if known.SerializeToString(deterministic=True) != request.SerializeToString(
                deterministic=True
            ):
                raise ValueError("unsupported wire fields")
            if not all(
                request.HasField(name)
                for name in ("ticket", "source_read_grant", "artifact_write_grant")
            ):
                raise ValueError("incomplete execution envelope")
            return DecodedAttempt(
                AttemptTicket.model_validate_json(json.dumps(cls._plain(request.ticket))),
                StorageGrant.model_validate_json(json.dumps(cls._plain(request.source_read_grant))),
                StorageGrant.model_validate_json(
                    json.dumps(cls._plain(request.artifact_write_grant))
                ),
            )
        except (DecodeError, ValueError, TypeError):
            # Validation errors may contain raw grant input. Do not chain them.
            raise CleanExecutionError(
                ErrorCode.INVALID_AUDIO, "invalid or unsupported v2 request"
            ) from None

    @classmethod
    def _plain(cls, message: Message) -> dict:
        # Native descriptor values preserve uint64 precision and optional bool
        # presence. ProtoJSON defaults must not turn omitted options into False.
        value = {}
        for field, item in message.ListFields():
            value[field.name] = cls._plain(item) if field.type == field.TYPE_MESSAGE else item
        for name in cls.NULLABLE.get(message.DESCRIPTOR.full_name, ()):
            value.setdefault(name, None)
        return value

    @classmethod
    def encode(cls, request: DecodedAttempt) -> bytes:
        message = wire.ExecuteAttemptRequest()
        ParseDict(request.ticket.model_dump(mode="json"), message.ticket)
        for grant, destination in (
            (request.source_read_grant, message.source_read_grant),
            (request.artifact_write_grant, message.artifact_write_grant),
        ):
            destination.reference = grant.reference
            destination.token = grant.token.get_secret_value()
            destination.expires_at = grant.expires_at.isoformat()
        payload = message.SerializeToString(deterministic=True)
        cls.decode(payload)
        return payload

    @classmethod
    def encode_result(cls, ticket: AttemptTicket, bundle: PublishedBundle) -> bytes:
        payload = ArtifactWriter.canonical_bytes(bundle.manifest.model_dump(mode="json"))
        manifest = ManifestVerifier.verify(payload, ticket, bundle.reference.sha256)
        if (
            bundle.reference.key != ticket.manifest_key
            or len(payload) != bundle.reference.size_bytes
        ):
            raise CleanExecutionError(ErrorCode.ARTIFACT_CONFLICT, "terminal receipt mismatch")
        reference = wire.ManifestReference(
            backend_id=manifest.backend_id,
            tenant_scope=manifest.tenant_scope,
            job_id=manifest.job_id,
            attempt_id=manifest.attempt_id,
            fence=manifest.fence,
            object_key=bundle.reference.key,
            object_version=bundle.reference.version,
            sha256=bundle.reference.sha256,
            size_bytes=bundle.reference.size_bytes,
            outcome=manifest.outcome,
        )
        if manifest.error_code is not None:
            reference.error_code = manifest.error_code.value
        encoded = reference.SerializeToString(deterministic=True)
        cls.decode_result(encoded, ticket)
        return encoded

    @classmethod
    def decode_result(cls, payload: bytes, ticket: AttemptTicket) -> TerminalReference:
        if not 0 < len(payload) <= cls.MAX_REFERENCE_BYTES:
            raise CleanExecutionError(
                ErrorCode.RESOURCE_EXHAUSTED, "invalid terminal reference size"
            )
        message = wire.ManifestReference()
        try:
            message.ParseFromString(payload)
            known = wire.ManifestReference()
            known.CopyFrom(message)
            known.DiscardUnknownFields()
            if known.SerializeToString(deterministic=True) != message.SerializeToString(
                deterministic=True
            ):
                raise ValueError("unsupported terminal fields")
            values = cls._plain(message)
            values.setdefault("error_code", None)
            reference = TerminalReference.model_validate_json(json.dumps(values))
        except (DecodeError, ValueError, TypeError):
            raise CleanExecutionError(
                ErrorCode.INVALID_AUDIO, "invalid v2 terminal reference"
            ) from None
        if (
            any(
                getattr(reference, name) != getattr(ticket, name)
                for name in ("backend_id", "tenant_scope", "job_id", "attempt_id", "fence")
            )
            or reference.object_key != ticket.manifest_key
        ):
            raise CleanExecutionError(ErrorCode.ARTIFACT_CONFLICT, "terminal attempt mismatch")
        return reference

    @classmethod
    def verify_result_bundle(
        cls, reference: TerminalReference, ticket: AttemptTicket, bundle: PublishedBundle
    ) -> None:
        expected = cls.decode_result(cls.encode_result(ticket, bundle), ticket)
        if reference != expected:
            raise CleanExecutionError(
                ErrorCode.ARTIFACT_CONFLICT, "terminal notice/bundle mismatch"
            )
