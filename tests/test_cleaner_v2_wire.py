import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr

from hear.proto import cleaner_v2_pb2 as wire
from hear.runtime.cleaner.wire import CleanerWireCodec, DecodedAttempt
from hear.services.magic_clean.artifacts import ArtifactWriter
from hear.services.magic_clean.contracts import (
    AttemptTicket,
    CleanExecutionError,
    ErrorCode,
    StorageGrant,
)
from tests.test_cleaner_v2_artifacts import MemoryStore, bundle
from tests.test_cleaner_v2_contracts import ticket as ticket_fixture

__all__ = ["bundle"]

ticket = ticket_fixture


@pytest.fixture
def envelope(ticket):
    ticket["plan"]["seed"] = 2**62 + 17
    grant = StorageGrant(
        reference="grant-1",
        token=SecretStr("private-token-never-log"),
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    return DecodedAttempt(AttemptTicket.model_validate_json(json.dumps(ticket)), grant, grant)


def test_complete_envelope_roundtrip_preserves_uint64_and_false_options(envelope):
    payload = CleanerWireCodec.encode(envelope)
    decoded = CleanerWireCodec.decode(payload)
    assert decoded == envelope
    assert decoded.ticket.plan.seed == 2**62 + 17
    message = wire.ExecuteAttemptRequest.FromString(payload)
    assert message.ticket.plan.HasField("shorten_pauses")
    assert not message.ticket.plan.shorten_pauses
    assert "private-token-never-log" not in repr(decoded)
    assert decoded.source_read_grant.token.get_secret_value() == "private-token-never-log"


@pytest.mark.parametrize(
    "field",
    ["shorten_pauses", "adjust_loudness", "mono_acknowledged", "seed", "match_comparison_loudness"],
)
def test_omitted_options_never_become_silent_defaults(envelope, field):
    message = wire.ExecuteAttemptRequest.FromString(CleanerWireCodec.encode(envelope))
    message.ticket.plan.ClearField(field)
    with pytest.raises(CleanExecutionError) as error:
        CleanerWireCodec.decode(message.SerializeToString())
    assert error.value.code == ErrorCode.INVALID_AUDIO


@pytest.mark.parametrize("nested", [False, True])
def test_unknown_fields_rejected_instead_of_dropped(envelope, nested):
    message = wire.ExecuteAttemptRequest.FromString(CleanerWireCodec.encode(envelope))
    target = message.ticket.plan if nested else message
    target.ParseFromString(target.SerializeToString() + b"\xa0\x06\x01")
    with pytest.raises(CleanExecutionError):
        CleanerWireCodec.decode(message.SerializeToString())


def test_missing_grant_and_unsupported_contract_rejected(envelope):
    message = wire.ExecuteAttemptRequest.FromString(CleanerWireCodec.encode(envelope))
    message.ClearField("source_read_grant")
    with pytest.raises(CleanExecutionError):
        CleanerWireCodec.decode(message.SerializeToString())
    message = wire.ExecuteAttemptRequest.FromString(CleanerWireCodec.encode(envelope))
    message.ticket.contract_version = "hear.cleaner.v99"
    with pytest.raises(CleanExecutionError):
        CleanerWireCodec.decode(message.SerializeToString())


def test_bad_grant_diagnostics_are_not_exposed(envelope):
    message = wire.ExecuteAttemptRequest.FromString(CleanerWireCodec.encode(envelope))
    message.source_read_grant.expires_at = "private-token-never-log"
    with pytest.raises(CleanExecutionError) as error:
        CleanerWireCodec.decode(message.SerializeToString())
    assert "private-token" not in str(error.value)
    assert error.value.__suppress_context__


@pytest.mark.parametrize("payload", [b"", b"x" * (256 * 1024 + 1)])
def test_wire_payload_bounded_before_parse(payload):
    with pytest.raises(CleanExecutionError) as error:
        CleanerWireCodec.decode(payload)
    assert error.value.code == ErrorCode.RESOURCE_EXHAUSTED


@pytest.mark.parametrize("code", [None, ErrorCode.PROCESS_FAILED, ErrorCode.CANCELLED])
def test_terminal_reference_roundtrips_verified_bundle(bundle, code):
    ticket, _, _, guard = bundle
    writer = ArtifactWriter(MemoryStore())
    published = (
        writer.publish(*bundle) if code is None else writer.publish_failure(ticket, code, guard)
    )
    payload = CleanerWireCodec.encode_result(ticket, published)
    reference = CleanerWireCodec.decode_result(payload, ticket)
    assert reference.error_code == code
    assert reference.sha256 == published.reference.sha256
    assert reference.object_version == published.reference.version
    CleanerWireCodec.verify_result_bundle(reference, ticket, published)
    assert "token" not in reference.model_dump()
    assert "plan" not in reference.model_dump()


@pytest.mark.parametrize(
    "field,value",
    [
        ("backend_id", "foreign"),
        ("tenant_scope", "foreign"),
        ("job_id", "foreign"),
        ("attempt_id", "foreign"),
        ("fence", 900),
        ("object_key", "foreign/manifest.json"),
    ],
)
def test_terminal_reference_cannot_rebind_an_attempt(bundle, field, value):
    published = ArtifactWriter(MemoryStore()).publish(*bundle)
    message = wire.ManifestReference.FromString(
        CleanerWireCodec.encode_result(bundle[0], published)
    )
    setattr(message, field, value)
    with pytest.raises(CleanExecutionError) as error:
        CleanerWireCodec.decode_result(message.SerializeToString(), bundle[0])
    assert error.value.code == ErrorCode.ARTIFACT_CONFLICT


def test_terminal_notice_cannot_contradict_verified_bundle(bundle):
    published = ArtifactWriter(MemoryStore()).publish(*bundle)
    message = wire.ManifestReference.FromString(
        CleanerWireCodec.encode_result(bundle[0], published)
    )
    message.outcome = "failed"
    message.error_code = "process_failed"
    reference = CleanerWireCodec.decode_result(message.SerializeToString(), bundle[0])
    with pytest.raises(CleanExecutionError) as error:
        CleanerWireCodec.verify_result_bundle(reference, bundle[0], published)
    assert error.value.code == ErrorCode.ARTIFACT_CONFLICT


def test_encode_result_rejects_corrupt_receipt(bundle):
    published = ArtifactWriter(MemoryStore()).publish(*bundle)
    wrong = replace(published, reference=replace(published.reference, sha256="b" * 64))
    with pytest.raises(CleanExecutionError):
        CleanerWireCodec.encode_result(bundle[0], wrong)


def test_terminal_unknown_fields_and_missing_outcome_rejected(bundle):
    published = ArtifactWriter(MemoryStore()).publish(*bundle)
    payload = CleanerWireCodec.encode_result(bundle[0], published)
    with pytest.raises(CleanExecutionError):
        CleanerWireCodec.decode_result(payload + b"\xa0\x06\x01", bundle[0])
    message = wire.ManifestReference.FromString(payload)
    message.ClearField("outcome")
    with pytest.raises(CleanExecutionError):
        CleanerWireCodec.decode_result(message.SerializeToString(), bundle[0])
