import hashlib
import json
import threading
import time
from datetime import UTC, datetime, timedelta

import pytest

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.services.magic_clean.artifacts import (
    ArtifactWriter,
    LocalArtifact,
    ManifestVerifier,
    StoredObject,
)
from hear.services.magic_clean.contracts import (
    AttemptTicket,
    CleanExecutionError,
    CleanResultManifest,
    ErrorCode,
    ValidationSummary,
)
from tests.test_cleaner_v2_contracts import ticket as ticket_fixture

ticket = ticket_fixture


class MemoryStore:
    def __init__(self, fail_at=None, bad_receipt=False):
        self.objects = {}
        self.calls = []
        self.fail_at = fail_at
        self.bad_receipt = bad_receipt

    def create(self, key, source, *, size_bytes, sha256, content_type, guard):
        guard.check()
        self.calls.append(key)
        if len(self.calls) == self.fail_at:
            raise CleanExecutionError(ErrorCode.STORAGE_FAILED, "injected storage interruption")
        data = source.read()
        assert len(data) == size_bytes
        assert hashlib.sha256(data).hexdigest() == sha256
        if key in self.objects and self.objects[key] != data:
            raise CleanExecutionError(ErrorCode.ARTIFACT_CONFLICT, "immutable key conflict")
        self.objects[key] = data
        return StoredObject(key, "version-1", "0" * 64 if self.bad_receipt else sha256, size_bytes)


@pytest.fixture
def bundle(tmp_path, ticket):
    ticket["deadline"] = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
    parsed = AttemptTicket.model_validate_json(json.dumps(ticket))
    guard = ResourceGuard(
        ResourceBudget(1024 * 1024, 1024 * 1024, 48000),
        tmp_path,
        time.monotonic() + 300,
        threading.Event(),
    )
    artifacts = []
    for role in ("cleaned_master", "delivery_audio", "validation_report"):
        path = tmp_path / role
        path.write_bytes(role.encode())
        artifacts.append(LocalArtifact(role, path))
    validation = ValidationSummary(
        hard_integrity="passed",
        wanted_content="review_required",
        warning_codes=("speech_loss_risk",),
    )
    return parsed, tuple(artifacts), validation, guard


def test_manifest_is_last_and_binds_attempt_source_and_plan(bundle):
    store = MemoryStore()
    result = ArtifactWriter(store).publish(*bundle)
    ticket, _, _, _ = bundle
    assert store.calls[-1] == ticket.manifest_key
    assert len(store.calls) == 4
    payload = store.objects[ticket.manifest_key]
    assert hashlib.sha256(payload).hexdigest() == result.reference.sha256
    parsed = CleanResultManifest.model_validate_json(payload)
    assert parsed.source == ticket.input
    assert parsed.plan == ticket.plan
    assert parsed.fence == ticket.fence
    assert parsed.attempt_id == ticket.attempt_id
    assert parsed.tenant_scope == ticket.tenant_scope
    assert parsed.validation.wanted_content == "review_required"
    assert "can_apply" not in json.loads(payload)
    for artifact in parsed.artifacts:
        assert hashlib.sha256(store.objects[artifact.object_key]).hexdigest() == artifact.sha256


@pytest.mark.parametrize("mutation", ["source_hash", "count", "channel", "interval"])
def test_manifest_rejects_foreign_or_out_of_bounds_speech_evidence(bundle, mutation):
    published = ArtifactWriter(MemoryStore()).publish(*bundle)
    raw = published.manifest.model_dump(mode="json")
    evidence = {
        "source_sha256": raw["source"]["sha256"],
        "output_sha256": "c" * 64,
        "analysis_sha256": "d" * 64,
        "comparison_sha256": "e" * 64,
        "source_active_frames": [10, 10],
        "output_active_frames": [0, 10],
        "source_loss_intervals": [{"channel": 0, "start_frame": 0, "end_frame": 10}],
        "evidence_truncated": False,
    }
    if mutation == "source_hash":
        evidence["source_sha256"] = "f" * 64
    elif mutation == "count":
        evidence["source_active_frames"] = [raw["source"]["frames"] + 1, 10]
    elif mutation == "channel":
        evidence["output_active_frames"] = [10]
    else:
        evidence["source_loss_intervals"][0]["end_frame"] = raw["source"]["frames"] + 1
    raw["validation"]["speech_activity"] = evidence
    raw["validation"]["warning_codes"].append("possible_speech_loss")
    with pytest.raises(ValueError):
        CleanResultManifest.model_validate_json(json.dumps(raw))


@pytest.mark.parametrize("fail_at", [1, 2, 3, 4])
def test_interrupted_bundle_has_no_terminal_marker(bundle, fail_at):
    store = MemoryStore(fail_at=fail_at)
    with pytest.raises(CleanExecutionError):
        ArtifactWriter(store).publish(*bundle)
    assert bundle[0].manifest_key not in store.objects


def test_wrong_remote_receipt_prevents_manifest(bundle):
    store = MemoryStore(bad_receipt=True)
    with pytest.raises(CleanExecutionError) as error:
        ArtifactWriter(store).publish(*bundle)
    assert error.value.code == ErrorCode.STORAGE_FAILED
    assert bundle[0].manifest_key not in store.objects


def test_cancelled_attempt_uploads_nothing(bundle):
    store = MemoryStore()
    bundle[3].cancelled.set()
    with pytest.raises(CleanExecutionError) as error:
        ArtifactWriter(store).publish(*bundle)
    assert error.value.code == ErrorCode.CANCELLED
    assert not store.objects


def test_failed_quality_cannot_publish_candidate(bundle):
    ticket, artifacts, _, guard = bundle
    store = MemoryStore()
    rejected = ValidationSummary(
        hard_integrity="rejected", wanted_content="rejected", warning_codes=()
    )
    with pytest.raises(CleanExecutionError):
        ArtifactWriter(store).publish(ticket, artifacts, rejected, guard)
    assert not store.objects


def test_manifest_retains_exact_preview_interval(bundle):
    ticket, artifacts, validation, guard = bundle
    payload = ticket.model_dump(mode="json")
    payload.update(purpose="sample_preview", sample={"start_frame": 12, "end_frame": 123})
    sample_ticket = AttemptTicket.model_validate_json(json.dumps(payload))
    result = ArtifactWriter(MemoryStore()).publish(sample_ticket, artifacts, validation, guard)
    assert result.manifest.sample.start_frame == 12
    assert result.manifest.timing == "sample_identity"


def test_immutable_conflict_cannot_become_success(bundle):
    store = MemoryStore()
    store.objects[bundle[0].artifact_prefix + "/cleaned_master.flac"] = b"existing other result"
    with pytest.raises(CleanExecutionError) as error:
        ArtifactWriter(store).publish(*bundle)
    assert error.value.code == ErrorCode.ARTIFACT_CONFLICT
    assert bundle[0].manifest_key not in store.objects


def test_verification_accepts_complete_expected_manifest(bundle):
    store = MemoryStore()
    result = ArtifactWriter(store).publish(*bundle)
    verified = ManifestVerifier.verify(
        store.objects[bundle[0].manifest_key], bundle[0], result.reference.sha256
    )
    assert verified == result.manifest


@pytest.mark.parametrize(
    "field,value",
    [
        ("attempt_id", "stale-attempt"),
        ("fence", 22),
        ("tenant_scope", "other-tenant"),
        ("plan_sha256", "b" * 64),
        ("expected_active_audio_revision", "other-revision"),
    ],
)
def test_verification_rejects_foreign_identity_even_with_valid_hash(bundle, field, value):
    store = MemoryStore()
    result = ArtifactWriter(store).publish(*bundle)
    changed = result.manifest.model_dump(mode="json")
    changed[field] = value
    payload = ArtifactWriter.canonical_bytes(changed)
    with pytest.raises(CleanExecutionError) as error:
        ManifestVerifier.verify(payload, bundle[0], hashlib.sha256(payload).hexdigest())
    assert error.value.code == ErrorCode.ARTIFACT_CONFLICT


def test_verification_rejects_source_substitution(bundle):
    result = ArtifactWriter(MemoryStore()).publish(*bundle)
    changed = result.manifest.model_dump(mode="json")
    changed["source"]["object_version"] = "other-version"
    payload = ArtifactWriter.canonical_bytes(changed)
    with pytest.raises(CleanExecutionError) as error:
        ManifestVerifier.verify(payload, bundle[0], hashlib.sha256(payload).hexdigest())
    assert error.value.code == ErrorCode.SOURCE_MISMATCH


def test_verification_rejects_artifacts_outside_attempt(bundle):
    result = ArtifactWriter(MemoryStore()).publish(*bundle)
    changed = result.manifest.model_dump(mode="json")
    changed["artifacts"][0]["object_key"] = "other-tenant/file.flac"
    payload = ArtifactWriter.canonical_bytes(changed)
    with pytest.raises(CleanExecutionError) as error:
        ManifestVerifier.verify(payload, bundle[0], hashlib.sha256(payload).hexdigest())
    assert error.value.code == ErrorCode.ARTIFACT_CONFLICT


def test_manifest_size_checked_before_parsing(bundle):
    with pytest.raises(CleanExecutionError) as error:
        ManifestVerifier.verify(b"x" * (ArtifactWriter.MAX_MANIFEST_BYTES + 1), bundle[0], "a" * 64)
    assert error.value.code == ErrorCode.RESOURCE_EXHAUSTED


def test_artifact_mutation_during_upload_has_no_manifest(bundle):
    class MutatingStore(MemoryStore):
        def create(self, key, source, **kwargs):
            receipt = super().create(key, source, **kwargs)
            bundle[1][0].path.write_bytes(b"changed")
            return receipt

    store = MutatingStore()
    with pytest.raises(CleanExecutionError) as error:
        ArtifactWriter(store).publish(*bundle)
    assert error.value.code == ErrorCode.ARTIFACT_CONFLICT
    assert bundle[0].manifest_key not in store.objects


@pytest.mark.parametrize("code", list(ErrorCode))
def test_failure_marker_binds_attempt_without_candidate_artifacts(bundle, code):
    ticket, _, _, guard = bundle
    store = MemoryStore()
    result = ArtifactWriter(store).publish_failure(ticket, code, guard)
    assert store.calls == [ticket.manifest_key]
    manifest = ManifestVerifier.verify(
        store.objects[ticket.manifest_key], ticket, result.reference.sha256
    )
    assert manifest.error_code == code
    assert manifest.outcome == ("cancelled" if code == ErrorCode.CANCELLED else "failed")
    assert manifest.artifacts == ()
    assert manifest.validation.hard_integrity == "not_applicable"
    assert manifest.validation.wanted_content == "not_applicable"


def test_failure_cannot_overwrite_success(bundle):
    ticket, _, _, guard = bundle
    store = MemoryStore()
    writer = ArtifactWriter(store)
    writer.publish(*bundle)
    original = store.objects[ticket.manifest_key]
    with pytest.raises(CleanExecutionError) as error:
        writer.publish_failure(ticket, ErrorCode.PROCESS_FAILED, guard)
    assert error.value.code == ErrorCode.ARTIFACT_CONFLICT
    assert store.objects[ticket.manifest_key] == original


def test_success_cannot_overwrite_failure(bundle):
    ticket, _, _, guard = bundle
    store = MemoryStore()
    writer = ArtifactWriter(store)
    writer.publish_failure(ticket, ErrorCode.PROCESS_FAILED, guard)
    original = store.objects[ticket.manifest_key]
    with pytest.raises(CleanExecutionError) as error:
        writer.publish(*bundle)
    assert error.value.code == ErrorCode.ARTIFACT_CONFLICT
    assert store.objects[ticket.manifest_key] == original


def test_failure_publication_never_clears_cancel_flag(bundle):
    ticket, _, _, guard = bundle
    guard.cancelled.set()
    store = MemoryStore()
    with pytest.raises(CleanExecutionError) as error:
        ArtifactWriter(store).publish_failure(ticket, ErrorCode.CANCELLED, guard)
    assert error.value.code == ErrorCode.CANCELLED
    assert guard.cancelled.is_set()
    assert not store.objects


def test_failure_publication_rejects_expired_ticket(bundle):
    ticket, _, _, guard = bundle
    raw = ticket.model_dump(mode="json")
    raw["deadline"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    expired = AttemptTicket.model_validate_json(json.dumps(raw))
    store = MemoryStore()
    with pytest.raises(CleanExecutionError) as error:
        ArtifactWriter(store).publish_failure(expired, ErrorCode.PROCESS_FAILED, guard)
    assert error.value.code == ErrorCode.DEADLINE_EXCEEDED
    assert not store.objects


@pytest.mark.parametrize(
    "changes",
    [
        {"outcome": "cancelled"},
        {"error_code": "cancelled"},
        {
            "validation": {
                "hard_integrity": "passed",
                "wanted_content": "passed",
                "warning_codes": [],
            }
        },
    ],
)
def test_verifier_rejects_incoherent_failure(bundle, changes):
    ticket, _, _, guard = bundle
    result = ArtifactWriter(MemoryStore()).publish_failure(ticket, ErrorCode.PROCESS_FAILED, guard)
    raw = result.manifest.model_dump(mode="json")
    raw.update(changes)
    payload = ArtifactWriter.canonical_bytes(raw)
    with pytest.raises(CleanExecutionError) as error:
        ManifestVerifier.verify(payload, ticket, hashlib.sha256(payload).hexdigest())
    assert error.value.code == ErrorCode.INVALID_AUDIO


def test_warning_intervals_cannot_escape_pinned_source(bundle):
    result = ArtifactWriter(MemoryStore()).publish(*bundle)
    raw = result.manifest.model_dump(mode="json")
    raw["validation"]["warning_codes"].append("possible_wanted_content_loss")
    raw["validation"]["source_warning_intervals"] = [
        {
            "start_frame": 0,
            "end_frame": bundle[0].input.frames + 1,
            "code": "possible_wanted_content_loss",
            "minimum_rms_ratio": 0.25,
        }
    ]
    payload = ArtifactWriter.canonical_bytes(raw)
    with pytest.raises(CleanExecutionError) as error:
        ManifestVerifier.verify(payload, bundle[0], hashlib.sha256(payload).hexdigest())
    assert error.value.code == ErrorCode.INVALID_AUDIO
