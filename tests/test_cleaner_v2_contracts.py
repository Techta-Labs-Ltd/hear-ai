import copy
import json

import pytest
from pydantic import ValidationError

from hear.runtime.cleaner.model_registry import CertifiedRuntime, EngineRegistry
from hear.services.magic_clean.contracts import AttemptTicket, CleanExecutionError, ErrorCode


@pytest.fixture
def ticket():
    digest = "a" * 64
    return {
        "contract_version": "hear.cleaner.v2",
        "backend_id": "backend-a",
        "tenant_scope": "tenant-1",
        "job_id": "job-1",
        "attempt_id": "attempt-1",
        "fence": 1,
        "provider": "pod",
        "purpose": "full_candidate",
        "input": {
            "revision_id": "revision-1",
            "media_id": "media-1",
            "object_key": "source.flac",
            "object_version": "object-v1",
            "sha256": digest,
            "size_bytes": 100,
            "sample_rate": 48000,
            "channels": 2,
            "frames": 48000,
        },
        "expected_active_audio_revision": "revision-1",
        "plan": {
            "profile": "natural",
            "profile_version": "v1",
            "catalogue_sha256": digest,
            "runtime": {
                "engine": "deepfilternet3",
                "runtime_sha256": digest,
                "checkpoint_sha256": digest,
                "precision_policy_sha256": digest,
                "longform_policy_sha256": digest,
            },
            "attenuation_limit_db": 18,
            "noise_reduction_db": None,
            "noise_reference": None,
            "prompt_sha256": None,
            "channel_policy": "preserve",
            "mono_acknowledged": False,
            "adjust_loudness": True,
            "match_comparison_loudness": True,
            "shorten_pauses": False,
            "seed": 42,
        },
        "sample": None,
        "artifact_prefix": "backend-a/job-1/attempt-1",
        "manifest_key": "backend-a/job-1/attempt-1/manifest.json",
        "deadline": "2026-09-22T00:00:00Z",
        "heartbeat_seconds": 10,
        "lease_seconds": 60,
        "correlation_id": "corr-1",
    }


def test_resolved_ticket_round_trip(ticket):
    parsed = AttemptTicket.model_validate_json(json.dumps(ticket))
    assert AttemptTicket.model_validate_json(parsed.model_dump_json()) == parsed


@pytest.mark.parametrize(
    "field,value",
    [
        ("contract_version", "v1"),
        ("fence", 0),
        ("fence", "1"),
        ("purpose", "sample_preview"),
        ("artifact_prefix", "../backend-a/job-1/attempt-1"),
        ("manifest_key", "another-attempt/manifest.json"),
        ("deadline", "2026-09-22T00:00:00"),
        ("lease_seconds", 5),
    ],
)
def test_invalid_attempts_rejected(ticket, field, value):
    ticket[field] = value
    with pytest.raises(ValidationError):
        AttemptTicket.model_validate_json(json.dumps(ticket))


def test_no_silent_option_defaults(ticket):
    for key in ticket["plan"]:
        invalid = copy.deepcopy(ticket)
        del invalid["plan"][key]
        with pytest.raises(ValidationError):
            AttemptTicket.model_validate_json(json.dumps(invalid))


def test_old_engine_and_unknown_options_rejected(ticket):
    ticket["plan"]["runtime"]["engine"] = "demucs"
    with pytest.raises(ValidationError):
        AttemptTicket.model_validate_json(json.dumps(ticket))
    ticket["plan"]["runtime"]["engine"] = "deepfilternet3"
    ticket["plan"]["strength"] = 0.5
    with pytest.raises(ValidationError):
        AttemptTicket.model_validate_json(json.dumps(ticket))


def test_sample_cannot_replace_full_candidate(ticket):
    ticket["sample"] = {"start_frame": 0, "end_frame": 200}
    with pytest.raises(ValidationError):
        AttemptTicket.model_validate_json(json.dumps(ticket))
    ticket["purpose"] = "sample_preview"
    assert AttemptTicket.model_validate_json(json.dumps(ticket)).sample.end_frame == 200
    ticket["sample"]["end_frame"] = 48001
    with pytest.raises(ValidationError):
        AttemptTicket.model_validate_json(json.dumps(ticket))


def test_voice_focus_cannot_silently_downmix_stereo(ticket):
    ticket["plan"].update(
        profile="voice_focus",
        attenuation_limit_db=None,
        prompt_sha256="b" * 64,
        channel_policy="mono",
    )
    ticket["plan"]["runtime"]["engine"] = "sam_audio_small"
    with pytest.raises(ValidationError):
        AttemptTicket.model_validate_json(json.dumps(ticket))
    ticket["plan"].update(channel_policy="validated_dual_mono", mono_acknowledged=True)
    AttemptTicket.model_validate_json(json.dumps(ticket))


def test_uncertified_engine_never_loads(ticket):
    plan = AttemptTicket.model_validate_json(json.dumps(ticket)).plan
    with pytest.raises(CleanExecutionError) as error:
        EngineRegistry((), {}).load(
            plan, frames=48000, size_bytes=100, sample_rate=48000, channels=2
        )
    assert error.value.code == ErrorCode.ENGINE_UNAVAILABLE


def test_runtime_mismatch_and_oversized_input_rejected_before_loading(ticket):
    plan = AttemptTicket.model_validate_json(json.dumps(ticket)).plan
    certification = CertifiedRuntime(plan.runtime, "c" * 64, 48000, 100, (48000,), (1, 2))

    def loader():
        pytest.fail("rejected request must never allocate a model")

    registry = EngineRegistry((certification,), {"deepfilternet3": loader})
    with pytest.raises(CleanExecutionError) as error:
        registry.load(plan, frames=48001, size_bytes=100, sample_rate=48000, channels=2)
    assert error.value.code == ErrorCode.RESOURCE_EXHAUSTED
    ticket["plan"]["runtime"]["checkpoint_sha256"] = "b" * 64
    mismatched_plan = AttemptTicket.model_validate_json(json.dumps(ticket)).plan
    with pytest.raises(CleanExecutionError) as error:
        registry.load(mismatched_plan, frames=48000, size_bytes=100, sample_rate=48000, channels=2)
    assert error.value.code == ErrorCode.ENGINE_UNAVAILABLE
