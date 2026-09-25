import pytest
from pydantic import ValidationError

from hear.contracts.jobs import (
    AttemptEnvelope,
    JobType,
    MagicCleanProfile,
    ReconstructionOperation,
    WorkerRole,
)
from hear.runtime.roles import RoleRegistry


def envelope(**changes):
    payload = {
        "job_id": "job-1",
        "run_id": "run-1",
        "attempt_id": "attempt-1",
        "job_type": JobType.PIPELINE,
        "track_id": "track-1",
        "user_id": "user-1",
        "source": {"url": "https://example.invalid/audio.mp3", "revision": 1},
        "storage": {
            "reference": "storage-1",
            "token": "secret",
            "expires_at": "2026-09-26T00:00:00Z",
        },
        "reporting": {
            "backend_base_url": "https://api.example.invalid",
            "token": "secret",
        },
    }
    payload.update(changes)
    return AttemptEnvelope.model_validate(payload)


def test_exact_four_job_types():
    assert {item.value for item in JobType} == {
        "pipeline",
        "transcription",
        "reconstruction",
        "magic_clean",
    }


def test_reconstruction_requires_operation():
    with pytest.raises(ValidationError):
        envelope(job_type=JobType.RECONSTRUCTION)


def test_magic_clean_requires_profile():
    with pytest.raises(ValidationError):
        envelope(job_type=JobType.MAGIC_CLEAN)


def test_reconstruction_contract():
    value = envelope(
        job_type=JobType.RECONSTRUCTION,
        operation=ReconstructionOperation.EDIT_TRANSCRIPT,
    )
    assert value.operation == ReconstructionOperation.EDIT_TRANSCRIPT


def test_magic_clean_contract():
    value = envelope(
        job_type=JobType.MAGIC_CLEAN,
        magic_clean_profile=MagicCleanProfile.NATURAL,
    )
    assert value.magic_clean_profile == MagicCleanProfile.NATURAL


def test_pipeline_worker_accepts_pipeline_and_transcription():
    capabilities = RoleRegistry().get(WorkerRole.PIPELINE)
    assert capabilities.accepts(JobType.PIPELINE)
    assert capabilities.accepts(JobType.TRANSCRIPTION)
    assert not capabilities.accepts(JobType.RECONSTRUCTION)


def test_magic_clean_profile_isolated():
    capabilities = RoleRegistry().get(WorkerRole.MAGIC_CLEAN_NATURAL)
    assert capabilities.accepts(JobType.MAGIC_CLEAN, MagicCleanProfile.NATURAL)
    assert not capabilities.accepts(JobType.MAGIC_CLEAN, MagicCleanProfile.VOICE_FOCUS)
