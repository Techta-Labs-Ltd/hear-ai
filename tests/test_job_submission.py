import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from hear.models.database import AiJob
from hear.config import settings
from hear.core.backend_registry import backend_registry
from hear.models.schemas import PipelineRequest as RequestModel, SegmentChange
from hear.services.jobs.submission import (
    _legacy_payload,
    normalize_request,
    request_fingerprint,
)


BACKEND_ID = "backend-a"
SERVICE_KEY = "backend-secret"
STORAGE = {
    "endpoint_url": "https://s3.example.test",
    "bucket_name": "backend-a-bucket",
    "key_id": "key-id",
    "application_key": "application-key",
    "folder_prefix": "users/user/jobs/job",
    "public_base_url": "https://cdn.example.test",
    "expires_at": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
}
REGISTRY_JSON = json.dumps({
    BACKEND_ID: {
        "service_key_sha256": hashlib.sha256(SERVICE_KEY.encode()).hexdigest(),
        "allowed_endpoint_urls": [STORAGE["endpoint_url"]],
        "allowed_buckets": [STORAGE["bucket_name"]],
        "allowed_public_base_urls": [STORAGE["public_base_url"]],
    }
})


@pytest.fixture(autouse=True)
def configured_backend_registry(monkeypatch):
    monkeypatch.setattr(settings, "BACKEND_REGISTRY_JSON", REGISTRY_JSON)
    backend_registry.cache_clear()
    yield
    backend_registry.cache_clear()


def pipeline_request(**kwargs):
    kwargs.setdefault("backend_id", BACKEND_ID)
    kwargs.setdefault("storage", STORAGE)
    return RequestModel(**kwargs)


def test_job_id_is_not_part_of_payload_fingerprint():
    first = normalize_request(pipeline_request(job_id="one", track_id="track", user_id="user", audio_url="https://audio.test/a.mp3"))
    second = normalize_request(pipeline_request(job_id="two", track_id="track", user_id="user", audio_url="https://audio.test/a.mp3"))

    assert request_fingerprint(first) == request_fingerprint(second)


def test_payload_change_changes_fingerprint():
    first = normalize_request(pipeline_request(job_id="job", track_id="track", user_id="user", audio_url="https://audio.test/a.mp3"))
    second = normalize_request(
        pipeline_request(job_id="job", track_id="track", user_id="user", job_type="magic-clean", audio_url="https://audio.test/a.mp3")
    )

    assert first["job_type"] == "pipeline"
    assert second["job_type"] == "magic_clean"
    assert request_fingerprint(first) != request_fingerprint(second)


def test_discovery_is_a_supported_standalone_job():
    request = normalize_request(
        pipeline_request(job_id="job", track_id="track", user_id="user", job_type="discovery", audio_url="https://audio.test/a.mp3")
    )

    assert request["job_type"] == "discovery"


@pytest.mark.parametrize(
    "job_type",
    ["pipeline", "magic_clean", "transcription", "audio_tag", "discovery"],
)
def test_audio_jobs_require_explicit_audio_url(job_type):
    with pytest.raises(ValueError, match="audio_url is required"):
        normalize_request(
            pipeline_request(job_id="job", track_id="track", user_id="user", job_type=job_type)
        )


def test_magic_clean_stem_levels_are_normalized_and_fingerprinted():
    default = normalize_request(
        pipeline_request(job_id="job", track_id="track", user_id="user", job_type="magic_clean", audio_url="https://audio.test/a.mp3")
    )
    customized = normalize_request(
        pipeline_request(
            job_id="job",
            track_id="track",
            user_id="user",
            job_type="magic_clean",
            audio_url="https://audio.test/a.mp3",
            speech=50,
            music=10,
            background=10,
        )
    )

    assert default["speech"] == 100
    assert default["music"] == 10
    assert default["background"] == 10
    assert customized["speech"] == 50
    assert customized["music"] == 10
    assert customized["background"] == 10
    assert request_fingerprint(customized) != request_fingerprint(default)


def test_magic_clean_silence_option_is_normalized_and_fingerprinted():
    keep = normalize_request(
        pipeline_request(job_id="job", track_id="track", user_id="user", job_type="magic_clean", audio_url="https://audio.test/a.mp3")
    )
    cut = normalize_request(
        pipeline_request(job_id="job", track_id="track", user_id="user", job_type="magic_clean", audio_url="https://audio.test/a.mp3", cut_silence=True)
    )

    assert keep["cut_silence"] is False
    assert cut["cut_silence"] is True
    assert request_fingerprint(cut) != request_fingerprint(keep)


@pytest.mark.parametrize("field", ["speech", "music", "background"])
def test_magic_clean_stem_levels_must_be_percentages(field):
    with pytest.raises(ValueError):
        pipeline_request(job_id="job", track_id="track", user_id="user", **{field: 101})


def test_magic_clean_stem_levels_must_be_supplied_together():
    with pytest.raises(ValueError, match="supplied together"):
        pipeline_request(job_id="job", track_id="track", user_id="user", speech=50)


@pytest.mark.parametrize("job_type", ["rebuild", "edit_transcript"])
def test_transcript_jobs_require_edited_transcript(job_type):
    with pytest.raises(ValueError, match="edited_transcript"):
        normalize_request(
            pipeline_request(job_id="job", track_id="track", user_id="user", job_type=job_type)
        )


def test_reconstruct_requires_valid_changes():
    with pytest.raises(ValueError, match="changes"):
        normalize_request(
            pipeline_request(job_id="job", track_id="track", user_id="user", job_type="reconstruct")
        )

    with pytest.raises(ValueError, match="end after"):
        normalize_request(
            pipeline_request(
                job_id="job",
                track_id="track",
                user_id="user",
                job_type="reconstruct",
                audio_url="https://audio.test/a.mp3",
                changes=[
                    SegmentChange(segment_start=2, segment_end=1, new_text="replacement")
                ],
            )
        )


def test_legacy_job_without_storage_cannot_match_current_request():
    job = AiJob(
        id="job",
        run_id="run",
        track_id="track",
        job_type="pipeline",
        max_tags=8,
        status="completed",
        input_url="https://audio.test/a.mp3",
        job_options={
            "grouped": False,
            "kind": "track",
            "track_count": 1,
            "speed_multipliers": [],
            "user_id": "user",
        },
    )
    request = normalize_request(pipeline_request(job_id="job", track_id="track", user_id="user", audio_url="https://audio.test/a.mp3"))

    assert request_fingerprint(_legacy_payload(job)) != request_fingerprint(request)


def test_user_id_is_normalized_and_cannot_be_blank():
    request = normalize_request(
        pipeline_request(
            job_id="job",
            track_id="track",
            user_id="  user-1  ",
            audio_url="https://audio.test/a.mp3",
        )
    )
    assert request["user_id"] == "user-1"

    with pytest.raises(ValueError, match="user_id is required"):
        normalize_request(
            pipeline_request(
                job_id="job",
                track_id="track",
                user_id="   ",
                audio_url="https://audio.test/a.mp3",
            )
        )
