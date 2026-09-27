from datetime import UTC, datetime, timedelta

import pytest

from hear.contracts.jobs import AttemptEnvelope, JobType
from hear.runtime.roles import WorkerCapabilityRegistry, WorkerRole


class TestFourJobContracts:
    def test_job_type_is_exactly_four(self):
        assert {item.value for item in JobType} == {
            "pipeline",
            "transcription",
            "reconstruction",
            "magic_clean",
        }

    def test_pipeline_worker_accepts_pipeline_and_transcription(self):
        capability = WorkerCapabilityRegistry().get(WorkerRole.PIPELINE)
        assert capability.job_types == (JobType.PIPELINE, JobType.TRANSCRIPTION)

    def test_reconstruction_requires_operation(self):
        with pytest.raises(ValueError):
            AttemptEnvelope(
                job_id="job-1",
                run_id="run-1",
                attempt_id="attempt-1",
                job_type=JobType.RECONSTRUCTION,
                track_id="track-1",
                user_id="user-1",
                source={"url": "https://example.com/a.mp3", "revision": 1},
                storage={
                    "endpoint_url": "https://s3.example.com",
                    "bucket_name": "bucket",
                    "key_id": "key",
                    "application_key": "secret",
                    "folder_prefix": "users/user-1/jobs/",
                    "public_base_url": "https://cdn.example.com/media",
                    "expires_at": datetime.now(UTC) + timedelta(hours=1),
                },
                artifact_prefix="jobs/job-1/attempt-1",
                deadline=datetime.now(UTC) + timedelta(minutes=30),
                reporting_grant="grant",
                backend_base_url="https://api.example.com",
            )

    @pytest.mark.parametrize("profile", ["natural", "studio_voice", "outdoor_mobile", "clean_raw"])
    def test_one_worker_accepts_all_deepfilter_profiles(self, profile):
        from types import SimpleNamespace
        envelope = SimpleNamespace(job_type=JobType.MAGIC_CLEAN, options={"profile": profile})
        capability = WorkerCapabilityRegistry().get(WorkerRole.MAGIC_CLEAN_NATURAL)
        assert capability.accepts(envelope)

    @pytest.mark.parametrize("profile", ["voice_focus", "music_atmosphere", "sam_audio"])
    def test_legacy_magic_clean_profiles_are_rejected(self, profile):
        with pytest.raises(ValueError, match="invalid_magic_clean_profile"):
            AttemptEnvelope(
                job_id="job-1",
                run_id="run-1",
                attempt_id="attempt-1",
                job_type=JobType.MAGIC_CLEAN,
                track_id="track-1",
                user_id="user-1",
                source={"url": "https://example.com/a.mp3", "revision": 1},
                storage={
                    "endpoint_url": "https://s3.example.com",
                    "bucket_name": "bucket",
                    "key_id": "key",
                    "application_key": "secret",
                    "folder_prefix": "users/user-1/jobs/",
                    "public_base_url": "https://cdn.example.com/media",
                    "expires_at": datetime.now(UTC) + timedelta(hours=1),
                },
                options={"profile": profile},
                artifact_prefix="jobs/job-1/attempt-1",
                deadline=datetime.now(UTC) + timedelta(minutes=30),
                reporting_grant="grant",
                backend_base_url="https://api.example.com",
            )
