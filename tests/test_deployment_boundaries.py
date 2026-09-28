import json
import threading
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from hear.contracts.cleaning import CleaningProfiles
from hear.contracts.jobs import ArtifactStorage
from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.ownership import BackendOwnershipPolicy
from hear.services.magic_clean.contracts import CleanExecutionError
from hear.services.sound_cleanup.background import BackgroundCleanup
from hear.storage.b2 import B2Storage
from tests.test_cleaning_profile_workflow import envelope


def policy():
    return BackendOwnershipPolicy.from_json(
        json.dumps(
            {
                "backend_id": "backend-a",
                "backend_base_urls": ["https://api.hear.media"],
                "bucket_name": "OldAlexa",
                "storage_endpoint": "https://s3.eu-central-003.backblazeb2.com",
                "public_base_url": "https://cdn.hear.media",
                "source_hosts": ["cdn.hear.media"],
            }
        )
    )


def owned():
    data = envelope("natural").model_dump()
    data.update(
        backend_id="backend-a",
        backend_base_url="https://api.hear.media",
        artifact_prefix="creators/example/audio/jobs/job-1/attempt-1",
    )
    data["source"].update(url="https://cdn.hear.media/test.mp3", file_sha256="a" * 64)
    data["storage"].update(
        bucket_name="OldAlexa",
        endpoint_url="https://s3.eu-central-003.backblazeb2.com",
        public_base_url="https://cdn.hear.media",
        folder_prefix="creators/example/audio/jobs/job-1/",
    )
    return type(envelope("natural")).model_validate(data)


def test_valid_production_boundary():
    policy().validate(owned())


@pytest.mark.parametrize(
    "change",
    [
        {"backend_id": "backend-a-dev"},
        {"backend_base_url": "https://api.hear.surf"},
        {"artifact_prefix": "creators/example/audio/jobs/job-2/attempt-1"},
    ],
)
def test_cross_environment_rejected(change):
    with pytest.raises(ValueError):
        policy().validate(owned().model_copy(update=change))


@pytest.mark.parametrize(
    "change",
    [
        {"bucket_name": "hear-dev-uploads"},
        {"endpoint_url": "https://s3.us-east-005.backblazeb2.com"},
        {"public_base_url": "https://media.hear.surf"},
        {"folder_prefix": "creators/example/audio/jobs/other-job/"},
        {"expires_at": datetime.now(UTC) - timedelta(seconds=1)},
    ],
)
def test_wrong_storage_boundary_rejected(change):
    request = owned()
    with pytest.raises(ValueError):
        policy().validate(
            request.model_copy(update={"storage": request.storage.model_copy(update=change)})
        )


def storage(monkeypatch, prefix="creators/example/audio/jobs/job-1/"):
    monkeypatch.setattr("hear.storage.b2.boto3.client", lambda *args, **kwargs: None)
    return B2Storage(owned().storage.model_copy(update={"folder_prefix": prefix}))


def test_no_duplicate_job_folder(monkeypatch):
    value = storage(monkeypatch)
    assert (
        value.key("jobs", "job-1", "attempt-1", "x.json")
        == "creators/example/audio/jobs/job-1/attempt-1/x.json"
    )
    with pytest.raises(ValueError):
        value.key("jobs", "job-2", "attempt-1", "x.json")
    owner = storage(monkeypatch, "creators/example/audio/")
    assert (
        owner.key("jobs", "job-1", "attempt-1", "x.json")
        == "creators/example/audio/jobs/job-1/attempt-1/x.json"
    )


@pytest.mark.parametrize("path", ["/absolute", "a/../b", "a/./b", "a//b", "a\\b", "a\x00b"])
def test_storage_path_normalization_cannot_hide_traversal(monkeypatch, path):
    value = storage(monkeypatch)
    with pytest.raises(ValueError):
        value.key(path)
    with pytest.raises(ValueError):
        value._validate_key(value._context.folder_prefix + path)


def test_expiry_checked_at_use(monkeypatch):
    value = storage(monkeypatch)
    value._context = value._context.model_copy(
        update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)}
    )
    with pytest.raises(ValueError, match="expired"):
        value.key("test.json")


def test_expiry_requires_timezone():
    value = owned().storage.model_dump()
    value["expires_at"] = datetime.now()
    with pytest.raises(ValueError, match="timezone"):
        ArtifactStorage.model_validate(value)


@pytest.mark.parametrize("profile", ["natural", "studio_voice", "outdoor_mobile"])
def test_optional_background_request(profile):
    assert CleaningProfiles.validate({"profile": profile, "reduce_stationary_noise": True})[
        "reduce_stationary_noise"
    ]


@pytest.mark.parametrize(
    "options",
    [
        {"profile": "clean_raw", "reduce_stationary_noise": True},
        {"profile": "natural", "reduce_stationary_noise": "false"},
    ],
)
def test_background_options_fail_closed(options):
    with pytest.raises(ValueError):
        CleaningProfiles.validate(options)


@pytest.fixture
def guard(tmp_path):
    return ResourceGuard(
        ResourceBudget(1_000_000_000, 100_000_000, 1_000_000),
        tmp_path,
        time.monotonic() + 90,
        threading.Event(),
    )


def evidence(n, channels, speech=True):
    count = (n + 1535) // 1536
    return SimpleNamespace(
        frames=n,
        channels=channels,
        step_frames=1536,
        speech=np.ones(count) if speech else np.zeros(count),
        protected=np.ones(count, dtype=bool) if speech else np.zeros(count, dtype=bool),
        clean_rms=np.ones((count, channels)) * 0.1,
    )


@pytest.mark.parametrize("channels", [1, 2])
def test_no_reference_no_tone_preserves_samples(tmp_path, guard, channels):
    t = np.arange(48000 * 3) / 48000
    x = (0.1 * np.sin(2 * np.pi * (175 * t + 11 * t * t))).astype("float32")
    x = np.column_stack([x if c == 0 else -x for c in range(channels)])
    source, target = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(source, x, 48000, subtype="FLOAT")
    report = BackgroundCleanup().render(source, target, evidence(len(x), channels), guard)
    result, _ = sf.read(target, dtype="float32", always_2d=True)
    np.testing.assert_array_equal(result, x)
    assert report["status"] == "no_confident_stationary_reference"


@pytest.mark.parametrize("fundamental", [50, 60])
def test_measured_mains_removal(tmp_path, guard, fundamental):
    t = np.arange(48000 * 4) / 48000
    wanted = 0.1 * np.sin(2 * np.pi * (180 * t + 8 * t * t))
    hum = 0.025 * np.sin(2 * np.pi * fundamental * t)
    source, target = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(source, wanted + hum, 48000, subtype="FLOAT")
    report = BackgroundCleanup().render(source, target, evidence(len(t), 1), guard)
    result, _ = sf.read(target)
    assert fundamental in report["detected_mains_lines_hz"]
    assert len(result) == len(t)
    assert np.mean((result[48000:] - wanted[48000:]) ** 2) < np.mean(hum[48000:] ** 2) * 0.1


def test_background_cancellation_before_output(tmp_path, guard):
    guard.cancelled.set()
    with pytest.raises(CleanExecutionError):
        BackgroundCleanup().render(tmp_path / "missing.wav", tmp_path / "out.wav", None, guard)
    assert not (tmp_path / "out.wav").exists()
