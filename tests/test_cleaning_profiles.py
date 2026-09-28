"""Profile contract and real FFmpeg tests; the model stub is explicitly not a denoising test."""

import asyncio
import hashlib
import shutil
import threading
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from hear.api.gateway import PodGateway
from hear.contracts.cleaning import CleaningProfiles, MagicCleanProfile
from hear.contracts.jobs import JobType
from hear.entrypoints.gateway import GatewayEntrypoint
from hear.queue.topology import RabbitMQTopology
from hear.runtime.cleaner.deepfilter_available import DeepFilterNetCleaner
from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.runtime.roles import WorkerCapabilityRegistry, WorkerRole
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode, RuntimeIdentity
from hear.services.magic_clean.mastering import AudioMasteringService
from hear.services.magic_clean.profile_dsp import ProfileDspService


@pytest.fixture
def guard(tmp_path):
    return ResourceGuard(
        ResourceBudget(200_000_000, 50_000_000, 4_800_000),
        tmp_path,
        time.monotonic() + 90,
        threading.Event(),
    )


@pytest.fixture
def dsp():
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg is required for real DSP tests")
    return ProfileDspService(CancellableProcessRunner())


@pytest.mark.parametrize("profile", list(MagicCleanProfile))
def test_every_profile_has_a_real_worker_route(profile):
    options = CleaningProfiles.validate({"profile": profile.value})
    envelope = SimpleNamespace(job_type=JobType.MAGIC_CLEAN, options=options)
    role = RabbitMQTopology().role_for(envelope)
    assert role == WorkerRole.MAGIC_CLEAN_NATURAL
    assert WorkerCapabilityRegistry().get(role).accepts(envelope)
    assert RabbitMQTopology().binding(role).routing_key == "magic_clean.natural.v4"
    assert profile.value in DeepFilterNetCleaner.supported_profiles


@pytest.mark.parametrize(
    "options",
    [
        {"profile": "sam_audio"},
        {"profile": "echo_room"},
        {"profile": "studio_voice", "prompt": "voice"},
        {"profile": "natural", "attenuation_limit_db": True},
        {"profile": "natural", "attenuation_limit_db": 99},
        {"profile": "natural", "auto_level": "false"},
        {"profile": "natural", "remove_clicks": 1},
        {"profile": "natural", "trim_silence": None},
        {"profile": "clean_raw", "auto_level": True},
        {"profile": "clean_raw", "remove_clicks": True},
        {"profile": "clean_raw", "trim_silence": True},
        {"profile": "studio_voice", "cleaner_ticket": {}},
    ],
)
def test_unsupported_or_ambiguous_options_rejected(options):
    with pytest.raises(ValueError):
        CleaningProfiles.validate(options)


def test_legacy_natural_defaults_remain_denoise_only():
    options = CleaningProfiles.validate({"profile": "natural"})
    assert options["attenuation_limit_db"] == 24
    assert not any(options[key] for key in ("auto_level", "trim_silence", "remove_clicks"))
    assert ProfileDspService.preparation_filters(options) == []
    assert ProfileDspService.finishing_filters(options) == []


def test_profiles_do_not_claim_echo_or_prompt_separation():
    catalogue = CleaningProfiles.catalogue()
    assert catalogue["recommended_profile"] == "studio_voice"
    assert set(catalogue["profiles"]) == {p.value for p in MagicCleanProfile}
    assert all(spec["engine"] == "deepfilternet3" for spec in catalogue["profiles"].values())


@pytest.mark.parametrize(
    "ready,mode", [(False, "available"), (True, "available"), (True, "certified")]
)
def test_catalogue_reports_lane_availability(ready, mode):
    class Gateway:
        async def lane_status(self):
            return {"magic_clean_natural": {"status": "ready" if ready else "loading"}}

    response = asyncio.run(PodGateway(Gateway(), "key", cleaning_mode=mode).capabilities())
    for name, spec in response["magic_clean"]["profiles"].items():
        assert spec["available"] is (ready and (mode == "available" or name == "natural"))


def tone(rate=48000, seconds=2, channels=1):
    t = np.arange(int(rate * seconds)) / rate
    signal = 0.08 * np.sin(2 * np.pi * 220 * t) + 0.04 * np.sin(2 * np.pi * 3300 * t)
    return np.column_stack([signal * (1 - i * 0.25) for i in range(channels)]).astype("float32")


@pytest.mark.parametrize("profile", list(MagicCleanProfile))
def test_each_filter_chain_executes_with_real_ffmpeg(dsp, guard, tmp_path, profile):
    source, prepared, finished = [
        tmp_path / name for name in ("source.wav", "prepared.wav", "finished.wav")
    ]
    samples = tone(channels=2)
    sf.write(source, samples, 48000, subtype="FLOAT")
    options = CleaningProfiles.validate({"profile": profile.value})
    dsp.render(source, prepared, dsp.preparation_filters(options), guard)
    dsp.render(prepared, finished, dsp.finishing_filters(options), guard)
    output, rate = sf.read(finished, always_2d=True)
    assert output.shape == samples.shape
    assert rate == 48000 and np.isfinite(output).all()
    if profile in (MagicCleanProfile.NATURAL, MagicCleanProfile.CLEAN_RAW):
        np.testing.assert_array_equal(output, samples)
    else:
        assert not np.allclose(output, samples)


def test_click_removal_really_runs_a_dedicated_filter(dsp, guard, tmp_path):
    source, output = tmp_path / "source.wav", tmp_path / "output.wav"
    samples = tone()
    samples[48000] = 0.9
    sf.write(source, samples, 48000, subtype="FLOAT")
    options = CleaningProfiles.validate({"profile": "natural", "remove_clicks": True})
    dsp.render(source, output, dsp.preparation_filters(options), guard)
    cleaned, _ = sf.read(output, always_2d=True)
    assert cleaned.shape == samples.shape
    assert abs(cleaned[48000, 0]) < abs(samples[48000, 0])


def test_edge_trim_preserves_internal_pause_and_quiet_handles(dsp, guard, tmp_path):
    source, output = tmp_path / "source.wav", tmp_path / "output.wav"
    samples = np.concatenate(
        [
            np.zeros((48000, 1)),
            tone(seconds=1),
            np.zeros((96000, 1)),
            tone(seconds=1),
            np.zeros((48000, 1)),
        ]
    )
    sf.write(source, samples, 48000, subtype="FLOAT")
    trim = dsp.trim_edges(source, output, guard)
    cleaned, _ = sf.read(output, always_2d=True)
    assert trim.start_frame == 36000
    assert trim.end_frame == 252000
    np.testing.assert_array_equal(
        cleaned, samples[trim.start_frame : trim.end_frame].astype("float32")
    )
    assert np.all(cleaned[60000:156000] == 0)


def test_silence_is_not_trimmed_to_an_empty_file(dsp, guard, tmp_path):
    source, output = tmp_path / "source.wav", tmp_path / "output.wav"
    sf.write(source, np.zeros((96000, 2)), 48000, subtype="FLOAT")
    trim = dsp.trim_edges(source, output, guard)
    assert (trim.start_frame, trim.end_frame) == (0, 96000)


def test_nonfinite_samples_are_rejected(dsp, guard, tmp_path):
    source = tmp_path / "nan.wav"
    sf.write(source, np.full((480, 1), np.nan), 48000, subtype="FLOAT")
    with pytest.raises(CleanExecutionError):
        dsp.validate(source, guard)


def test_expired_attempt_starts_no_codec_process(dsp, guard, tmp_path):
    guard.bind_deadline(datetime.now(UTC) - timedelta(seconds=1))
    with pytest.raises(CleanExecutionError) as error:
        dsp.render(tmp_path / "missing.wav", tmp_path / "out.wav", [], guard)
    assert error.value.code == ErrorCode.DEADLINE_EXCEEDED
    assert not (tmp_path / "out.wav").exists()


class PassthroughModelForPipelineTest:
    """Exercise real DSP/mastering separately from optional real-checkpoint tests."""

    def __init__(self):
        self.plans = []
        self.closed = False

    def open_session(self, plan, guard):
        self.plans.append(plan)
        return self

    def process(self, source, destination, plan, guard):
        shutil.copyfile(source, destination)

    def close(self):
        self.closed = True


def stubbed_cleaner():
    cleaner = DeepFilterNetCleaner.__new__(DeepFilterNetCleaner)
    cleaner._budget = ResourceBudget(200_000_000, 50_000_000, 4_800_000)
    digest = hashlib.sha256(b"explicit-test-model-stub").hexdigest()
    cleaner._identity = RuntimeIdentity(
        engine="deepfilternet3",
        runtime_sha256=digest,
        checkpoint_sha256=digest,
        precision_policy_sha256=digest,
        longform_policy_sha256=digest,
    )
    cleaner._engine = PassthroughModelForPipelineTest()
    cleaner._runner = CancellableProcessRunner()
    cleaner._dsp = ProfileDspService(cleaner._runner)
    cleaner._mastering = AudioMasteringService(cleaner._runner)
    return cleaner


@pytest.mark.parametrize("profile", list(MagicCleanProfile))
@pytest.mark.parametrize("channels", [1, 2])
def test_complete_profile_pipeline_exports_measured_files(dsp, tmp_path, profile, channels):
    source, target = tmp_path / "source.wav", tmp_path / "cleaned_master.flac"
    samples = tone(rate=44100, seconds=3, channels=channels)
    sf.write(source, samples, 44100, subtype="FLOAT")
    cleaner = stubbed_cleaner()
    report = cleaner.clean(
        source,
        target,
        tmp_path,
        {"profile": profile.value},
        datetime.now(UTC) + timedelta(seconds=90),
        90,
    )
    assert target.is_file() and (tmp_path / "delivery_audio.mp3").is_file()
    assert sf.info(target).subtype == "PCM_24"
    assert sf.info(target).frames == 144000 and sf.info(target).channels == channels
    assert report["profile"] == profile.value and report["channels"] == channels
    assert report["master_measurement"]["true_peak_dbtp"] <= -1
    assert report["delivery_measurement"]["true_peak_dbtp"] <= -1
    assert report["perceptual_review_required"] is True
    assert cleaner._engine.closed
    assert (
        cleaner._engine.plans[0].attenuation_limit_db
        == report["effective_options"]["attenuation_limit_db"]
    )


def test_raw_profile_preserves_dynamics_and_does_not_boost(dsp, tmp_path):
    source, target = tmp_path / "source.wav", tmp_path / "cleaned_master.flac"
    samples = tone(seconds=3)
    sf.write(source, samples, 48000, subtype="FLOAT")
    report = stubbed_cleaner().clean(
        source,
        target,
        tmp_path,
        {"profile": "clean_raw"},
        datetime.now(UTC) + timedelta(seconds=90),
        90,
    )
    result, _ = sf.read(target, always_2d=True)
    assert report["gain_db"] == 0 and report["target_lufs"] is None
    np.testing.assert_allclose(result, samples, atol=2e-7, rtol=0)


def test_retired_role_does_not_break_other_workers_during_migration(monkeypatch):
    monkeypatch.setenv("HEAR_POD_STACK_ROLES", "pipeline,magic_clean_natural,magic_clean_sam_audio")
    assert GatewayEntrypoint.roles() == {WorkerRole.PIPELINE, WorkerRole.MAGIC_CLEAN_NATURAL}
