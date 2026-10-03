"""Contract and DSP invariants; detector fixtures do not certify perceptual quality."""

import asyncio
import hashlib
import math
import threading
import time
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from hear.contracts.cleaning import CleaningProfiles
from hear.contracts.sound_cleanup import SoundCleanupOptions
from hear.execution.native import NativeExecutor
from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.services.magic_clean.contracts import CleanExecutionError
from hear.services.sound_cleanup.analysis import SoundAnalysis
from hear.services.sound_cleanup.planner import SoundRepairPlanner
from hear.services.sound_cleanup.service import SoundCleanupService


@pytest.fixture
def guard(tmp_path):
    return ResourceGuard(
        ResourceBudget(100_000_000, 30_000_000, 1_000_000),
        tmp_path,
        time.monotonic() + 60,
        threading.Event(),
    )


class FixtureAnalyser:
    def __init__(self, evidence):
        self.evidence = evidence
        self.called = False

    def analyse(self, *args, **kwargs):
        self.called = True
        return self.evidence


def fixture_audio(tmp_path, channels=1):
    samples = np.random.default_rng(7).normal(0, 0.0001, (480000, channels)).astype("float32")
    samples[144000:168000] += (
        np.random.default_rng(8).normal(0, 0.08, (24000, channels)).astype("float32")
    )
    path = tmp_path / "baseline.wav"
    sf.write(path, samples, 48000, subtype="FLOAT")
    count = math.ceil(len(samples) / 1536)
    rms = np.zeros((count, channels), dtype="float32")
    for i in range(count):
        rms[i] = np.sqrt(np.mean(samples[i * 1536 : (i + 1) * 1536].astype("float64") ** 2, axis=0))
    scores = {
        kind: np.zeros(count, dtype="float32")
        for kind in ("handling", "impact", "animal", "cough", "click", "protected_content")
    }
    scores["impact"][94:109] = 0.85
    evidence = SoundAnalysis(
        len(samples),
        channels,
        np.zeros(count, dtype="float32"),
        rms,
        rms.copy(),
        scores,
        np.zeros(count, dtype=bool),
        "a" * 64,
    )
    return path, samples, evidence


@pytest.mark.parametrize("channels", [1, 2])
def test_repairs_only_declared_regions_and_preserves_original(tmp_path, guard, channels):
    path, samples, evidence = fixture_audio(tmp_path, channels)
    before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    target = tmp_path / "repaired.wav"
    report = SoundCleanupService(FixtureAnalyser(evidence)).run(
        path, path, target, SoundCleanupOptions(enabled=True), guard
    )
    output, rate = sf.read(target, dtype="float32", always_2d=True)
    assert rate == 48000 and output.shape == samples.shape
    assert report["repaired_count"] == 1 and report["audio_changed"]
    event = report["events"][0]
    mask = np.zeros(len(samples), dtype=bool)
    mask[event["start"] : event["end"]] = True
    np.testing.assert_array_equal(output[~mask], samples[~mask])
    assert event["reduction_db"] >= 3
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_hash
    assert report["word_retention_certified"] is False
    assert not list(tmp_path.glob("*.incomplete.wav"))


def test_speech_overlap_is_reported_without_muting(tmp_path, guard):
    path, samples, evidence = fixture_audio(tmp_path)
    evidence.protected[:] = True
    report = SoundCleanupService(FixtureAnalyser(evidence)).run(
        path, path, tmp_path / "out.wav", SoundCleanupOptions(enabled=True), guard
    )
    assert report["status"] == "partial" and report["repaired_count"] == 0
    np.testing.assert_array_equal(
        sf.read(tmp_path / "out.wav", dtype="float32", always_2d=True)[0], samples
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"enabled": "true"},
        {"enabled": 1},
        {"enabled": True, "auto_detect": "false"},
        {"enabled": True, "targets": ["speech"]},
        {"enabled": True, "targets": ["impact", "impact"]},
        {"enabled": True, "targets": ["cough"]},
        {"enabled": True, "targets": []},
        {"enabled": True, "auto_detect": False},
        {"enabled": False, "remove_coughs": True},
        {
            "enabled": True,
            "regions": [{"start_seconds": -1.0, "end_seconds": 1.0, "kind": "impact"}],
        },
        {
            "enabled": True,
            "regions": [{"start_seconds": 1.0, "end_seconds": float("inf"), "kind": "impact"}],
        },
        {
            "enabled": True,
            "regions": [{"start_seconds": 1.0, "end_seconds": 10.0, "kind": "impact"}],
        },
        {
            "enabled": True,
            "regions": [{"start_seconds": "1", "end_seconds": 2.0, "kind": "impact"}],
        },
    ],
)
def test_strict_options_reject_unsafe_or_ambiguous_requests(payload):
    with pytest.raises(ValueError):
        SoundCleanupOptions.model_validate(payload)


@pytest.mark.parametrize(
    "other",
    [
        {"profile": "clean_raw"},
        {"profile": "natural", "trim_silence": True},
        {"profile": "natural", "cleaner_ticket": {}},
    ],
)
def test_denoise_only_and_timeline_contracts_cannot_be_overridden(other):
    with pytest.raises(ValueError):
        CleaningProfiles.validate({**other, "sound_cleanup": {"enabled": True}})


def test_old_profile_defaults_remain_unchanged():
    assert "sound_cleanup" not in CleaningProfiles.validate({"profile": "natural"})


def test_manual_confirmation_is_explicit_and_audited(tmp_path, guard):
    path, _, evidence = fixture_audio(tmp_path)
    evidence.protected[:] = True
    options = SoundCleanupOptions(
        enabled=True,
        auto_detect=False,
        regions=[
            {
                "start_seconds": 3.0,
                "end_seconds": 3.5,
                "kind": "impact",
                "confirmed_no_speech": True,
            }
        ],
    )
    result = SoundCleanupService(FixtureAnalyser(evidence)).run(
        path, path, tmp_path / "out.wav", options, guard
    )
    assert result["repaired_count"] == 1 and result["explicit_speech_overrides"] == 1
    assert result["events"][0]["reason"] == "user_confirmed_no_speech"
    assert result["events"][0]["method"] == "bounded_event_attenuation"


def test_out_of_bounds_regions_fail_before_model_analysis(tmp_path, guard):
    path, _, evidence = fixture_audio(tmp_path)
    analyser = FixtureAnalyser(evidence)
    options = SoundCleanupOptions(
        enabled=True,
        auto_detect=False,
        regions=[{"start_seconds": 11.0, "end_seconds": 12.0, "kind": "impact"}],
    )
    with pytest.raises(CleanExecutionError):
        SoundCleanupService(analyser).run(path, path, tmp_path / "out.wav", options, guard)
    assert not analyser.called and not (tmp_path / "out.wav").exists()


def test_invalid_model_grid_cannot_authorise_repair(tmp_path):
    _, _, evidence = fixture_audio(tmp_path)
    bad = replace(evidence, protected=np.zeros(1, dtype=bool))
    with pytest.raises(ValueError, match="grid"):
        SoundRepairPlanner.plan(bad, SoundCleanupOptions(enabled=True))


def test_existing_output_is_never_overwritten(tmp_path, guard):
    path, _, evidence = fixture_audio(tmp_path)
    target = tmp_path / "out.wav"
    target.write_bytes(b"previous-approved-file")
    with pytest.raises(CleanExecutionError):
        SoundCleanupService(FixtureAnalyser(evidence)).run(
            path, path, target, SoundCleanupOptions(enabled=True), guard
        )
    assert target.read_bytes() == b"previous-approved-file"


def test_native_cancellation_signals_worker_and_waits_for_safe_exit():
    started, stopped, cancellation = threading.Event(), threading.Event(), threading.Event()

    def work(*, cancelled):
        started.set()
        cancelled.wait(3)
        stopped.set()
        return "finished"

    async def run():
        native = NativeExecutor("cancel-sound-test")
        task = asyncio.create_task(native.run_cancellable(work, cancelled=cancellation))
        assert await asyncio.to_thread(started.wait, 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancellation.is_set() and stopped.is_set()
        await native.close()

    asyncio.run(run())


class FixtureSeparator:
    def __init__(self, rejected=False):
        self.called = False
        self.closed = False
        self.rejected = rejected

    def propose(self, audio, region, evidence, analyser, guard):
        self.called = True
        if self.rejected:
            return None, "possible_speech_loss_or_leakage", {"removed_speech_frames": 5}
        audio.seek(region.start)
        return audio.read(region.end - region.start, dtype="float32", always_2d=True) * 0.5, "", {}

    def close(self):
        self.closed = True


@pytest.mark.parametrize("rejected", [False, True])
def test_overlap_requires_selection_and_keeps_review_status(tmp_path, guard, rejected):
    path, samples, evidence = fixture_audio(tmp_path)
    evidence.protected[:] = True
    separator = FixtureSeparator(rejected)
    options = SoundCleanupOptions(
        enabled=True,
        auto_detect=False,
        preview_overlaps=True,
        regions=[{"start_seconds": 3.0, "end_seconds": 3.5, "kind": "animal"}],
    )
    result = SoundCleanupService(FixtureAnalyser(evidence), separator).run(
        path, path, tmp_path / "out.wav", options, guard
    )
    assert separator.called and separator.closed
    assert result["status"] == "partial" and result["requires_approval"]
    assert result["events"][0]["outcome"] == ("rejected" if rejected else "preview")
    if rejected:
        np.testing.assert_array_equal(
            sf.read(tmp_path / "out.wav", dtype="float32", always_2d=True)[0], samples
        )
    else:
        assert result["preview_count"] == 1 and result["repaired_count"] == 0


def test_overlap_request_without_assets_fails_closed(tmp_path, guard):
    path, _, evidence = fixture_audio(tmp_path)
    options = SoundCleanupOptions(
        enabled=True,
        auto_detect=False,
        preview_overlaps=True,
        regions=[{"start_seconds": 3.0, "end_seconds": 3.5, "kind": "animal"}],
    )
    with pytest.raises(CleanExecutionError, match="not_provisioned"):
        SoundCleanupService(FixtureAnalyser(evidence)).run(
            path, path, tmp_path / "out.wav", options, guard
        )


def test_cancelled_request_never_runs_detector(tmp_path, guard):
    path, _, evidence = fixture_audio(tmp_path)
    analyser = FixtureAnalyser(evidence)
    guard.cancelled.set()
    with pytest.raises(CleanExecutionError, match="cancelled"):
        SoundCleanupService(analyser).run(
            path, path, tmp_path / "out.wav", SoundCleanupOptions(enabled=True), guard
        )
    assert not analyser.called and not (tmp_path / "out.wav").exists()


def test_overlap_preview_cannot_be_enabled_without_selected_region():
    with pytest.raises(ValueError, match="selected_regions"):
        SoundCleanupOptions(enabled=True, preview_overlaps=True)


@pytest.mark.parametrize("provisioned", [False, True])
def test_capability_gates_follow_provisioned_bundles(provisioned):
    from hear.api.gateway import PodGateway

    class Gateway:
        async def lane_status(self):
            return {"magic_clean_natural": {"status": "ready"}}

    result = asyncio.run(
        PodGateway(
            Gateway(),
            "key",
            sound_cleanup_available=provisioned,
            overlap_preview_available=provisioned,
        ).capabilities()
    )
    assert result["magic_clean"]["sound_cleanup"]["available"] is provisioned
    assert result["magic_clean"]["sound_cleanup"]["selected_overlap_preview"] is provisioned


def test_corrupt_model_digest_does_not_enable_a_bundle(tmp_path):
    from hear.services.sound_cleanup.assets import SoundCleanupAssets

    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(CleanExecutionError, match="manifest_mismatch"):
        SoundCleanupAssets.load(tmp_path, "a" * 64)


def test_frame_count_and_channel_layout_are_validated(tmp_path):
    _, _, evidence = fixture_audio(tmp_path, 2)
    with pytest.raises(ValueError, match="channel_layout"):
        SoundRepairPlanner.plan(
            replace(evidence, source_rms=evidence.source_rms[:, :1]),
            SoundCleanupOptions(enabled=True),
        )


def test_overlap_preview_respects_total_edit_coverage(tmp_path):
    _, _, evidence = fixture_audio(tmp_path)
    evidence.protected[:] = True
    options = SoundCleanupOptions(
        enabled=True,
        auto_detect=False,
        preview_overlaps=True,
        regions=[{"start_seconds": 1.0, "end_seconds": 9.0, "kind": "animal"}],
    )
    regions, _ = SoundRepairPlanner.plan(evidence, options)
    assert regions[0].outcome == "needs_review" and regions[0].reason == "repair_coverage_limit"


@pytest.mark.parametrize("expired", [False, True])
def test_unavailable_or_expired_workflow_does_not_download(tmp_path, expired):
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace

    from hear.workflows.available_magic_clean import AvailableMagicCleanWorkflow
    from tests.test_cleaning_profile_workflow import InlineNative, envelope

    class NoDownload:
        async def download_source(self, *args):
            pytest.fail("rejected request must not download")

    request = envelope("natural")
    request = request.model_copy(
        update={
            "options": CleaningProfiles.validate(
                {"profile": "natural", "sound_cleanup": {"enabled": True}}
            )
        }
    )
    if expired:
        request = request.model_copy(update={"deadline": datetime.now(UTC) - timedelta(seconds=1)})
    workflow = AvailableMagicCleanWorkflow(
        NoDownload(),
        None,
        InlineNative(),
        workspace_root=tmp_path,
        timeout_seconds=30,
        model_cleaner=SimpleNamespace(sound_cleanup_available=False),
    )

    async def collect():
        return [event async for event in workflow.stream(request)]

    result = asyncio.run(collect())[-1].data["outcome"]
    assert result["error_code"] == ("deadline_exceeded" if expired else "engine_unavailable")
