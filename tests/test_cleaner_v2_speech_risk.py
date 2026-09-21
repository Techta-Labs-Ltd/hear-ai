import json
import os
import threading
import time
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hear.runtime.cleaner.resampling import AudioResampler
from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.speech_activity import (
    CpuSpeechActivity,
    SpeechActivityPolicy,
    SpeechActivityReport,
    SpeechInterval,
)
from hear.runtime.cleaner.speech_risk import SpeechRiskComparison
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.services.magic_clean.contracts import (
    AttemptTicket,
    CleanExecutionError,
    ValidationSummary,
)
from hear.services.magic_clean.quality import AudioQualityGate
from tests.test_cleaner_v2_contracts import ticket as ticket_fixture
from tests.test_cleaner_v2_speech_activity import fixture_source
from tests.test_cleaner_v2_speech_activity import setup as setup_fixture

ticket = ticket_fixture
setup = setup_fixture


def reports(tmp_path):
    path, expected = fixture_source(tmp_path, np.full((16000, 2), 0.1))
    before = SpeechActivityReport(
        expected.sha256,
        "a" * 64,
        16000,
        16000,
        2,
        (16000, 16000),
        (SpeechInterval(0, 0, 16000, 0.9), SpeechInterval(1, 0, 16000, 0.8)),
        False,
    )
    after = replace(before, source_sha256="b" * 64)
    return path, expected, before, after


def test_loss_positions_and_channels_are_preserved(tmp_path):
    _, expected, before, after = reports(tmp_path)
    after = replace(
        after,
        active_frames=(8000, 16000),
        intervals=(
            SpeechInterval(0, 0, 4000, 0.9),
            SpeechInterval(0, 12000, 16000, 0.9),
            SpeechInterval(1, 0, 16000, 0.9),
        ),
    )
    result = SpeechRiskComparison.compare(before, after, expected)
    assert [(v.channel, v.start_frame, v.end_frame) for v in result.source_loss_intervals] == [
        (0, 4000, 12000)
    ]
    assert result.source_active_frames == (16000, 16000)
    assert result.output_active_frames == (8000, 16000)


def test_speech_elsewhere_does_not_hide_lost_interval(tmp_path):
    _, expected, before, after = reports(tmp_path)
    before = replace(
        before,
        active_frames=(8000, 8000),
        intervals=(
            SpeechInterval(0, 0, 8000, 0.9),
            SpeechInterval(1, 0, 8000, 0.9),
        ),
    )
    after = replace(
        after,
        active_frames=(8000, 8000),
        intervals=(
            SpeechInterval(0, 8000, 16000, 0.9),
            SpeechInterval(1, 8000, 16000, 0.9),
        ),
    )
    evidence = SpeechRiskComparison.compare(before, after, expected)
    assert len(evidence.source_loss_intervals) == 2


@pytest.mark.parametrize("truncated_source", [True, False])
def test_truncated_intervals_are_never_interpreted_as_absent_speech(tmp_path, truncated_source):
    _, expected, before, after = reports(tmp_path)
    before = replace(before, intervals_truncated=truncated_source)
    after = replace(after, intervals=(), intervals_truncated=not truncated_source)
    result = SpeechRiskComparison.compare(before, after, expected)
    assert result.evidence_truncated
    assert not result.source_loss_intervals


def test_small_boundary_difference_is_not_localized_as_loss(tmp_path):
    _, expected, before, after = reports(tmp_path)
    after = replace(
        after,
        active_frames=(15000, 15000),
        intervals=(
            SpeechInterval(0, 1000, 16000, 0.9),
            SpeechInterval(1, 1000, 16000, 0.9),
        ),
    )
    assert not SpeechRiskComparison.compare(before, after, expected).source_loss_intervals


def test_mono_target_is_compared_to_each_source_channel_without_downmixing(tmp_path):
    _, expected, before, after = reports(tmp_path)
    after = replace(
        after, channels=1, active_frames=(16000,), intervals=(SpeechInterval(0, 0, 16000, 0.9),)
    )
    assert not SpeechRiskComparison.compare(before, after, expected).source_loss_intervals


@pytest.mark.parametrize(
    "change",
    [
        {"policy_sha256": "c" * 64},
        {"frames": 15999},
        {"channels": 3},
        {"active_frames": (16001, 16000)},
    ],
)
def test_mismatched_report_is_rejected(tmp_path, change):
    _, expected, before, after = reports(tmp_path)
    with pytest.raises(CleanExecutionError):
        SpeechRiskComparison.compare(before, replace(after, **change), expected)


def test_quality_emits_evidence_and_never_approves_matching_activity(tmp_path, setup, ticket):
    _, _, guard = setup
    source, expected, before, after = reports(tmp_path)
    output = tmp_path / "processed.wav"
    sf.write(output, np.full((16000, 2), 0.1), 16000, subtype="FLOAT")
    plan = AttemptTicket.model_validate_json(json.dumps(ticket)).plan

    class Analyser:
        def __init__(self):
            self.after = after

        def evaluate(self, source, processed, pinned, guard):
            assert pinned == expected
            return SpeechRiskComparison.compare(before, self.after, expected)

    analyser = Analyser()
    gate = AudioQualityGate(analyser)
    result = gate.evaluate(source, output, plan, guard, expected_source=expected)
    assert result.wanted_content == "review_required"
    assert result.speech_activity is not None
    assert "possible_speech_loss" not in result.warning_codes
    analyser.after = replace(
        after, active_frames=(0, 16000), intervals=(SpeechInterval(1, 0, 16000, 0.9),)
    )
    result = gate.evaluate(source, output, plan, guard, expected_source=expected)
    assert "possible_speech_loss" in result.warning_codes
    assert result.speech_activity.source_loss_intervals[0].channel == 0
    with pytest.raises(CleanExecutionError):
        gate.evaluate(source, output, plan, guard)


def test_absent_analyser_is_explicit_review_not_success_evidence(tmp_path, setup, ticket):
    _, _, guard = setup
    source, _, _, _ = reports(tmp_path)
    plan = AttemptTicket.model_validate_json(json.dumps(ticket)).plan
    result = AudioQualityGate().evaluate(source, source, plan, guard)
    assert result.speech_activity is None
    assert "speech_activity_unavailable" in result.warning_codes
    assert result.wanted_content == "review_required"


def test_missing_warning_and_failure_evidence_rejected(tmp_path):
    _, expected, before, after = reports(tmp_path)
    after = replace(after, active_frames=(0, 0), intervals=())
    evidence = SpeechRiskComparison.compare(before, after, expected)
    for status in ("passed", "not_applicable"):
        with pytest.raises(ValueError):
            ValidationSummary(
                hard_integrity=status,
                wanted_content="review_required",
                warning_codes=(),
                speech_activity=evidence,
            )


def test_concrete_comparison_checks_files_and_cleans_temporary_audio(tmp_path, setup, ticket):
    model, policy, guard = setup
    scanner = CpuSpeechActivity(model, policy, AudioResampler(CancellableProcessRunner()), guard)
    source, expected = fixture_source(tmp_path, np.full((24007, 2), [0.1, -0.1]), rate=24000)
    processed = tmp_path / "processed.wav"
    sf.write(processed, np.full((24007, 2), [0.01, -0.1]), 24000, subtype="FLOAT")
    plan = AttemptTicket.model_validate_json(json.dumps(ticket)).plan
    result = AudioQualityGate(SpeechRiskComparison(scanner)).evaluate(
        source, processed, plan, guard, expected_source=expected
    )
    assert "possible_speech_loss" in result.warning_codes
    assert result.speech_activity.source_active_frames == (24007, 24007)
    assert result.speech_activity.output_active_frames == (0, 24007)
    assert result.speech_activity.source_loss_intervals[0].end_frame == 24007
    assert not list(tmp_path.glob("speech-analysis-*"))


def test_configured_analyser_failure_is_typed_without_fallback(tmp_path, setup, ticket):
    _, _, guard = setup
    source, expected, _, _ = reports(tmp_path)
    plan = AttemptTicket.model_validate_json(json.dumps(ticket)).plan

    class BrokenAnalyser:
        def evaluate(self, *args):
            raise RuntimeError("private-model-path-and-secret")

    with pytest.raises(CleanExecutionError) as error:
        AudioQualityGate(BrokenAnalyser()).evaluate(
            source, source, plan, guard, expected_source=expected
        )
    assert error.value.code.value == "process_failed"
    assert "private" not in str(error.value)
    assert error.value.__suppress_context__


@pytest.mark.skipif(
    not os.environ.get("HEAR_TEST_SILERO_ONNX"), reason="explicit real model path required"
)
def test_real_cpu_comparison_preserves_review_for_identical_audio(tmp_path, ticket):
    policy = SpeechActivityPolicy(
        "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
        version("onnxruntime"),
        version("numpy"),
    )
    guard = ResourceGuard(
        ResourceBudget(20_000_000, 10_000_000, 100_000),
        tmp_path,
        time.monotonic() + 60,
        threading.Event(),
    )
    scanner = CpuSpeechActivity(
        Path(os.environ["HEAR_TEST_SILERO_ONNX"]),
        policy,
        AudioResampler(CancellableProcessRunner()),
        guard,
    )
    source, expected = fixture_source(
        tmp_path, np.random.default_rng(12).normal(0, 0.05, (16007, 2)).astype(np.float32)
    )
    plan = AttemptTicket.model_validate_json(json.dumps(ticket)).plan
    result = AudioQualityGate(SpeechRiskComparison(scanner)).evaluate(
        source, source, plan, guard, expected_source=expected
    )
    assert result.wanted_content == "review_required"
    assert "possible_speech_loss" not in result.warning_codes
    assert (
        result.speech_activity.source_active_frames == result.speech_activity.output_active_frames
    )
    assert result.speech_activity.source_sha256 == result.speech_activity.output_sha256
