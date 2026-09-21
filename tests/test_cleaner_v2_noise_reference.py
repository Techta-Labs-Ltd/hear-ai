import hashlib
import json
import os
import threading
import time
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from hear.runtime.cleaner.noise_reference import SpeechAwareNoiseReferenceAnalyser
from hear.runtime.cleaner.resampling import AudioResampler
from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.speech_activity import (
    CpuSpeechActivity,
    SpeechActivityPolicy,
    SpeechActivityReport,
)
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.services.magic_clean.contracts import (
    CleanExecutionError,
    NoiseReferenceSelection,
)
from hear.services.magic_clean.engines.noise_profile import NoiseProfileEngine
from tests.test_cleaner_v2_contracts import ticket as ticket_fixture
from tests.test_cleaner_v2_noise_profile import processing as processing_fixture

ticket = ticket_fixture
processing = processing_fixture


class SpeechScanner:
    def __init__(self):
        self.fail = None
        self.policy = SimpleNamespace(digest="a" * 64)
        self.seen = None

    def scan(self, path, identity, guard):
        data, rate = sf.read(path, always_2d=True)
        self.seen = data.copy()
        assert rate == identity.sample_rate
        assert hashlib.sha256(path.read_bytes()).hexdigest() == identity.sha256
        if self.fail == "crash":
            raise RuntimeError("private-decoder-path")
        if self.fail == "cancel":
            guard.cancelled.set()
        result = SpeechActivityReport(
            identity.sha256,
            self.policy.digest,
            rate,
            len(data),
            data.shape[1],
            tuple(int(v) for v in np.sum(np.abs(data) > 0.05, axis=0)),
            (),
            False,
        )
        return replace(result, source_sha256="f" * 64) if self.fail == "identity" else result


@pytest.fixture
def reference_audio(tmp_path, processing):
    plan, guard = processing
    data = np.random.default_rng(12).uniform(-0.01, 0.01, (16000, 2)).astype(np.float32)
    # Speech-like evidence outside the chosen interval must not veto this sample.
    data[:1000] = 0.2
    source = tmp_path / "source.wav"
    sf.write(source, data, 48000, subtype="FLOAT")
    selection = NoiseReferenceSelection(revision_id="revision-1", start_frame=2000, end_frame=12000)
    scanner = SpeechScanner()
    return source, data, selection, scanner, SpeechAwareNoiseReferenceAnalyser(scanner), plan, guard


def test_negative_speech_remains_uncertain_and_selection_is_exact(reference_audio):
    source, data, selection, scanner, analyser, _, guard = reference_audio
    review = analyser.review(source, selection, guard)
    np.testing.assert_array_equal(scanner.seen, data[2000:12000])
    assert not review.assessment.speech_detected
    assert review.assessment.uncertain
    assert "music_analysis_unavailable" in review.warning_codes
    assert "noise_reference_requires_confirmation" in review.warning_codes
    assert review.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert not list(guard.workspace.glob("noise-reference-*"))


@pytest.mark.parametrize("change", ["source", "revision", "interval", "policy"])
def test_digest_binds_actual_source_revision_selection_and_analysis(reference_audio, change):
    source, data, selection, scanner, analyser, _, guard = reference_audio
    initial = analyser.review(source, selection, guard).assessment.analysis_sha256
    if change == "source":
        data[0] = 0.3  # Even changes outside the selected interval create a new source identity.
        sf.write(source, data, 48000, subtype="FLOAT")
    elif change == "revision":
        selection = selection.model_copy(update={"revision_id": "revision-2"})
    elif change == "interval":
        selection = selection.model_copy(update={"start_frame": 2001})
    else:
        scanner.policy = SimpleNamespace(digest="b" * 64)
    assert analyser.review(source, selection, guard).assessment.analysis_sha256 != initial


def test_review_digest_is_recomputed_and_consumed_by_noise_engine(reference_audio, tmp_path):
    source, _, selection, _, analyser, plan, guard = reference_audio
    digest = analyser.review(source, selection, guard).assessment.analysis_sha256
    raw = plan.model_dump(mode="json")
    raw["runtime"] = NoiseProfileEngine.describe(analyser).model_dump(mode="json")
    raw["noise_reference"]["analysis_sha256"] = digest
    plan = type(plan).model_validate_json(json.dumps(raw))
    output = tmp_path / "clean.wav"
    NoiseProfileEngine(plan.runtime, analyser).open_session(plan, guard).process(
        source, output, plan, guard
    )
    assert sf.info(output).frames == 16000
    assert not list(tmp_path.glob("noise-reference-*"))


@pytest.mark.parametrize("failure", ["speech", "stale_digest"])
def test_speech_or_stale_reference_is_rejected_before_output(reference_audio, tmp_path, failure):
    source, data, selection, _, analyser, plan, guard = reference_audio
    if failure == "speech":
        data[4000:8000, 1] = -0.2  # Speech on only one stereo channel still vetoes the reference.
        sf.write(source, data, 48000, subtype="FLOAT")
    review = analyser.review(source, selection, guard)
    raw = plan.model_dump(mode="json")
    raw["runtime"] = NoiseProfileEngine.describe(analyser).model_dump(mode="json")
    raw["noise_reference"]["analysis_sha256"] = (
        review.assessment.analysis_sha256 if failure == "speech" else "0" * 64
    )
    plan = type(plan).model_validate_json(json.dumps(raw))
    output = tmp_path / "clean.wav"
    with pytest.raises(CleanExecutionError):
        NoiseProfileEngine(plan.runtime, analyser).open_session(plan, guard).process(
            source, output, plan, guard
        )
    assert not output.exists()


@pytest.mark.parametrize("failure", ["crash", "cancel", "identity"])
def test_failed_analysis_is_typed_and_cleans_its_crop(reference_audio, failure):
    source, _, selection, scanner, analyser, _, guard = reference_audio
    scanner.fail = failure
    with pytest.raises(CleanExecutionError) as error:
        analyser.review(source, selection, guard)
    assert "private" not in str(error.value)
    assert not list(guard.workspace.glob("noise-reference-*"))


@pytest.mark.parametrize("interval", [(0, 100), (15000, 20000)])
def test_invalid_interval_rejected(reference_audio, interval):
    source, _, _, _, analyser, _, guard = reference_audio
    selection = NoiseReferenceSelection(
        revision_id="revision-1", start_frame=interval[0], end_frame=interval[1]
    )
    with pytest.raises(CleanExecutionError):
        analyser.review(source, selection, guard)


@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf")])
def test_silent_or_invalid_selected_pcm_rejected_before_speech_scan(reference_audio, value):
    source, data, selection, scanner, analyser, _, guard = reference_audio
    data[selection.start_frame : selection.end_frame] = value
    sf.write(source, data, 48000, subtype="FLOAT")
    with pytest.raises(CleanExecutionError) as error:
        analyser.review(source, selection, guard)
    assert error.value.code.value == "invalid_audio"
    assert scanner.seen is None
    assert not list(guard.workspace.glob("noise-reference-*"))


@pytest.mark.skipif(
    not os.environ.get("HEAR_TEST_SILERO_ONNX"), reason="explicit real model path required"
)
def test_real_reference_review_is_repeatable_and_never_confident_from_negative_vad(tmp_path):
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
    source = tmp_path / "source.wav"
    sf.write(source, np.random.default_rng(44).normal(0, 0.005, (16000, 2)), 16000, subtype="FLOAT")
    selection = NoiseReferenceSelection(revision_id="revision-1", start_frame=2000, end_frame=12000)
    analyser = SpeechAwareNoiseReferenceAnalyser(scanner)
    first = analyser.review(source, selection, guard)
    second = analyser.review(source, selection, guard)
    assert first == second
    assert first.assessment.uncertain
    assert first.analysis_policy_sha256 == policy.digest
