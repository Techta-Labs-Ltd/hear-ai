import json
import threading
import time

import numpy as np
import pytest
import soundfile as sf

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.services.magic_clean.contracts import AttemptTicket, CleanExecutionError
from hear.services.magic_clean.engines.noise_profile import (
    NoiseProfileEngine,
    NoiseProfileSession,
    NoiseReferenceAssessment,
)
from tests.test_cleaner_v2_contracts import ticket as ticket_fixture

ticket = ticket_fixture


class ReferenceAnalyser:
    policy_sha256 = "a" * 64

    def __init__(self, *, speech=False, music=False):
        self.speech = speech
        self.music = music

    def assess(self, source, plan, guard):
        return NoiseReferenceAssessment(
            plan.noise_reference.analysis_sha256, self.speech, self.music, False
        )


@pytest.fixture
def processing(tmp_path, ticket):
    ticket["plan"].update(
        profile="music_atmosphere",
        attenuation_limit_db=None,
        noise_reduction_db=3,
        adjust_loudness=False,
        noise_reference={
            "revision_id": "revision-1",
            "start_frame": 2000,
            "end_frame": 12000,
            "confirmed_noise_only": True,
            "analysis_sha256": "b" * 64,
        },
    )
    ticket["plan"]["runtime"] = NoiseProfileEngine.describe(ReferenceAnalyser()).model_dump(
        mode="json"
    )
    plan = AttemptTicket.model_validate_json(json.dumps(ticket)).plan
    guard = ResourceGuard(
        ResourceBudget(10_000_000, 10_000_000, 100000),
        tmp_path,
        time.monotonic() + 30,
        threading.Event(),
    )
    return plan, guard


@pytest.mark.parametrize("frames", [16000, 16001, 20003])
@pytest.mark.parametrize("polarity", [1, -1])
def test_noise_reduction_keeps_stereo_shape_polarity_and_partial_tail(
    tmp_path, processing, frames, polarity
):
    plan, guard = processing
    noise = np.random.default_rng(42).normal(0, 0.02, frames)
    source = tmp_path / "source.wav"
    destination = tmp_path / "clean.wav"
    sf.write(source, np.column_stack((noise, noise * polarity)), 48000, subtype="FLOAT")
    session = NoiseProfileEngine(plan.runtime, ReferenceAnalyser()).open_session(plan, guard)
    session.process(source, destination, plan, guard)
    output, rate = sf.read(destination, always_2d=True)
    assert sf.info(destination).format == "RF64"
    assert rate == 48000 and output.shape == (frames, 2)
    assert np.isfinite(output).all()
    np.testing.assert_allclose(output[:, 0], output[:, 1] * polarity, atol=1e-8)
    ratio = np.sqrt(np.mean(output[:, 0] ** 2) / np.mean(noise**2))
    assert 0.70 < ratio < 0.98
    # The tail remains audible rather than being truncated or zero-filled.
    assert np.max(np.abs(output[-20:])) > 0.001
    session.close()


@pytest.mark.parametrize("speech,music", [(True, False), (False, True)])
def test_contaminated_reference_rejected_before_output(tmp_path, processing, speech, music):
    plan, guard = processing
    source = tmp_path / "source.wav"
    output = tmp_path / "clean.wav"
    sf.write(source, np.ones((16000, 2)) * 0.1, 48000, subtype="FLOAT")
    engine = NoiseProfileEngine(plan.runtime, ReferenceAnalyser(speech=speech, music=music))
    with pytest.raises(CleanExecutionError):
        engine.open_session(plan, guard).process(source, output, plan, guard)
    assert not output.exists()


def test_silent_noise_reference_has_no_fake_success(tmp_path, processing):
    plan, guard = processing
    source = tmp_path / "source.wav"
    output = tmp_path / "clean.wav"
    sf.write(source, np.zeros((16000, 2)), 48000, subtype="FLOAT")
    session = NoiseProfileEngine(plan.runtime, ReferenceAnalyser()).open_session(plan, guard)
    with pytest.raises(CleanExecutionError):
        session.process(source, output, plan, guard)
    assert not output.exists()


def test_cancelled_processing_does_not_create_output(tmp_path, processing):
    plan, guard = processing
    session = NoiseProfileEngine(plan.runtime, ReferenceAnalyser()).open_session(plan, guard)
    guard.cancelled.set()
    with pytest.raises(CleanExecutionError):
        session.process(tmp_path / "source.wav", tmp_path / "clean.wav", plan, guard)
    assert not (tmp_path / "clean.wav").exists()


@pytest.mark.parametrize(
    "field", ["runtime_sha256", "precision_policy_sha256", "longform_policy_sha256"]
)
def test_unrelated_runtime_descriptor_cannot_label_noise_processing(processing, field):
    plan, _ = processing
    forged = plan.runtime.model_copy(update={field: "0" * 64})
    with pytest.raises(CleanExecutionError) as error:
        NoiseProfileEngine(forged, ReferenceAnalyser())
    assert error.value.code.value == "engine_unavailable"


@pytest.mark.parametrize(
    "change", ["reference", "numpy", "soundfile", "libsndfile", "window", "smoothing"]
)
def test_dependency_or_policy_drift_rejected_before_session(processing, monkeypatch, change):
    plan, guard = processing
    analyser = ReferenceAnalyser()
    engine = NoiseProfileEngine(plan.runtime, analyser)
    if change == "reference":
        analyser.policy_sha256 = "b" * 64
    elif change == "numpy":
        monkeypatch.setattr(np, "__version__", "unexpected")
    elif change == "soundfile":
        monkeypatch.setattr(sf, "__version__", "unexpected")
    elif change == "libsndfile":
        monkeypatch.setattr(sf, "__libsndfile_version__", "unexpected")
    elif change == "window":
        monkeypatch.setattr(NoiseProfileSession, "WINDOW", 4096)
    else:
        monkeypatch.setattr(NoiseProfileSession, "HISTORY_WEIGHT", 0.9)
    with pytest.raises(CleanExecutionError) as error:
        engine.open_session(plan, guard)
    assert error.value.code.value == "engine_unavailable"


@pytest.mark.parametrize("digest", ["b" * 64, "invalid", None])
def test_reference_policy_rechecked_before_processing(processing, tmp_path, digest):
    plan, guard = processing
    analyser = ReferenceAnalyser()
    session = NoiseProfileEngine(plan.runtime, analyser).open_session(plan, guard)
    analyser.policy_sha256 = digest
    with pytest.raises(CleanExecutionError) as error:
        session.process(tmp_path / "missing-source.wav", tmp_path / "output.wav", plan, guard)
    assert error.value.code.value == "engine_unavailable"
    assert not (tmp_path / "output.wav").exists()
