"""Regressions for deliberately retained events and introduced tonal artifacts."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.services.sound_cleanup.preview_integrity import PreviewIntegrity
from hear.services.sound_cleanup.separator import EventSeparator


def samples(seconds=2):
    t = np.arange(round(48000 * seconds)) / 48000
    voice = 0.08 * np.sin(2 * np.pi * 190 * t) + 0.04 * np.sin(2 * np.pi * 570 * t)
    return t, voice[:, None].astype("float32")


@pytest.mark.parametrize("frequency", [50, 60, 100, 120, 1000])
def test_rejects_added_persistent_tone(frequency):
    t, before = samples()
    after = before + 0.01 * np.sin(2 * np.pi * frequency * t)[:, None]
    reason, report = PreviewIntegrity.assess(before, after)
    assert reason == "introduced_tonal_artifact"
    assert report["introduced_tones"]


@pytest.mark.parametrize("channels", [1, 2])
def test_does_not_label_existing_voice_harmonics_or_hum_as_new_artifacts(channels):
    t, source = samples()
    source = source + 0.005 * np.sin(2 * np.pi * 60 * t)[:, None]
    source = np.repeat(source, channels, axis=1)
    reason, report = PreviewIntegrity.assess(source, source * 0.8)
    assert reason == "" and not report["introduced_tones"]


def test_rejects_tone_in_one_stereo_channel():
    t, source = samples()
    before = np.repeat(source, 2, axis=1)
    after = before.copy()
    after[:, 1] += 0.01 * np.sin(2 * np.pi * 50 * t)
    reason, report = PreviewIntegrity.assess(before, after)
    assert reason == "introduced_tonal_artifact"
    assert {item["channel"] for item in report["introduced_tones"]} == {1}


def test_rejects_added_dc_offset():
    _, before = samples()
    assert PreviewIntegrity.assess(before, before + 0.02)[0] == "introduced_dc_artifact"


@pytest.mark.parametrize("corruption", ["nan", "length", "channels", "empty"])
def test_rejects_invalid_candidate(corruption):
    _, before = samples()
    after = before.copy()
    if corruption == "nan":
        after[40] = np.nan
    elif corruption == "length":
        after = after[:-1]
    elif corruption == "channels":
        after = np.repeat(after, 2, axis=1)
    else:
        before, after = before[:0], after[:0]
    assert PreviewIntegrity.assess(before, after)[0] == "preview_integrity_invalid_samples"


def test_silent_and_short_valid_audio():
    for frames in (1, 240, 96000):
        source = np.zeros((frames, 1))
        assert PreviewIntegrity.assess(source, source)[0] == ""


def test_full_estimate_is_subtracted_not_deliberately_left_at_fifteen_percent(tmp_path):
    """Deterministic model stub tests the integration coefficient, not model quality."""
    t = np.arange(480000) / 48000
    speech = 0.04 * np.sin(2 * np.pi * 200 * t)
    event = 0.06 * np.sin(2 * np.pi * 1000 * t)
    sf.write(tmp_path / "source.wav", speech + event, 48000, subtype="FLOAT")
    separator = EventSeparator.__new__(EventSeparator)
    separator._load = lambda guard: None
    separator.device = "cpu"
    separator.digest = "a" * 64
    separator.queries = {"animal": [0.0] * 512}
    target = torch.from_numpy(
        (0.06 * np.sin(2 * np.pi * 1000 * np.arange(320000) / 32000)).astype("float32")
    )
    separator.model = lambda waveform, query: target.reshape(1, 1, -1)

    class SpeechChecks:
        def _vad(self, path, guard):
            count = (sf.info(path).frames + 1535) // 1536
            value = 0.95 if "speech-check" in path.name else 0.0
            return np.full((count, 1), value, dtype="float32"), None

    region = SimpleNamespace(start=192000, end=240000, kind="animal")
    evidence = SimpleNamespace(speech=np.ones(313))
    guard = ResourceGuard(
        ResourceBudget(100_000_000, 10_000_000, 500000),
        tmp_path,
        time.monotonic() + 30,
        threading.Event(),
    )
    with sf.SoundFile(tmp_path / "source.wav") as source:
        candidate, reason, metrics = separator.propose(
            source, region, evidence, SpeechChecks(), guard
        )
    assert reason == "" and candidate is not None
    wanted = speech[region.start : region.end]
    residual = candidate[:, 0] - wanted
    assert np.sqrt(np.mean(residual**2)) < 0.0002
    assert metrics["target_subtraction_gain"] == 1.0
    assert metrics["artifact_check"]["assessment"] == "passed"
    assert not list(tmp_path.glob("separator-*-check.wav"))
