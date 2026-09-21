import json
import threading
import time

import numpy as np
import pytest
import soundfile as sf

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.services.magic_clean.contracts import AttemptTicket
from hear.services.magic_clean.quality import AudioQualityGate
from tests.test_cleaner_v2_contracts import ticket as ticket_fixture

ticket = ticket_fixture


@pytest.mark.parametrize("blocks", [5, 260])
def test_warning_locations_are_source_frames_and_bounded(tmp_path, ticket, blocks):
    plan = AttemptTicket.model_validate_json(json.dumps(ticket)).plan
    source, output = tmp_path / "in.wav", tmp_path / "out.wav"
    block = np.full((32768, 2), 0.1, dtype=np.float32)
    with (
        sf.SoundFile(source, "w", samplerate=48000, channels=2, subtype="FLOAT") as before,
        sf.SoundFile(output, "w", samplerate=48000, channels=2, subtype="FLOAT") as after,
    ):
        for index in range(blocks):
            before.write(block)
            after.write(block * (0.25 if index % 2 == 0 else 1))
    guard = ResourceGuard(
        ResourceBudget(200000000, 100000000, blocks * 32768),
        tmp_path,
        time.monotonic() + 30,
        threading.Event(),
    )
    summary = AudioQualityGate().evaluate(source, output, plan, guard)
    assert len(summary.source_warning_intervals) == min(128, (blocks + 1) // 2)
    assert summary.warning_intervals_truncated == (blocks > 256)
    for index, interval in enumerate(summary.source_warning_intervals):
        assert interval.start_frame == index * 2 * 32768
        assert interval.end_frame == interval.start_frame + 32768
        assert interval.minimum_rms_ratio == pytest.approx(0.25)
    assert summary.wanted_content == "review_required"


def test_adjacent_loss_blocks_merge_and_retain_partial_tail(tmp_path, ticket):
    plan = AttemptTicket.model_validate_json(json.dumps(ticket)).plan
    source, output = tmp_path / "in.wav", tmp_path / "out.wav"
    frames = 65539
    samples = np.full((frames, 2), 0.1, dtype=np.float32)
    sf.write(source, samples, 48000, subtype="FLOAT")
    sf.write(output, samples * 0.25, 48000, subtype="FLOAT")
    guard = ResourceGuard(
        ResourceBudget(2000000, 1000000, frames),
        tmp_path,
        time.monotonic() + 30,
        threading.Event(),
    )
    summary = AudioQualityGate().evaluate(source, output, plan, guard)
    assert len(summary.source_warning_intervals) == 1
    interval = summary.source_warning_intervals[0]
    assert interval.start_frame == 0
    assert interval.end_frame == frames
    assert not summary.warning_intervals_truncated
