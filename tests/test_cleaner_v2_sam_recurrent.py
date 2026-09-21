import threading
import time

import pytest
import torch

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.sam_recurrent import SamLSTMStream
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


@pytest.fixture
def recurrent(tmp_path):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        layer = torch.nn.LSTM(16, 16, num_layers=2).eval()
    guard = ResourceGuard(
        ResourceBudget(4096, 4096, 1000), tmp_path, time.monotonic() + 30, threading.Event()
    )
    yield layer, guard
    torch.set_num_threads(threads)


@pytest.mark.parametrize("batch", [1, 2])
def test_chunked_recurrence_matches_full_sequence_and_preserves_tail(recurrent, batch):
    layer, guard = recurrent
    features = torch.sin(torch.arange(batch * 16 * 103).reshape(batch, 16, 103).float())
    with torch.inference_mode():
        sequence = features.permute(2, 0, 1)
        full, _ = layer(sequence)
        expected = (full + sequence).permute(1, 2, 0)
    stream = SamLSTMStream(layer, max_frames=17, batch_size=batch)
    chunks = [
        stream.process(features[:, :, offset : offset + 17], offset, guard)
        for offset in range(0, 103, 17)
    ]
    actual = torch.cat(chunks, dim=2)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert actual.shape == features.shape
    assert stream.next_frame == 103
    assert stream._state[0].shape == (2, batch, 16)
    stream.close()
    assert stream._state is None


def test_chunk_identity_and_size_are_checked_before_state_advance(recurrent):
    layer, guard = recurrent
    stream = SamLSTMStream(layer, max_frames=8, batch_size=1)
    for features, offset, code in [
        (torch.zeros(1, 16, 9), 0, ErrorCode.INVALID_AUDIO),
        (torch.zeros(1, 16, 8), 1, ErrorCode.ARTIFACT_CONFLICT),
        (torch.full((1, 16, 8), float("nan")), 0, ErrorCode.INVALID_AUDIO),
    ]:
        with pytest.raises(CleanExecutionError) as error:
            stream.process(features, offset, guard)
        assert error.value.code == code
        assert stream.next_frame == 0
    stream.process(torch.zeros(1, 16, 8), 0, guard)
    with pytest.raises(CleanExecutionError):
        stream.process(torch.zeros(1, 16, 8), 0, guard)


def test_post_call_cancellation_discards_state(recurrent):
    layer, guard = recurrent
    stream = SamLSTMStream(layer, max_frames=8, batch_size=1)
    hook = layer.register_forward_hook(lambda *args: guard.cancelled.set())
    try:
        with pytest.raises(CleanExecutionError) as error:
            stream.process(torch.zeros(1, 16, 8), 0, guard)
        assert error.value.code == ErrorCode.CANCELLED
        assert stream.closed and stream._state is None
    finally:
        hook.remove()


def test_streams_do_not_share_recurrent_state(recurrent):
    layer, guard = recurrent
    first = SamLSTMStream(layer, max_frames=8, batch_size=1)
    second = SamLSTMStream(layer, max_frames=8, batch_size=1)
    first.process(torch.ones(1, 16, 8), 0, guard)
    assert second._state is None and second.next_frame == 0
    first.close()
    with pytest.raises(CleanExecutionError):
        first.process(torch.zeros(1, 16, 8), 8, guard)


def test_bidirectional_model_cannot_be_silently_streamed():
    with pytest.raises(ValueError):
        SamLSTMStream(torch.nn.LSTM(16, 16, bidirectional=True).eval(), max_frames=8, batch_size=1)


def test_caller_autocast_does_not_change_precision(recurrent):
    layer, guard = recurrent
    first = SamLSTMStream(layer, max_frames=8, batch_size=1)
    second = SamLSTMStream(layer, max_frames=8, batch_size=1)
    features = torch.ones(1, 16, 8)
    expected = first.process(features, 0, guard)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = second.process(features, 0, guard)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
