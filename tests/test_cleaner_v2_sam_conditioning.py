import threading
import time

import numpy as np
import pytest
import torch

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.sam_conditioning import AudioOnlySamForward, SamConditionedField
from hear.runtime.cleaner.sam_features import SamFeatureFile
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


class AlignmentFixture(torch.nn.Module):
    def forward(self, anchor, video):
        assert video.shape == (1, 1024, anchor.shape[1])
        assert not video.any()
        return anchor + 0.25


class AnchorFixture(torch.nn.Module):
    def forward(self, features, ids, alignment):
        assert ids is None and alignment is None
        return features


class TimeFixture(torch.nn.Module):
    def forward(self, time, pos):
        assert pos is time
        return time[:, None].expand(1, 1536)


class TransformerFixture(torch.nn.Module):
    def forward(self, aligned, time, *, padding_mask, memory, memory_padding_mask):
        assert padding_mask is None
        assert memory_padding_mask.dtype == torch.bool
        return aligned[..., :256] + memory[..., :256].mean(dim=1, keepdim=True)


@pytest.fixture
def setup(tmp_path):
    core = torch.nn.Module()
    core.proj = torch.nn.Linear(768, 1536)
    core.memory_proj = torch.nn.Linear(768, 1536)
    core.align_masked_video = AlignmentFixture()
    core.embed_anchors = AnchorFixture()
    core.timestep_emb = TimeFixture()
    core.transformer = TransformerFixture()
    core.eval()
    guard = ResourceGuard(
        ResourceBudget(100000, 100000, 1000), tmp_path, time.monotonic() + 30, threading.Event()
    )
    inputs = [
        torch.ones(1, 3, 256),
        torch.full((1, 3, 128), 0.5),
        torch.ones(1, 2, 768),
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([0.5]),
    ]
    return core, guard, inputs


def test_conditioning_retains_alignment_and_text(setup):
    core, guard, inputs = setup
    state, mean, text, mask, time_value = inputs
    with torch.inference_mode():
        audio = torch.cat((mean, mean), dim=2)
        expected = (
            core.proj(torch.cat((state, torch.zeros_like(audio), audio), dim=2))[..., :256]
            + 0.25
            + (core.memory_proj(text)[..., :256] + 0.5).mean(dim=1, keepdim=True)
        )
    actual = AudioOnlySamForward(core, guard).forward(*inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("invalid", ["window", "mean", "zero_text", "mask", "dtype", "time", "nan"])
def test_invalid_conditioning_is_rejected(setup, invalid):
    core, guard, inputs = setup
    if invalid == "window":
        inputs[0] = torch.ones(1, 251, 256)
    elif invalid == "mean":
        inputs[1] = torch.ones(1, 3, 256)
    elif invalid == "zero_text":
        inputs[2].zero_()
    elif invalid == "mask":
        inputs[3].zero_()
    elif invalid == "dtype":
        inputs[0] = inputs[0].half()
    elif invalid == "time":
        inputs[4].fill_(1.1)
    else:
        inputs[1].fill_(float("nan"))
    with pytest.raises(CleanExecutionError) as error:
        AudioOnlySamForward(core, guard).forward(*inputs)
    assert error.value.code == ErrorCode.INVALID_AUDIO


def test_native_cancellation_returns_no_prediction(setup):
    core, guard, inputs = setup
    hook = core.transformer.register_forward_hook(lambda *args: guard.cancelled.set())
    try:
        with pytest.raises(CleanExecutionError) as error:
            AudioOnlySamForward(core, guard).forward(*inputs)
        assert error.value.code == ErrorCode.CANCELLED
    finally:
        hook.remove()


def test_training_core_is_rejected(setup):
    core, guard, inputs = setup
    core.transformer.train()
    with pytest.raises(ValueError):
        AudioOnlySamForward(core, guard).forward(*inputs)


def test_caller_autocast_does_not_change_conditioning_precision(setup):
    core, guard, inputs = setup
    adapter = AudioOnlySamForward(core, guard)
    expected = adapter.forward(*inputs)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = adapter.forward(*inputs)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.fixture
def field_setup(setup):
    core, guard, inputs = setup
    values = np.sin(np.arange(13 * 128, dtype=np.float32)).reshape(1, 128, 13)
    mean = SamFeatureFile(
        guard.workspace / "mean", frames=13, batch=1, channels=128, guard=guard, create=True
    )
    mean.write(0, values)
    adapter = AudioOnlySamForward(core, guard)
    field = SamConditionedField(adapter, mean, inputs[2], inputs[3])
    yield adapter, field, mean, values, inputs
    field.close()
    mean.close(remove=True)


def test_solver_binding_uses_global_feature_offset_and_owned_result(field_setup):
    adapter, field, mean, values, inputs = field_setup
    state = np.ones((3, 256), np.float32)
    expected = adapter.forward(
        torch.from_numpy(state).unsqueeze(0),
        torch.from_numpy(values[..., 7:10].transpose(0, 2, 1)),
        inputs[2],
        inputs[3],
        torch.tensor([0.5]),
    )[0].numpy()
    actual = field.evaluate(state, start_frame=7, time=0.5)
    np.testing.assert_array_equal(actual, expected)
    actual.fill(0)
    np.testing.assert_array_equal(field.evaluate(state, start_frame=7, time=0.5), expected)
    assert np.all(state == 1)


def test_solver_binding_freezes_text_and_borrows_mean(field_setup):
    adapter, field, mean, values, inputs = field_setup
    state = np.ones((3, 256), np.float32)
    expected = field.evaluate(state, start_frame=0, time=0.5)
    inputs[2].zero_()
    inputs[3].zero_()
    np.testing.assert_array_equal(field.evaluate(state, start_frame=0, time=0.5), expected)
    field.close()
    assert field.text is None and field.text_mask is None
    np.testing.assert_array_equal(mean.read(0, mean.frames), values)
    with pytest.raises(CleanExecutionError) as error:
        field.evaluate(state, start_frame=0, time=0.5)
    assert error.value.code == ErrorCode.ENGINE_UNAVAILABLE


@pytest.mark.parametrize("start,instant", [(-1, 0.5), (12, 0.5), (0, float("nan")), (0, 1.1)])
def test_solver_binding_rejects_out_of_range_windows(field_setup, start, instant):
    _, field, _, _, _ = field_setup
    with pytest.raises(CleanExecutionError) as error:
        field.evaluate(np.ones((3, 256), np.float32), start_frame=start, time=instant)
    assert error.value.code == ErrorCode.INVALID_AUDIO
