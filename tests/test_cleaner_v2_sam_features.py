import threading
import time

import numpy as np
import pytest
import torch

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.sam_features import SamFeatureFile, SamFeatureRunner
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


@pytest.fixture
def features(tmp_path):
    guard = ResourceGuard(
        ResourceBudget(100000, 100000, 1000), tmp_path, time.monotonic() + 30, threading.Event()
    )
    source = SamFeatureFile(
        tmp_path / "input", frames=103, batch=1, channels=2, guard=guard, create=True
    )
    values = np.sin(np.arange(206, dtype=np.float32)).reshape(1, 2, 103)
    source.write(0, values)
    yield source, values
    source.close()


def test_disk_layout_and_read_copy_ownership(features):
    source, values = features
    raw = np.fromfile(source.path, dtype=np.float32).reshape(103, 1, 2)
    np.testing.assert_array_equal(raw, values.transpose(2, 0, 1))
    read = source.read(3, 13)
    read[:] = 0
    np.testing.assert_array_equal(source.read(3, 13), values[:, :, 3:13])


@pytest.mark.parametrize("transposed", [False, True])
def test_disk_convolution_preserves_complete_tail(features, transposed):
    source, values = features
    cls = torch.nn.ConvTranspose1d if transposed else torch.nn.Conv1d
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12)
        layer = cls(2, 3, 4, stride=2, padding=1).eval()
    with torch.inference_mode():
        expected = layer(torch.from_numpy(values)).numpy()
    output = SamFeatureRunner.convolution(
        layer, source, source.path.parent / "output", tile_frames=7
    )
    try:
        np.testing.assert_allclose(output.read(0, output.frames), expected, atol=1e-6, rtol=1e-6)
    finally:
        output.close()


def test_existing_destination_preserved(features):
    source, _ = features
    path = source.path.parent / "keep"
    path.write_bytes(b"keep")
    with pytest.raises(CleanExecutionError) as error:
        SamFeatureRunner.convolution(
            torch.nn.Conv1d(2, 3, 3, padding=1).eval(), source, path, tile_frames=7
        )
    assert error.value.code == ErrorCode.ARTIFACT_CONFLICT
    assert path.read_bytes() == b"keep"


def test_cancelled_native_work_removes_only_created_output(features):
    source, _ = features
    layer = torch.nn.Conv1d(2, 3, 3, padding=1).eval()
    hook = layer.register_forward_hook(lambda *args: source.guard.cancelled.set())
    path = source.path.parent / "output"
    try:
        with pytest.raises(CleanExecutionError) as error:
            SamFeatureRunner.convolution(layer, source, path, tile_frames=7)
        assert error.value.code == ErrorCode.CANCELLED
        assert not path.exists()
        assert source.path.exists()
    finally:
        hook.remove()


def test_incomplete_feature_output_cannot_be_consumed(features):
    source, _ = features
    output = SamFeatureFile(
        source.path.parent / "incomplete",
        frames=10,
        batch=1,
        channels=2,
        guard=source.guard,
        create=True,
    )
    try:
        with pytest.raises(CleanExecutionError):
            output.read(0, 1)
        with pytest.raises(CleanExecutionError):
            output.write(1, np.zeros((1, 2, 1), np.float32))
        output.write(0, np.ones((1, 2, 5), np.float32))
        with pytest.raises(CleanExecutionError):
            output.read(0, 6)
    finally:
        output.close(remove=True)


def test_readonly_file_cannot_be_written_or_removed(features):
    source, _ = features
    reader = SamFeatureFile(
        source.path, frames=source.frames, batch=1, channels=2, guard=source.guard
    )
    try:
        with pytest.raises(CleanExecutionError):
            reader.write(0, np.zeros((1, 2, 1), np.float32))
    finally:
        reader.close(remove=True)
    assert source.path.exists()


def test_scratch_reservation_rejects_before_creation(features):
    source, _ = features
    path = source.path.parent / "oversize"
    with pytest.raises(CleanExecutionError) as error:
        SamFeatureFile(path, frames=1000, batch=2, channels=100, guard=source.guard, create=True)
    assert error.value.code == ErrorCode.RESOURCE_EXHAUSTED
    assert not path.exists()


def test_tile_limit_is_checked_before_model_call(features, monkeypatch):
    source, _ = features
    monkeypatch.setattr(SamFeatureFile, "MAX_TILE_BYTES", 16)
    with pytest.raises(CleanExecutionError) as error:
        SamFeatureRunner.convolution(
            torch.nn.Conv1d(2, 3, 3, padding=1).eval(),
            source,
            source.path.parent / "output",
            tile_frames=7,
        )
    assert error.value.code == ErrorCode.RESOURCE_EXHAUSTED
    assert not (source.path.parent / "output").exists()


@pytest.mark.parametrize("failure", ["nonfinite", "exception"])
def test_failed_native_work_removes_output_and_preserves_input(features, failure):
    source, values = features
    layer = torch.nn.Conv1d(2, 3, 3, padding=1).eval()

    def fail(module, args, result):
        if failure == "exception":
            raise RuntimeError("native fixture failure")
        return torch.full_like(result, float("nan"))

    hook = layer.register_forward_hook(fail)
    path = source.path.parent / "failed"
    try:
        with pytest.raises((CleanExecutionError, RuntimeError)):
            SamFeatureRunner.convolution(layer, source, path, tile_frames=7)
        assert not path.exists()
        np.testing.assert_array_equal(source.read(0, source.frames), values)
    finally:
        hook.remove()


@pytest.mark.parametrize("transposed", [False, True])
def test_halo_limit_rejects_before_native_call(features, monkeypatch, transposed):
    source, _ = features
    monkeypatch.setattr(SamFeatureFile, "MAX_TILE_BYTES", 32)
    cls = torch.nn.ConvTranspose1d if transposed else torch.nn.Conv1d
    layer = cls(2, 2, 17, padding=8).eval()
    calls = []
    hook = layer.register_forward_pre_hook(lambda *args: calls.append(True))
    path = source.path.parent / "halo"
    try:
        with pytest.raises(CleanExecutionError) as error:
            SamFeatureRunner.convolution(layer, source, path, tile_frames=1)
        assert error.value.code == ErrorCode.RESOURCE_EXHAUSTED
        assert calls == []
        assert not path.exists()
    finally:
        hook.remove()


def test_feature_file_rejects_path_outside_workspace(features, tmp_path):
    source, _ = features
    path = tmp_path.parent / (tmp_path.name + "-outside")
    with pytest.raises(CleanExecutionError) as error:
        SamFeatureFile(path, frames=1, batch=1, channels=2, guard=source.guard, create=True)
    assert error.value.code == ErrorCode.INVALID_AUDIO
    assert not path.exists()


def test_existing_malformed_input_is_preserved(features):
    source, _ = features
    path = source.path.parent / "malformed"
    path.write_bytes(b"bad")
    with pytest.raises(CleanExecutionError) as error:
        SamFeatureFile(path, frames=1, batch=1, channels=2, guard=source.guard)
    assert error.value.code == ErrorCode.INVALID_AUDIO
    assert path.read_bytes() == b"bad"


@pytest.mark.parametrize("tile_frames", [1, 7, 64])
def test_disk_recurrence_preserves_residual_and_partial_tail(features, tile_frames):
    source, values = features
    layer = torch.nn.LSTM(2, 2, num_layers=2).eval()
    with torch.inference_mode():
        sequence = torch.from_numpy(values).permute(2, 0, 1)
        predicted, _ = layer(sequence)
        expected = (predicted + sequence).permute(1, 2, 0).numpy()
    output = SamFeatureRunner.recurrent(
        layer, source, source.path.parent / "recurrent", tile_frames=tile_frames
    )
    try:
        np.testing.assert_allclose(output.read(0, output.frames), expected, atol=1e-6, rtol=1e-6)
    finally:
        output.close(remove=True)


def test_disk_recurrence_cancellation_clears_state_and_output(features, monkeypatch):
    from hear.runtime.cleaner.sam_recurrent import SamLSTMStream

    source, _ = features
    closed = []
    original = SamLSTMStream.close

    def close(stream):
        original(stream)
        closed.append(stream)

    monkeypatch.setattr(SamLSTMStream, "close", close)
    layer = torch.nn.LSTM(2, 2).eval()
    hook = layer.register_forward_hook(lambda *args: source.guard.cancelled.set())
    destination = source.path.parent / "cancelled-recurrent"
    try:
        with pytest.raises(CleanExecutionError) as error:
            SamFeatureRunner.recurrent(layer, source, destination, tile_frames=7)
        assert error.value.code == ErrorCode.CANCELLED
        assert closed and all(stream.closed and stream._state is None for stream in closed)
        assert not destination.exists()
        assert source.path.exists()
    finally:
        hook.remove()


def test_disk_recurrence_rejects_bidirectional_before_file_creation(features):
    source, _ = features
    destination = source.path.parent / "bidirectional"
    with pytest.raises(ValueError):
        SamFeatureRunner.recurrent(
            torch.nn.LSTM(2, 2, bidirectional=True).eval(), source, destination, tile_frames=7
        )
    assert not destination.exists()


def test_disk_recurrence_preserves_existing_destination(features):
    source, _ = features
    destination = source.path.parent / "existing-recurrent"
    destination.write_bytes(b"keep")
    with pytest.raises(CleanExecutionError) as error:
        SamFeatureRunner.recurrent(torch.nn.LSTM(2, 2).eval(), source, destination, tile_frames=7)
    assert error.value.code == ErrorCode.ARTIFACT_CONFLICT
    assert destination.read_bytes() == b"keep"


@pytest.mark.parametrize("cls", [torch.nn.ELU, torch.nn.Tanh, torch.nn.Identity])
def test_disk_activation_matches_full_output(features, cls):
    source, values = features
    layer = cls().eval()
    with torch.inference_mode():
        expected = layer(torch.from_numpy(values)).numpy()
    output = SamFeatureRunner.activation(
        layer, source, source.path.parent / "activation", tile_frames=7
    )
    try:
        np.testing.assert_allclose(output.read(0, output.frames), expected, atol=1e-7, rtol=1e-6)
        np.testing.assert_array_equal(source.read(0, source.frames), values)
    finally:
        output.close(remove=True)


def test_temporal_module_cannot_be_used_as_activation(features):
    source, _ = features
    path = source.path.parent / "invalid-activation"
    with pytest.raises(ValueError):
        SamFeatureRunner.activation(torch.nn.Conv1d(2, 2, 3).eval(), source, path, tile_frames=7)
    assert not path.exists()


@pytest.mark.parametrize("crop", [0, 4])
def test_residual_preserves_centered_shortcut_and_tail(features, crop):
    shortcut, values = features
    branch = SamFeatureFile(
        shortcut.path.parent / "branch",
        frames=103 - crop,
        batch=1,
        channels=2,
        guard=shortcut.guard,
        create=True,
    )
    branch_values = np.full((1, 2, 103 - crop), 0.125, np.float32)
    branch.write(0, branch_values)
    output = SamFeatureRunner.residual(
        branch, shortcut, shortcut.path.parent / "sum", tile_frames=7
    )
    try:
        np.testing.assert_array_equal(
            output.read(0, output.frames), branch_values + values[..., crop // 2 : 103 - crop // 2]
        )
        np.testing.assert_array_equal(shortcut.read(0, shortcut.frames), values)
    finally:
        output.close(remove=True)
        branch.close(remove=True)


@pytest.mark.parametrize("frames,true_skip", [(102, False), (99, True), (104, False)])
def test_residual_rejects_incompatible_lengths_before_creation(features, frames, true_skip):
    shortcut, _ = features
    branch = SamFeatureFile(
        shortcut.path.parent / "branch",
        frames=frames,
        batch=1,
        channels=2,
        guard=shortcut.guard,
        create=True,
    )
    path = shortcut.path.parent / "bad-sum"
    try:
        with pytest.raises(ValueError):
            SamFeatureRunner.residual(branch, shortcut, path, tile_frames=7, true_skip=true_skip)
        assert not path.exists()
    finally:
        branch.close(remove=True)


def test_failed_activation_removes_output_preserves_input(features):
    source, values = features
    layer = torch.nn.Tanh().eval()
    hook = layer.register_forward_hook(
        lambda module, args, output: torch.full_like(output, float("nan"))
    )
    path = source.path.parent / "nonfinite-activation"
    try:
        with pytest.raises(CleanExecutionError):
            SamFeatureRunner.activation(layer, source, path, tile_frames=7)
        assert not path.exists()
        np.testing.assert_array_equal(source.read(0, source.frames), values)
    finally:
        hook.remove()


def test_watermark_blend_preserves_scale_inputs_and_tail(features):
    source, values = features
    output = SamFeatureRunner.blend(
        source, source, source.path.parent / "blend", alpha=0.25, tile_frames=7
    )
    try:
        np.testing.assert_array_equal(
            output.read(0, output.frames), values + np.float32(0.25) * values
        )
        np.testing.assert_array_equal(source.read(0, source.frames), values)
    finally:
        output.close(remove=True)


@pytest.mark.parametrize("alpha", [0, 1, float("nan"), 0.5])
def test_watermark_cannot_be_disabled_or_rescaled(features, alpha):
    source, _ = features
    path = source.path.parent / "invalid-blend"
    with pytest.raises(ValueError):
        SamFeatureRunner.blend(source, source, path, alpha=alpha, tile_frames=7)
    assert not path.exists()


def test_watermark_blend_failure_preserves_preexisting_output(features):
    source, _ = features
    path = source.path.parent / "existing-blend"
    path.write_bytes(b"keep")
    with pytest.raises(CleanExecutionError) as error:
        SamFeatureRunner.blend(source, source, path, alpha=0.25, tile_frames=7)
    assert error.value.code == ErrorCode.ARTIFACT_CONFLICT
    assert path.read_bytes() == b"keep"


def test_unknown_message_processor_rejected_before_output(features):
    source, _ = features
    path = source.path.parent / "invalid-message"
    with pytest.raises(ValueError):
        SamFeatureRunner.message(
            torch.nn.Identity().eval(), torch.zeros(1, 16), source, path, tile_frames=7
        )
    assert not path.exists()


@pytest.mark.parametrize("frames", [960, 961, 1919, 1920, 1921, 3839])
@pytest.mark.parametrize("tile_frames", [127, 1024])
def test_reflect_padding_matches_original_torch_policy(tmp_path, frames, tile_frames):
    guard = ResourceGuard(
        ResourceBudget(1000000, 1000000, 6000), tmp_path, time.monotonic() + 30, threading.Event()
    )
    source = SamFeatureFile(
        tmp_path / "waveform", frames=frames, batch=2, channels=1, guard=guard, create=True
    )
    values = np.sin(np.arange(2 * frames, dtype=np.float32)).reshape(2, 1, frames)
    source.write(0, values)
    padding = (-frames) % 1920
    try:
        if padding >= frames:
            with pytest.raises(CleanExecutionError) as error:
                SamFeatureRunner.reflect_pad(source, tmp_path / "padded", tile_frames=tile_frames)
            assert error.value.code == ErrorCode.INVALID_AUDIO
            assert not (tmp_path / "padded").exists()
        else:
            expected = torch.nn.functional.pad(
                torch.from_numpy(values), (0, padding), mode="reflect"
            ).numpy()
            output = SamFeatureRunner.reflect_pad(
                source, tmp_path / "padded", tile_frames=tile_frames
            )
            try:
                np.testing.assert_array_equal(output.read(0, output.frames), expected)
            finally:
                output.close(remove=True)
        np.testing.assert_array_equal(source.read(0, source.frames), values)
    finally:
        source.close(remove=True)


def test_feature_selection_preserves_frame_and_channel_offsets(features):
    source, values = features
    output = SamFeatureRunner.select(
        source,
        source.path.parent / "selected",
        frame_start=3,
        frame_end=102,
        channel_start=1,
        channel_end=2,
        tile_frames=7,
    )
    try:
        np.testing.assert_array_equal(output.read(0, output.frames), values[:, 1:2, 3:102])
    finally:
        output.close(remove=True)


@pytest.mark.parametrize("start,end", [(0, 104), (-1, 10), (5, 5)])
def test_invalid_feature_selection_does_not_create_output(features, start, end):
    source, _ = features
    path = source.path.parent / "invalid-selection"
    with pytest.raises(ValueError):
        SamFeatureRunner.select(
            source,
            path,
            frame_start=start,
            frame_end=end,
            channel_start=0,
            channel_end=1,
            tile_frames=7,
        )
    assert not path.exists()


@pytest.mark.parametrize("tile", [1, 7, 17])
def test_joint_latents_preserve_target_residual_order_and_tail(tmp_path, tile):
    guard = ResourceGuard(
        ResourceBudget(100000, 100000, 1000), tmp_path, time.monotonic() + 30, threading.Event()
    )
    source = SamFeatureFile(
        tmp_path / "joint", frames=19, batch=1, channels=256, guard=guard, create=True
    )
    values = np.arange(256 * 19, dtype=np.float32).reshape(1, 256, 19)
    source.write(0, values)
    output = SamFeatureRunner.paired_latents(source, tmp_path / "paired", tile_frames=tile)
    try:
        expected = torch.from_numpy(values).reshape(2, 128, 19).numpy()
        np.testing.assert_array_equal(output.read(0, output.frames), expected)
        np.testing.assert_array_equal(source.read(0, source.frames), values)
        assert output.batch == 2 and output.channels == 128
    finally:
        output.close(remove=True)
        source.close(remove=True)


def test_invalid_joint_layout_is_rejected_before_output(features):
    source, _ = features
    path = source.path.parent / "bad-pair"
    with pytest.raises(ValueError):
        SamFeatureRunner.paired_latents(source, path, tile_frames=7)
    assert not path.exists()
