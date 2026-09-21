import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.sam_codec_graph import SamCodecGraph
from hear.runtime.cleaner.sam_features import SamFeatureFile
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


@pytest.fixture
def source(tmp_path):
    guard = ResourceGuard(
        ResourceBudget(100000, 100000, 1000), tmp_path, time.monotonic() + 30, threading.Event()
    )
    feature = SamFeatureFile(
        tmp_path / "source", frames=103, batch=1, channels=2, guard=guard, create=True
    )
    feature.write(0, np.sin(np.arange(206, dtype=np.float32)).reshape(1, 2, 103))
    yield feature
    feature.close()


def test_composed_sequence_matches_native_and_releases_intermediates(source):
    layer = torch.nn.Sequential(
        torch.nn.Conv1d(2, 3, 3, padding=1),
        torch.nn.ELU(),
        torch.nn.Sequential(torch.nn.Conv1d(3, 2, 3, padding=1), torch.nn.Tanh()),
    ).eval()
    values = source.read(0, source.frames)
    with torch.inference_mode():
        expected = layer(torch.from_numpy(values)).numpy()
    output = SamCodecGraph(tile_frames=7).run(layer, source)
    try:
        np.testing.assert_allclose(output.read(0, output.frames), expected, atol=1e-6, rtol=1e-6)
        assert set(source.path.parent.iterdir()) == {source.path, output.path}
        np.testing.assert_array_equal(source.read(0, source.frames), values)
    finally:
        output.close(remove=True)


@pytest.mark.parametrize("cancel", [False, True])
def test_failed_graph_removes_every_intermediate_preserves_input(source, cancel):
    layer = torch.nn.Sequential(torch.nn.ELU(), torch.nn.Tanh()).eval()

    def fail(module, args, result):
        if cancel:
            source.guard.cancelled.set()
            return result
        return torch.full_like(result, float("nan"))

    hook = layer[1].register_forward_hook(fail)
    try:
        with pytest.raises(CleanExecutionError) as error:
            SamCodecGraph(tile_frames=7).run(layer, source)
        assert error.value.code == (ErrorCode.CANCELLED if cancel else ErrorCode.INVALID_AUDIO)
        assert list(source.path.parent.iterdir()) == [source.path]
    finally:
        hook.remove()


def test_unknown_module_fails_without_retaining_prior_stage(source):
    layer = torch.nn.Sequential(torch.nn.ELU(), torch.nn.BatchNorm1d(2)).eval()
    with pytest.raises(ValueError):
        SamCodecGraph(tile_frames=7).run(layer, source)
    assert list(source.path.parent.iterdir()) == [source.path]


def test_empty_sequence_returns_independently_owned_copy(source):
    output = SamCodecGraph(tile_frames=7).run(torch.nn.Sequential().eval(), source)
    assert output is not source
    np.testing.assert_array_equal(output.read(0, output.frames), source.read(0, source.frames))
    output.close(remove=True)
    assert source.path.exists()


@pytest.mark.parametrize("projection_channels", [256, 255])
def test_encode_uses_mean_without_sampling_and_cleans_intermediates(tmp_path, projection_channels):
    guard = ResourceGuard(
        ResourceBudget(1000000, 1000000, 8192), tmp_path, time.monotonic() + 30, threading.Event()
    )
    source = SamFeatureFile(
        tmp_path / "waveform", frames=3841, batch=1, channels=1, guard=guard, create=True
    )
    values = np.sin(np.arange(3841, dtype=np.float32)).reshape(1, 1, 3841)
    source.write(0, values)
    codec = SimpleNamespace(
        encoder=torch.nn.Conv1d(1, 4, 1920, stride=1920).eval(),
        quantizer=SimpleNamespace(in_proj=torch.nn.Conv1d(4, projection_channels, 1).eval()),
    )
    state = torch.random.get_rng_state().clone()
    output = None
    try:
        if projection_channels != 256:
            with pytest.raises(ValueError):
                SamCodecGraph(tile_frames=257).encode_mean(codec, source)
            assert set(tmp_path.iterdir()) == {source.path}
        else:
            with torch.inference_mode():
                padded = torch.nn.functional.pad(
                    torch.from_numpy(values), (0, 1919), mode="reflect"
                )
                expected = codec.quantizer.in_proj(codec.encoder(padded))[:, :128, :].numpy()
            output = SamCodecGraph(tile_frames=257).encode_mean(codec, source)
            np.testing.assert_allclose(
                output.read(0, output.frames), expected, atol=1e-6, rtol=1e-5
            )
            assert set(tmp_path.iterdir()) == {source.path, output.path}
        assert torch.equal(state, torch.random.get_rng_state())
        np.testing.assert_array_equal(source.read(0, source.frames), values)
    finally:
        if output is not None:
            output.close(remove=True)
        source.close(remove=True)


def test_decode_latent_shape_rejected_before_model_access(source):
    with pytest.raises(ValueError):
        SamCodecGraph(tile_frames=7).decode_latents(None, source, frames=103, message=None)
    assert list(source.path.parent.iterdir()) == [source.path]


def test_failed_joint_decode_removes_pair_but_preserves_joint_input(tmp_path, monkeypatch):
    guard = ResourceGuard(
        ResourceBudget(100000, 100000, 1000), tmp_path, time.monotonic() + 30, threading.Event()
    )
    source = SamFeatureFile(
        tmp_path / "joint", frames=3, batch=1, channels=256, guard=guard, create=True
    )
    values = np.ones((1, 256, 3), np.float32)
    source.write(0, values)

    def fail(*args, **kwargs):
        raise CleanExecutionError(ErrorCode.CANCELLED, "fixture cancellation")

    monkeypatch.setattr(SamCodecGraph, "decode_latents", fail)
    try:
        with pytest.raises(CleanExecutionError):
            SamCodecGraph(tile_frames=7).decode_joint(None, source, frames=5760, message=None)
        assert set(tmp_path.iterdir()) == {source.path}
        np.testing.assert_array_equal(source.read(0, source.frames), values)
    finally:
        source.close(remove=True)
