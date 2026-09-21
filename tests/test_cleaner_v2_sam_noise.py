import hashlib
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.sam_features import SamFeatureFile
from hear.runtime.cleaner.sam_noise import SamNoise, SamNoisePolicy
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


@pytest.fixture
def guard(tmp_path):
    return ResourceGuard(
        ResourceBudget(3000000, 3000000, 2000), tmp_path, time.monotonic() + 30, threading.Event()
    )


@pytest.mark.parametrize("tile", [1, 7, 128, 250, 1024])
def test_noise_bytes_do_not_depend_on_write_tiles(guard, tile):
    sampler = SamNoise()
    whole = sampler.create(
        guard.workspace / "whole", frames=257, seed=42, guard=guard, tile_frames=257
    )
    tiled = sampler.create(
        guard.workspace / "tiled", frames=257, seed=42, guard=guard, tile_frames=tile
    )
    try:
        assert whole.path.read_bytes() == tiled.path.read_bytes()
        assert whole.complete and tiled.complete
        assert np.isfinite(tiled.read(0, 257)).all()
    finally:
        whole.close(remove=True)
        tiled.close(remove=True)


def test_seed_and_message_streams_are_independent_and_repeatable(guard):
    sampler = SamNoise()
    expected = sampler.watermark(42)
    noise = sampler.create(guard.workspace / "noise", frames=257, seed=42, guard=guard)
    other = sampler.create(guard.workspace / "other", frames=257, seed=43, guard=guard)
    try:
        assert noise.path.read_bytes() != other.path.read_bytes()
        np.testing.assert_array_equal(sampler.watermark(42), expected)
        assert expected.shape == (2, 16) and expected.dtype == np.float32
        assert set(np.unique(expected)) <= {0, 1}
        assert sampler.identity(seed=42, frames=257) != sampler.identity(seed=43, frames=257)
        assert sampler.identity(seed=42, frames=257) != sampler.identity(seed=42, frames=258)
    finally:
        noise.close(remove=True)
        other.close(remove=True)


@pytest.mark.parametrize("seed", [-1, 2**63, True, 1.5])
def test_invalid_seed_rejected_before_creation(guard, seed):
    path = guard.workspace / "bad"
    with pytest.raises(ValueError):
        SamNoise().create(path, frames=1, seed=seed, guard=guard)
    assert not path.exists()


def test_rng_version_mismatch_fails_closed():
    with pytest.raises(CleanExecutionError) as error:
        SamNoise(SamNoisePolicy(numpy_version="0.0.0"))
    assert error.value.code == ErrorCode.ENGINE_UNAVAILABLE


@pytest.mark.parametrize("frames", [1, 257])
def test_cancelled_noise_write_removes_only_created_file(guard, monkeypatch, frames):
    keep = guard.workspace / "keep"
    keep.write_bytes(b"keep")
    original = SamFeatureFile.write

    def cancel(output, start, values):
        original(output, start, values)
        guard.cancelled.set()

    monkeypatch.setattr(SamFeatureFile, "write", cancel)
    path = guard.workspace / "noise"
    with pytest.raises(CleanExecutionError) as error:
        SamNoise().create(path, frames=frames, seed=42, guard=guard, tile_frames=7)
    assert error.value.code == ErrorCode.CANCELLED
    assert not path.exists() and keep.read_bytes() == b"keep"


def test_existing_noise_destination_preserved(guard):
    path = guard.workspace / "existing"
    path.write_bytes(b"keep")
    with pytest.raises(CleanExecutionError) as error:
        SamNoise().create(path, frames=1, seed=42, guard=guard)
    assert error.value.code == ErrorCode.ARTIFACT_CONFLICT
    assert path.read_bytes() == b"keep"


@pytest.mark.parametrize("value", [True, 1.5, 0, -1])
def test_invalid_frame_counts_cannot_create_files_or_identities(guard, value):
    sampler = SamNoise()
    path = guard.workspace / "invalid"
    with pytest.raises(ValueError):
        sampler.create(path, frames=value, seed=42, guard=guard)
    with pytest.raises(ValueError):
        sampler.identity(seed=42, frames=value)
    assert not path.exists()


@pytest.mark.parametrize("value", [True, 1.5, 0, 65537])
def test_invalid_noise_tiles_rejected_before_creation(guard, value):
    path = guard.workspace / "invalid"
    with pytest.raises(ValueError):
        SamNoise().create(path, frames=1, seed=42, guard=guard, tile_frames=value)
    assert not path.exists()


def test_allocation_failure_is_typed_and_removes_partial_noise(guard, monkeypatch):
    calls = []

    def allocate(shape, *, dtype):
        calls.append(shape)
        if len(calls) == 2:
            raise MemoryError("fixture allocation failure")
        return np.zeros(shape, dtype=dtype)

    monkeypatch.setattr(
        SamNoise,
        "_generator",
        staticmethod(lambda *args: SimpleNamespace(standard_normal=allocate)),
    )
    path = guard.workspace / "noise"
    keep = guard.workspace / "keep"
    keep.write_bytes(b"keep")
    with pytest.raises(CleanExecutionError) as error:
        SamNoise().create(path, frames=17, seed=42, guard=guard, tile_frames=7)
    assert error.value.code == ErrorCode.RESOURCE_EXHAUSTED
    assert len(calls) == 2 and not path.exists()
    assert keep.read_bytes() == b"keep"


def test_rng_state_allocation_failure_does_not_prevent_identity(guard, monkeypatch):
    def fail(*args, **kwargs):
        raise MemoryError("fixture RNG failure")

    monkeypatch.setattr(np.random, "PCG64", fail)
    sampler = SamNoise()
    assert (
        sampler.identity(seed=42, frames=257)
        == "304015eb64fc28c791c6408e401d0ad62b602567226f438f4898e24d306388a2"
    )
    path = guard.workspace / "noise"
    with pytest.raises(CleanExecutionError) as error:
        sampler.create(path, frames=1, seed=42, guard=guard)
    assert error.value.code == ErrorCode.RESOURCE_EXHAUSTED
    assert not path.exists()


def test_watermark_allocation_failure_is_typed(monkeypatch):
    def fail(*args, **kwargs):
        raise MemoryError("fixture message failure")

    monkeypatch.setattr(
        SamNoise, "_generator", staticmethod(lambda *args: SimpleNamespace(integers=fail))
    )
    with pytest.raises(CleanExecutionError) as error:
        SamNoise().watermark(42)
    assert error.value.code == ErrorCode.RESOURCE_EXHAUSTED


def test_pinned_rng_known_answer_and_global_state_isolation(guard):
    state = np.random.get_state()
    sampler = SamNoise()
    noise = sampler.create(
        guard.workspace / "vector", frames=257, seed=42, guard=guard, tile_frames=7
    )
    try:
        assert (
            sampler.policy.digest
            == "7a463a86fffaee8ecfa8fcdb03d19fe2d1caad01056a88784e861c779bd9ae7a"
        )
        assert (
            sampler.identity(seed=42, frames=257)
            == "304015eb64fc28c791c6408e401d0ad62b602567226f438f4898e24d306388a2"
        )
        assert (
            hashlib.sha256(noise.path.read_bytes()).hexdigest()
            == "9ee91b1449f1ece474e876b5e4e1ab42ef40fc620a678d9380909a01ec1a509f"
        )
        assert (
            hashlib.sha256(sampler.watermark(42).tobytes()).hexdigest()
            == "ce1c6a84d98f3563860c6bbe0216e06d81913a44bf6ee3a95ba524e5e1a56034"
        )
        current = np.random.get_state()
        assert current[0] == state[0] and current[2:] == state[2:]
        np.testing.assert_array_equal(current[1], state[1])
    finally:
        noise.close(remove=True)
