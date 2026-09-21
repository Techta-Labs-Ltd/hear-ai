import threading
import time

import numpy as np
import pytest

from hear.runtime.cleaner.longform_sam import SolverPolicy, WindowedMidpointSolver
from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


@pytest.fixture
def inputs(tmp_path):
    noise, output = tmp_path / "noise", tmp_path / "output"
    initial = np.random.default_rng(5).normal(size=(101, 4)).astype(np.float32)
    noise.write_bytes(initial.tobytes())
    guard = ResourceGuard(
        ResourceBudget(100000, 100000, 1000),
        tmp_path,
        time.monotonic() + 20,
        threading.Event(),
    )
    return noise, output, initial, guard


@pytest.mark.parametrize("window,overlap", [(16, 4), (32, 16), (128, 16)])
def test_midpoint_uses_global_states_and_exact_partial_tails(inputs, window, overlap):
    noise, output, initial, guard = inputs
    observations = {}

    class LinearField:
        def evaluate(self, state, *, start_frame, time):
            assert len(state) <= window
            observations.setdefault(time, []).append((start_frame, state.copy()))
            return state

    solver = WindowedMidpointSolver(SolverPolicy(window, overlap, 4))
    solver.solve(noise, output, frames=101, channels=4, field=LinearField(), guard=guard)
    result = np.fromfile(output, dtype=np.float32).reshape(initial.shape)
    np.testing.assert_allclose(
        result, initial * (1 + 0.25 + 0.25**2 / 2) ** 4, rtol=1e-6, atol=1e-6
    )
    for instant, windows in observations.items():
        completed_steps = int(instant * 4)
        expected = initial * (1 + 0.25 + 0.25**2 / 2) ** completed_steps
        if instant * 4 != completed_steps:
            expected *= 1.125
        for start, values in windows:
            np.testing.assert_allclose(
                values, expected[start : start + len(values)], rtol=1e-6, atol=1e-6
            )
    assert not list(guard.workspace.glob("sam-solver-*"))
    np.testing.assert_array_equal(
        np.fromfile(noise, dtype=np.float32).reshape(initial.shape), initial
    )


@pytest.mark.parametrize("failure", ["cancel", "nonfinite", "shape"])
def test_failure_cleans_partial_latents(inputs, failure):
    noise, output, _, guard = inputs

    class BadField:
        def evaluate(self, state, *, start_frame, time):
            if failure == "cancel":
                guard.cancelled.set()
            if failure == "nonfinite":
                state[:] = np.nan
            return state[:-1] if failure == "shape" else state

    with pytest.raises(CleanExecutionError) as error:
        WindowedMidpointSolver(SolverPolicy(16, 4)).solve(
            noise,
            output,
            frames=101,
            channels=4,
            field=BadField(),
            guard=guard,
        )
    assert error.value.code == (
        ErrorCode.CANCELLED if failure == "cancel" else ErrorCode.INVALID_AUDIO
    )
    assert not output.exists()
    assert noise.exists()
    assert not list(guard.workspace.glob("sam-solver-*"))


def test_budget_rejection_precedes_destination_creation(inputs):
    noise, output, _, guard = inputs
    guard.budget = ResourceBudget(2000, 2000, 1000)
    with pytest.raises(CleanExecutionError) as error:
        WindowedMidpointSolver(SolverPolicy(16, 4)).solve(
            noise,
            output,
            frames=101,
            channels=4,
            field=None,
            guard=guard,
        )
    assert error.value.code == ErrorCode.RESOURCE_EXHAUSTED
    assert not output.exists()


def test_dirty_mapping_flushes_before_eviction_and_aliases_only_once(inputs):
    _, _, _, guard = inputs
    calls = []

    class MapHandle:
        def madvise(self, option):
            calls.append("evict")

    class Mapping:
        mode = "r+"
        _mmap = MapHandle()

        def flush(self):
            calls.append("flush")

    mapping = Mapping()
    WindowedMidpointSolver._evict(guard, mapping, mapping)
    assert calls == ["flush", "evict"]


def test_eviction_failure_cleans_partial_output(inputs, monkeypatch):
    noise, output, _, guard = inputs

    def broken(*args):
        raise OSError("injected page eviction failure")

    monkeypatch.setattr(WindowedMidpointSolver, "_evict", broken)
    with pytest.raises(OSError):
        WindowedMidpointSolver(SolverPolicy(16, 4)).solve(
            noise,
            output,
            frames=101,
            channels=4,
            field=None,
            guard=guard,
        )
    assert not output.exists()
    assert not list(guard.workspace.glob("sam-solver-*"))
