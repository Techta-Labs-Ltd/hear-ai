import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import hear.runtime.cleaner.sam_pipeline as pipeline_module
from hear.runtime.cleaner.longform_sam import SolverPolicy
from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.sam_features import SamFeatureFile
from hear.runtime.cleaner.sam_pipeline import SamSeparationPipeline
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


class FieldFixture:
    def __init__(self, forward, mean, text, mask):
        self.text, self.mask = text, mask
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def pipeline_inputs(tmp_path, monkeypatch):
    guard = ResourceGuard(
        ResourceBudget(1000000, 1000000, 8192), tmp_path, time.monotonic() + 30, threading.Event()
    )
    source = SamFeatureFile(
        tmp_path / "source", frames=3841, batch=1, channels=1, guard=guard, create=True
    )
    source.write(0, np.ones((1, 1, 3841), np.float32))
    noise = SamFeatureFile(
        tmp_path / "noise", frames=3, batch=1, channels=256, guard=guard, create=True
    )
    noise.write(0, np.zeros((1, 256, 3), np.float32))
    monkeypatch.setattr(pipeline_module, "SamConditionedField", FieldFixture)
    yield source, noise
    source.close(remove=True)
    noise.close(remove=True)


@pytest.mark.parametrize("failure", [None, "encoding", "solver", "decoder", "after_decode"])
def test_pipeline_lifecycle_preserves_borrowed_inputs(pipeline_inputs, monkeypatch, failure):
    source, noise = pipeline_inputs
    guard = source.guard
    field_seen = []
    text = torch.ones(1, 2, 768)
    mask = torch.ones(1, 2, dtype=torch.bool)
    message = torch.ones(2, 16)

    def encode(codec, borrowed):
        assert borrowed is source
        if failure == "encoding":
            raise RuntimeError("fixture encoding failure")
        text.zero_()
        mask.zero_()
        message.zero_()
        mean = SamFeatureFile(
            guard.workspace / "mean", frames=3, batch=1, channels=128, guard=guard, create=True
        )
        mean.write(0, np.ones((1, 128, 3), np.float32))
        return mean

    def solve(borrowed, destination, *, frames, channels, field, guard):
        assert borrowed == noise.path
        field_seen.append(field)
        assert field.text.all() and field.mask.all()
        # Deliberately leave a partial solver file when the fixture fails.
        destination.write_bytes(np.zeros((frames, channels), np.float32).tobytes())
        if failure == "solver":
            raise RuntimeError("fixture solver failure")

    def decode(codec, joint, *, frames, message):
        assert message.all() and joint.complete
        if failure == "decoder":
            raise RuntimeError("fixture decoder failure")
        output = SamFeatureFile(
            guard.workspace / "result", frames=frames, batch=2, channels=1, guard=guard, create=True
        )
        output.write(0, np.ones((2, 1, frames), np.float32))
        if failure == "after_decode":
            guard.cancelled.set()
        return output

    pipeline = SamSeparationPipeline(None, None)
    pipeline.graph = SimpleNamespace(encode_mean=encode, decode_joint=decode)
    monkeypatch.setattr(pipeline.solver, "solve", solve)
    output = None
    try:
        if failure:
            with pytest.raises((RuntimeError, CleanExecutionError)) as error:
                pipeline.separate(source, noise, text=text, text_mask=mask, message=message)
            if failure == "after_decode":
                assert error.value.code == ErrorCode.CANCELLED
            assert set(guard.workspace.iterdir()) == {source.path, noise.path}
        else:
            output = pipeline.separate(source, noise, text=text, text_mask=mask, message=message)
            assert (output.frames, output.batch, output.channels) == (3841, 2, 1)
            assert set(guard.workspace.iterdir()) == {source.path, noise.path, output.path}
        assert all(field.closed for field in field_seen)
        assert np.fromfile(source.path, dtype=np.float32).sum() == 3841
        assert not np.fromfile(noise.path, dtype=np.float32).any()
    finally:
        if output is not None:
            output.close(remove=True)


def test_nonproduction_solver_steps_rejected():
    with pytest.raises(ValueError):
        SamSeparationPipeline(None, None, policy=SolverPolicy(5, 2, 2))


def test_incomplete_noise_rejected_before_encoding(pipeline_inputs):
    source, noise = pipeline_inputs
    incomplete = SamFeatureFile(
        source.path.parent / "incomplete",
        frames=3,
        batch=1,
        channels=256,
        guard=source.guard,
        create=True,
    )
    try:
        with pytest.raises(ValueError):
            SamSeparationPipeline(None, None).separate(
                source,
                incomplete,
                text=torch.ones(1, 2, 768),
                text_mask=torch.ones(1, 2, dtype=torch.bool),
                message=torch.ones(2, 16),
            )
        assert not incomplete.complete
    finally:
        incomplete.close(remove=True)
