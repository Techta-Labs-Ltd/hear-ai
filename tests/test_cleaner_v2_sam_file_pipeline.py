import hashlib
import threading
import time

import numpy as np
import pytest
import soundfile as sf
import torch

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.sam_features import SamFeatureFile
from hear.runtime.cleaner.sam_noise import SamNoise
from hear.runtime.cleaner.sam_pipeline import SamSeparationPipeline
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


@pytest.mark.parametrize("failure", [None, "separation", "cancel", "cleanup"])
def test_file_pipeline_seeded_inputs_and_lifecycle(tmp_path, monkeypatch, failure):
    guard = ResourceGuard(
        ResourceBudget(1000000, 1000000, 10000), tmp_path, time.monotonic() + 30, threading.Event()
    )
    source = tmp_path / "source.wav"
    values = np.linspace(-0.5, 0.5, 3841, dtype=np.float32)
    sf.write(source, values, 48000, subtype="FLOAT")
    original = source.read_bytes()
    output = tmp_path / "target.wav"
    pipeline = SamSeparationPipeline(None, None)
    noise_hashes = []
    text = torch.ones(1, 2, 768)
    mask = torch.ones(1, 2, dtype=torch.bool)

    def separate(prepared, noise, *, text, text_mask, message):
        assert prepared.complete and noise.complete
        assert (noise.frames, noise.batch, noise.channels) == (3, 1, 256)
        np.testing.assert_array_equal(prepared.read(0, 3841)[0, 0], values)
        np.testing.assert_array_equal(message.numpy(), SamNoise().watermark(42))
        noise_hashes.append(hashlib.sha256(noise.path.read_bytes()).hexdigest())
        if failure == "separation":
            raise RuntimeError("fixture separation failure")
        if failure == "cancel":
            guard.cancelled.set()
            guard.check()
        result = SamFeatureFile(
            tmp_path / "result", frames=3841, batch=2, channels=1, guard=guard, create=True
        )
        result.write(0, np.stack([values * 0.5, -values])[:, None])
        return result

    monkeypatch.setattr(pipeline, "separate", separate)
    close = SamFeatureFile.close

    def close_then_fail(feature, *, remove=False):
        close(feature, remove=remove)
        if failure == "cleanup" and feature.path.name == "result":
            raise RuntimeError("fixture close failure")

    monkeypatch.setattr(SamFeatureFile, "close", close_then_fail)
    if failure:
        with pytest.raises((RuntimeError, CleanExecutionError)):
            pipeline.separate_file(source, output, seed=42, text=text, text_mask=mask, guard=guard)
        assert set(tmp_path.iterdir()) == {source}
    else:
        for _ in range(2):
            identity = pipeline.separate_file(
                source, output, seed=42, text=text, text_mask=mask, guard=guard
            )
            assert identity == SamNoise().identity(seed=42, frames=3)
            data, rate = sf.read(output, dtype="float32")
            assert rate == 48000
            np.testing.assert_array_equal(data, values * 0.5)
            assert set(tmp_path.iterdir()) == {source, output}
            output.unlink()
        assert noise_hashes[0] == noise_hashes[1]
    assert source.read_bytes() == original


def test_existing_output_rejected_before_import_or_model_work(tmp_path, monkeypatch):
    guard = ResourceGuard(
        ResourceBudget(1000000, 1000000, 10000), tmp_path, time.monotonic() + 30, threading.Event()
    )
    output = tmp_path / "keep.wav"
    output.write_bytes(b"keep")
    pipeline = SamSeparationPipeline(None, None)
    with pytest.raises(CleanExecutionError) as error:
        pipeline.separate_file(
            tmp_path / "absent.wav", output, seed=42, text=None, text_mask=None, guard=guard
        )
    assert error.value.code == ErrorCode.ARTIFACT_CONFLICT
    assert output.read_bytes() == b"keep"
