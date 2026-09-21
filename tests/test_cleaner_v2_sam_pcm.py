import threading
import time

import numpy as np
import pytest
import soundfile as sf

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.sam_features import SamFeatureFile
from hear.runtime.cleaner.sam_pcm import SamPCM
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


@pytest.fixture
def guard(tmp_path):
    return ResourceGuard(
        ResourceBudget(1000000, 1000000, 10000), tmp_path, time.monotonic() + 30, threading.Event()
    )


@pytest.mark.parametrize("tile", [1, 7, 128])
def test_prepared_input_and_explicit_output_streams_are_exact(guard, tile):
    values = np.linspace(-0.8, 0.8, 257, dtype=np.float32)
    path = guard.workspace / "input.wav"
    sf.write(path, values, 48000, subtype="FLOAT")
    original = path.read_bytes()
    pcm = SamPCM(tile_frames=tile)
    source = pcm.read(path, guard.workspace / "input.f32", guard)
    paired = SamFeatureFile(
        guard.workspace / "paired.f32", frames=257, batch=2, channels=1, guard=guard, create=True
    )
    try:
        np.testing.assert_array_equal(source.read(0, 257)[0, 0], values)
        paired.write(0, np.stack([values, -values])[:, None])
        for name, expected in (("target", values), ("residual", -values)):
            output = guard.workspace / (name + ".wav")
            pcm.write(paired, output, stream=name)
            with sf.SoundFile(output) as audio:
                assert (audio.frames, audio.channels, audio.samplerate) == (257, 1, 48000)
                assert (audio.format, audio.subtype) == ("RF64", "FLOAT")
                np.testing.assert_array_equal(audio.read(dtype="float32"), expected)
        assert path.read_bytes() == original
        assert paired.complete and source.complete
    finally:
        source.close(remove=True)
        paired.close(remove=True)


@pytest.mark.parametrize("rate,channels", [(44100, 1), (48000, 2)])
def test_no_implicit_resampling_or_downmix(guard, rate, channels):
    path = guard.workspace / "input.wav"
    sf.write(path, np.ones((17, channels)), rate, subtype="FLOAT")
    output = guard.workspace / "output"
    with pytest.raises(CleanExecutionError) as error:
        SamPCM().read(path, output, guard)
    assert error.value.code == ErrorCode.INVALID_AUDIO
    assert not output.exists()


def test_nonfinite_input_removes_partial_import(guard):
    path = guard.workspace / "input.wav"
    sf.write(path, np.array([0.1, 0.2, np.nan], dtype=np.float32), 48000, subtype="FLOAT")
    output = guard.workspace / "output"
    with pytest.raises(CleanExecutionError) as error:
        SamPCM(tile_frames=2).read(path, output, guard)
    assert error.value.code == ErrorCode.INVALID_AUDIO
    assert not output.exists() and path.exists()


@pytest.mark.parametrize("operation", ["read", "write"])
def test_existing_output_is_preserved(guard, operation):
    path = guard.workspace / "input.wav"
    sf.write(path, np.zeros(17), 48000, subtype="FLOAT")
    output = guard.workspace / "keep"
    output.write_bytes(b"keep")
    paired = SamFeatureFile(
        guard.workspace / "paired", frames=17, batch=2, channels=1, guard=guard, create=True
    )
    paired.write(0, np.zeros((2, 1, 17), dtype=np.float32))
    try:
        with pytest.raises(CleanExecutionError) as error:
            if operation == "read":
                SamPCM().read(path, output, guard)
            else:
                SamPCM().write(paired, output, stream="target")
        assert error.value.code == ErrorCode.ARTIFACT_CONFLICT
        assert output.read_bytes() == b"keep"
    finally:
        paired.close(remove=True)


@pytest.mark.parametrize("operation", ["read", "write"])
def test_final_tile_cancellation_cleans_only_output(guard, monkeypatch, operation):
    path = guard.workspace / "input.wav"
    sf.write(path, np.zeros(17), 48000, subtype="FLOAT")
    paired = SamFeatureFile(
        guard.workspace / "paired", frames=17, batch=2, channels=1, guard=guard, create=True
    )
    paired.write(0, np.zeros((2, 1, 17), dtype=np.float32))
    original = SamFeatureFile.write if operation == "read" else SamFeatureFile.read

    def cancel(*args):
        result = original(*args)
        guard.cancelled.set()
        return result

    monkeypatch.setattr(SamFeatureFile, "write" if operation == "read" else "read", cancel)
    output = guard.workspace / "output"
    try:
        with pytest.raises(CleanExecutionError) as error:
            if operation == "read":
                SamPCM().read(path, output, guard)
            else:
                SamPCM().write(paired, output, stream="target")
        assert error.value.code == ErrorCode.CANCELLED
        assert not output.exists() and paired.path.exists() and path.exists()
    finally:
        paired.close(remove=True)


@pytest.mark.parametrize("tile", [True, 0, -1, 1.5, 65537])
def test_invalid_tile_size(tile):
    with pytest.raises(ValueError):
        SamPCM(tile_frames=tile)


def test_import_byte_limit_is_enforced_before_output_creation(guard):
    path = guard.workspace / "input.wav"
    sf.write(path, np.zeros(17), 48000, subtype="FLOAT")
    guard.budget = ResourceBudget(1000000, 1, 10000)
    output = guard.workspace / "output"
    with pytest.raises(CleanExecutionError) as error:
        SamPCM().read(path, output, guard)
    assert error.value.code == ErrorCode.RESOURCE_EXHAUSTED
    assert not output.exists()


def test_export_reservation_accounts_for_existing_workspace_files(guard):
    paired = SamFeatureFile(
        guard.workspace / "paired", frames=17, batch=2, channels=1, guard=guard, create=True
    )
    paired.write(0, np.zeros((2, 1, 17), dtype=np.float32))
    guard.budget = ResourceBudget(4200, 1000000, 10000)
    output = guard.workspace / "output.wav"
    try:
        with pytest.raises(CleanExecutionError) as error:
            SamPCM().write(paired, output, stream="target")
        assert error.value.code == ErrorCode.RESOURCE_EXHAUSTED
        assert not output.exists() and paired.complete
    finally:
        paired.close(remove=True)


@pytest.mark.parametrize(
    "batch,complete,stream", [(1, True, "target"), (2, False, "target"), (2, True, "both")]
)
def test_export_rejects_wrong_batch_incomplete_data_or_implicit_mixing(
    guard, batch, complete, stream
):
    source = SamFeatureFile(
        guard.workspace / "source", frames=17, batch=batch, channels=1, guard=guard, create=True
    )
    if complete:
        source.write(0, np.zeros((batch, 1, 17), dtype=np.float32))
    output = guard.workspace / "output.wav"
    try:
        with pytest.raises(ValueError):
            SamPCM().write(source, output, stream=stream)
        assert not output.exists()
    finally:
        source.close(remove=True)
