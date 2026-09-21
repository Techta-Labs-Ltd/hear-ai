import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from hear.runtime.cleaner.resampling import AudioResampler
from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.services.magic_clean.contracts import CleanExecutionError, ErrorCode


@pytest.fixture
def resampling(tmp_path):
    guard = ResourceGuard(
        ResourceBudget(20000000, 5000000, 600000),
        tmp_path,
        time.monotonic() + 30,
        threading.Event(),
    )
    return AudioResampler(CancellableProcessRunner()), guard


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 44100, 88200, 96000])
@pytest.mark.parametrize("frames", [1, 17, 10003])
def test_real_resampler_roundtrip_keeps_frame_grid_and_channels(tmp_path, resampling, rate, frames):
    resampler, guard = resampling
    source, prepared, output = (tmp_path / name for name in ("in.wav", "48k.wav", "out.wav"))
    wave = np.zeros((frames, 2), dtype=np.float32)
    wave[frames // 2, 0] = 0.5
    sf.write(source, wave, rate, subtype="FLOAT")
    resampler.convert(source, prepared, 48000, guard)
    assert sf.info(prepared).frames == resampler.frame_count(frames, rate, 48000)
    resampler.convert(prepared, output, rate, guard, exact_frames=frames)
    result, actual_rate = sf.read(output, dtype="float32", always_2d=True)
    assert actual_rate == rate
    assert result.shape == wave.shape
    assert np.isfinite(result).all()
    assert np.max(np.abs(result[:, 1])) == 0
    assert abs(int(np.argmax(np.abs(result[:, 0]))) - frames // 2) <= 1


def test_rejects_invalid_trim_and_preserves_existing_output(tmp_path, resampling):
    resampler, guard = resampling
    source, output = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(source, np.zeros(1000), 44100, subtype="FLOAT")
    with pytest.raises(CleanExecutionError):
        resampler.convert(source, output, 48000, guard, exact_frames=1)
    assert not output.exists()
    output.write_bytes(b"existing")
    with pytest.raises(CleanExecutionError) as error:
        resampler.convert(source, output, 48000, guard)
    assert error.value.code == ErrorCode.ARTIFACT_CONFLICT
    assert output.read_bytes() == b"existing"


def test_nonfinite_input_rejected(tmp_path, resampling):
    resampler, guard = resampling
    source, output = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(source, np.array([0, np.nan]), 44100, subtype="FLOAT")
    with pytest.raises(CleanExecutionError) as error:
        resampler.convert(source, output, 48000, guard)
    assert error.value.code == ErrorCode.INVALID_AUDIO
    assert not output.exists()


def test_cancelled_resample_writes_nothing(tmp_path, resampling):
    resampler, guard = resampling
    source, output = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(source, np.zeros(1000), 44100, subtype="FLOAT")
    guard.cancelled.set()
    with pytest.raises(CleanExecutionError) as error:
        resampler.convert(source, output, 48000, guard)
    assert error.value.code == ErrorCode.CANCELLED
    assert not output.exists()


@pytest.mark.parametrize("native_failure", [False, True])
def test_racing_destination_is_never_removed(tmp_path, resampling, native_failure):
    resampler, guard = resampling
    source, output = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(source, np.zeros(1000), 44100, subtype="FLOAT")
    original = resampler.runner

    def race(command, active_guard):
        output.write_bytes(b"concurrent owner")
        if native_failure:
            raise CleanExecutionError(ErrorCode.PROCESS_FAILED, "fixture native failure")
        return original.run(command, active_guard)

    resampler.runner = SimpleNamespace(run=race)
    with pytest.raises(CleanExecutionError) as error:
        resampler.convert(source, output, 48000, guard)
    expected = ErrorCode.PROCESS_FAILED if native_failure else ErrorCode.ARTIFACT_CONFLICT
    assert error.value.code == expected
    assert output.read_bytes() == b"concurrent owner"
    assert set(tmp_path.iterdir()) == {source, output}


def test_cancel_after_native_output_cleans_private_staging(tmp_path, resampling):
    resampler, guard = resampling
    source, output = tmp_path / "in.wav", tmp_path / "out.wav"
    sf.write(source, np.zeros(1000), 44100, subtype="FLOAT")
    original = resampler.runner

    def cancel(command, active_guard):
        original.run(command, active_guard)
        active_guard.cancelled.set()

    resampler.runner = SimpleNamespace(run=cancel)
    with pytest.raises(CleanExecutionError) as error:
        resampler.convert(source, output, 48000, guard)
    assert error.value.code == ErrorCode.CANCELLED
    assert set(tmp_path.iterdir()) == {source}
