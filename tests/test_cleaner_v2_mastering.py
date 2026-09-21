import json
import threading
import time

import numpy as np
import pytest
import soundfile as sf

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.services.magic_clean.contracts import AttemptTicket, CleanExecutionError
from hear.services.magic_clean.mastering import AudioMasteringService, LoudnessMeasurement
from tests.test_cleaner_v2_contracts import ticket as ticket_fixture

ticket = ticket_fixture


@pytest.fixture
def mastering(tmp_path, ticket):
    guard = ResourceGuard(
        ResourceBudget(20_000_000, 5_000_000, 300000),
        tmp_path,
        time.monotonic() + 30,
        threading.Event(),
    )
    plan = AttemptTicket.model_validate_json(json.dumps(ticket)).plan
    return AudioMasteringService(CancellableProcessRunner()), plan, guard


@pytest.mark.parametrize("channels,amplitude", [(1, 0.02), (2, 0.7)])
@pytest.mark.parametrize("rate", [8000, 22050, 48000, 96000])
def test_real_master_and_delivery_validate(tmp_path, mastering, channels, amplitude, rate):
    service, plan, guard = mastering
    frames = 2 * rate + 1
    wave = amplitude * np.sin(2 * np.pi * 997 * np.arange(frames) / rate)
    samples = np.repeat(wave[:, None], channels, axis=1)
    source = tmp_path / "engine.wav"
    sf.write(source, samples, rate, subtype="FLOAT")
    result = service.master(source, plan, guard)
    with sf.SoundFile(result.master) as master:
        assert master.subtype == "PCM_24"
        assert master.frames == frames
        assert master.samplerate == rate
        assert master.channels == channels
    assert result.gain_db <= 6
    assert result.delivery_rate == 48000
    assert result.processing_rate == rate
    assert result.delivery_measurement.true_peak_dbtp <= -1
    assert result.master_measurement.true_peak_dbtp <= -1
    assert result.delivery_measurement.integrated_lufs is not None


@pytest.mark.parametrize("frames", [100, 48000])
def test_silence_has_null_loudness_and_reason(tmp_path, mastering, frames):
    service, plan, guard = mastering
    source = tmp_path / "engine.wav"
    sf.write(source, np.zeros(frames), 48000, subtype="FLOAT")
    result = service.master(source, plan, guard)
    assert result.master_measurement.integrated_lufs is None
    assert result.master_measurement.unavailable_reason is not None
    assert result.gain_db == 0


def test_loudness_off_preserves_safe_amplitude(tmp_path, mastering):
    service, plan, guard = mastering
    payload = plan.model_dump(mode="json")
    payload["adjust_loudness"] = False
    plan = type(plan).model_validate_json(json.dumps(payload))
    source = tmp_path / "engine.wav"
    data = 0.1 * np.sin(np.arange(48000) * 0.1)
    sf.write(source, data, 48000, subtype="FLOAT")
    result = service.master(source, plan, guard)
    assert result.gain_db == 0
    decoded, _ = sf.read(result.master)
    np.testing.assert_allclose(decoded, data, atol=2e-7)


def test_nonfinite_audio_rejected_before_encoding(tmp_path, mastering):
    service, plan, guard = mastering
    source = tmp_path / "engine.wav"
    sf.write(source, np.array([0.0, np.nan]), 48000, subtype="FLOAT")
    with pytest.raises(CleanExecutionError):
        service.master(source, plan, guard)
    assert not list(tmp_path.glob("*.flac"))


def test_encoded_input_cannot_be_mastered_again(tmp_path, mastering):
    service, plan, guard = mastering
    source = tmp_path / "old.flac"
    sf.write(source, np.zeros(48000), 48000, subtype="PCM_24")
    with pytest.raises(CleanExecutionError):
        service.master(source, plan, guard)


def test_codec_correction_renders_new_master_from_float_source(tmp_path, mastering):
    _, plan, guard = mastering

    class RecordingRunner(CancellableProcessRunner):
        def __init__(self):
            super().__init__()
            self.commands = []

        def run(self, argv, resource_guard):
            self.commands.append(argv)
            return super().run(argv, resource_guard)

    class OvershootOnce(AudioMasteringService):
        def measure(self, path, resource_guard, duration):
            measured = super().measure(path, resource_guard, duration)
            if path.name == "delivery-0.mp3":
                return LoudnessMeasurement(measured.integrated_lufs, None, -0.5)
            return measured

    source = tmp_path / "engine.wav"
    sf.write(source, 0.1 * np.sin(np.arange(48000) * 0.1), 48000, subtype="FLOAT")
    runner = RecordingRunner()
    result = OvershootOnce(runner).master(source, plan, guard)
    assert result.master.name == "master-1.flac"
    assert not (tmp_path / "master-0.flac").exists()
    assert not (tmp_path / "delivery-0.mp3").exists()
    encodes = [args for args in runner.commands if "-c:a" in args]
    for args in encodes:
        input_path = args[args.index("-i") + 1]
        codec = args[args.index("-c:a") + 1]
        if codec == "flac":
            assert input_path == str(source)
        else:
            assert input_path.endswith(".flac")
