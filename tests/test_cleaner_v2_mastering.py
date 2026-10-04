import json
import threading
import time

import numpy as np
import pytest
import soundfile as sf

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.services.magic_clean.contracts import CleanExecutionError
from hear.services.magic_clean.mastering import AudioMasteringService, LoudnessMeasurement
from tests.cleaner_plans import natural_plan


@pytest.fixture
def mastering(tmp_path):
    guard = ResourceGuard(
        ResourceBudget(20_000_000, 5_000_000, 300000),
        tmp_path,
        time.monotonic() + 30,
        threading.Event(),
    )
    plan = natural_plan()
    return AudioMasteringService(CancellableProcessRunner()), plan, guard


@pytest.mark.parametrize("channels,amplitude", [(1, 0.02), (2, 0.7)])
@pytest.mark.parametrize("rate", [8000, 22050, 48000, 96000])
def test_real_delivery_validates(tmp_path, mastering, channels, amplitude, rate):
    service, plan, guard = mastering
    frames = 2 * rate + 1
    wave = amplitude * np.sin(2 * np.pi * 997 * np.arange(frames) / rate)
    samples = np.repeat(wave[:, None], channels, axis=1)
    source = tmp_path / "engine.wav"
    sf.write(source, samples, rate, subtype="FLOAT")
    result = service.master(source, plan, guard)
    with sf.SoundFile(result.delivery) as delivery:
        assert delivery.samplerate == 48000
        assert delivery.channels == channels
        assert abs(delivery.frames - round(frames * 48000 / rate)) <= 1440
    assert result.frames == frames
    assert result.gain_db <= 6
    assert result.delivery_rate == 48000
    assert result.processing_rate == rate
    assert result.delivery_measurement.true_peak_dbtp <= -1
    assert result.delivery_measurement.integrated_lufs is not None
    assert not list(tmp_path.glob("*.flac"))


@pytest.mark.parametrize("frames", [100, 48000])
def test_silence_has_null_loudness_and_reason(tmp_path, mastering, frames):
    service, plan, guard = mastering
    source = tmp_path / "engine.wav"
    sf.write(source, np.zeros(frames), 48000, subtype="FLOAT")
    result = service.master(source, plan, guard)
    assert result.delivery_measurement.integrated_lufs is None
    assert result.delivery_measurement.unavailable_reason is not None
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
    # MP3 is not sample-exact; the delivery peak must sit at the source peak (-20 dBFS).
    assert abs(result.delivery_measurement.true_peak_dbtp - (-20.0)) < 1.0


def test_nonfinite_audio_rejected_before_encoding(tmp_path, mastering):
    service, plan, guard = mastering
    source = tmp_path / "engine.wav"
    sf.write(source, np.array([0.0, np.nan]), 48000, subtype="FLOAT")
    with pytest.raises(CleanExecutionError):
        service.master(source, plan, guard)
    assert not list(tmp_path.glob("*.mp3"))


def test_encoded_input_cannot_be_mastered_again(tmp_path, mastering):
    service, plan, guard = mastering
    source = tmp_path / "old.flac"
    sf.write(source, np.zeros(48000), 48000, subtype="PCM_24")
    with pytest.raises(CleanExecutionError):
        service.master(source, plan, guard)


def test_codec_correction_rerenders_delivery_from_float_source(tmp_path, mastering):
    _, plan, guard = mastering

    class OvershootOnce(AudioMasteringService):
        # Summaries: 1 = float master, 2 = delivery-0 (overshoots), 3 = delivery-1.
        summaries = 0

        def summarize(self, measurements, rate):
            measured = super().summarize(measurements, rate)
            self.summaries += 1
            if self.summaries == 2:
                return LoudnessMeasurement(measured.integrated_lufs, None, -0.5)
            return measured

    source = tmp_path / "engine.wav"
    sf.write(source, 0.1 * np.sin(np.arange(48000) * 0.1), 48000, subtype="FLOAT")
    service = OvershootOnce(CancellableProcessRunner())
    result = service.master(source, plan, guard)
    assert result.delivery.name == "delivery-1.mp3"
    assert not (tmp_path / "delivery-0.mp3").exists()
    # The second render is cut again from the float master, never from MP3.
    assert not list(tmp_path.glob("mp3-*/*.mp3"))
    assert service.summaries == 3


