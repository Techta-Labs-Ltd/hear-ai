import json
import math
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hear.runtime.cleaner.resource_guard import ResourceBudget, ResourceGuard
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.services.magic_clean.mastering import AudioMasteringService
from hear.services.transcription.service import TranscriptionService


@pytest.mark.parametrize("rate,channels", [(44100, 1), (48000, 1), (48000, 2), (96000, 2)])
def test_measurement_only_meter_matches_previous_reference(tmp_path, rate, channels):
    path = tmp_path / "source.wav"
    t = np.arange(rate * 3) / rate
    wave = 0.6 * np.sin(2 * np.pi * 997 * t) + 0.1 * np.sin(2 * np.pi * 0.37 * rate * t)
    signal = np.column_stack([wave * (1 - channel * 0.2) for channel in range(channels)])
    sf.write(path, signal, rate, subtype="FLOAT")
    guard = ResourceGuard(
        ResourceBudget(64 * 1024**2, 10 * 1024**2, 96000 * 10),
        tmp_path,
        time.monotonic() + 30,
        threading.Event(),
    )
    runner = CancellableProcessRunner()
    meter = AudioMasteringService(runner)
    old = runner.run(
        meter._input_args(path)
        + ["-af", "loudnorm=I=-19:TP=-1:LRA=11:print_format=json", "-f", "null", "-"],
        guard,
    ).decode()
    expected = json.loads(old[old.rfind("{") : old.rfind("}") + 1])
    result = meter.measure(path, guard, 3)
    assert abs(result.integrated_lufs - float(expected["input_i"])) <= 0.2
    assert abs(result.true_peak_dbtp - float(expected["input_tp"])) <= 0.25
    assert math.isfinite(result.true_peak_dbtp)


def test_meter_does_not_apply_normalization_to_audio(tmp_path):
    commands = []

    class Runner:
        def run(self, command, guard):
            commands.append(command)
            return (
                b"Summary:\nIntegrated loudness:\n I: -19.0 LUFS\n True peak:\n Peak: -1.3 dBFS\n"
            )

    value = AudioMasteringService(Runner()).measure(Path("test.wav"), None, 60)
    assert "ebur128=peak=true:framelog=verbose" in commands[0]
    assert not any("loudnorm" in part for part in commands[0])
    assert value.true_peak_dbtp == -1.25


def test_transcription_timings_are_separate_from_words(tmp_path):
    import asyncio

    path = tmp_path / "source.wav"
    sf.write(path, np.zeros(16000), 16000)

    class Model:
        async def transcribe_window(self, samples, batch_size, language):
            return {
                "segments": [{"text": "Hello", "start": 0, "end": 1}],
                "_runtime_timing": {"model_window_seconds": 0.1, "executor_wait_seconds": 0.2},
            }

    result = asyncio.run(
        TranscriptionService(Model(), chunk_seconds=60, batch_size=8).transcribe_file(str(path))
    )
    assert result["transcript"] == "Hello"
    assert result["performance"]["executor_wait_seconds"] == 0.2
    assert result["performance"]["model_window_seconds"] == 0.1
    assert result["performance"]["batch_size"] == 8
    assert "_runtime_timing" not in result
