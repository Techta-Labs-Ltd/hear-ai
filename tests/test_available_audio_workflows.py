import asyncio
import math
import wave
from pathlib import Path

import numpy as np
import pytest

from hear.audio.workspace import AudioWorkspace
from hear.execution.native import NativeExecutor
from hear.workflows.available_reconstruction import AvailableReconstructionWorkflow


def write_wave(path: Path, seconds: float = 2.0) -> None:
    sample_rate = 16000
    samples = np.array(
        [int(math.sin(index * 2 * math.pi * 220 / sample_rate) * 12000) for index in range(int(sample_rate * seconds))],
        dtype=np.int16,
    )
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())


def test_available_reconstruction_removes_interval_with_disk_renderer(tmp_path):
    source = tmp_path / "source.wav"
    output = tmp_path / "output.mp3"
    write_wave(source)
    native = NativeExecutor("available-reconstruction-test")
    workflow = AvailableReconstructionWorkflow(
        object(),
        object(),
        native,
        workspace_root=tmp_path,
        timeout_seconds=30,
    )
    workspace = AudioWorkspace(tmp_path, "job", "attempt")

    async def run():
        result = await workflow._splice(
            source,
            output,
            [{"segment_start": 0.5, "segment_end": 1.0, "is_deletion": True}],
            workspace,
        )
        duration = await native.run(workflow._duration, output)
        await native.close()
        return result, duration

    result, duration = asyncio.run(run())

    assert result == [{"segment_start": 0.5, "segment_end": 1.0, "is_deletion": True}]
    assert duration == pytest.approx(1.5, abs=0.1)
