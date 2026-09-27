from types import SimpleNamespace

import numpy as np
import pytest

import hear.inference.qwen_asr as qwen_asr


@pytest.fixture
def silero_api(monkeypatch):
    calls = {}
    model = object()

    def load_silero_vad(*, onnx):
        calls["onnx"] = onnx
        return model

    def get_speech_timestamps(waveform, actual_model, **kwargs):
        calls["waveform"] = waveform
        calls["model"] = actual_model
        calls["kwargs"] = kwargs
        return [{"start": 1600, "end": 8000}]

    monkeypatch.setattr(
        qwen_asr.LocalSileroVad,
        "_silero_vad_api",
        staticmethod(lambda: SimpleNamespace(
            load_silero_vad=load_silero_vad,
            get_speech_timestamps=get_speech_timestamps,
        )),
    )
    return calls, model


def test_local_silero_vad_uses_cpu_model_and_converts_sample_offsets(silero_api):
    calls, model = silero_api
    vad = qwen_asr.LocalSileroVad(0.65, 30)
    result = vad(
        {
            "waveform": np.zeros(16000, dtype=np.float32),
            "sample_rate": 16000,
        }
    )

    assert calls["onnx"] is False
    assert calls["model"] is model
    assert calls["kwargs"] == {
        "threshold": 0.65,
        "sampling_rate": 16000,
        "max_speech_duration_s": 30.0,
    }
    assert [(segment.start, segment.end) for segment in result] == [(0.1, 0.5)]


def test_local_silero_vad_rejects_wrong_rate_before_inference(silero_api):
    calls, _ = silero_api
    vad = qwen_asr.LocalSileroVad(0.65, 30)

    with pytest.raises(ValueError, match="silero_vad_requires_16000hz"):
        vad(
            {
                "waveform": np.zeros(8000, dtype=np.float32),
                "sample_rate": 8000,
            }
        )

    assert "waveform" not in calls


@pytest.mark.parametrize("vad_onset,chunk_size", [(0, 30), (1, 30), (0.65, 0)])
def test_local_silero_vad_rejects_invalid_configuration(vad_onset, chunk_size, silero_api):
    with pytest.raises(ValueError):
        qwen_asr.LocalSileroVad(vad_onset, chunk_size)
