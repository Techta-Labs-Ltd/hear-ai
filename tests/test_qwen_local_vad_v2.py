import numpy as np

from hear.inference.qwen_asr import LocalSileroVad


def test_local_silero_vad_uses_packaged_model():
    vad = LocalSileroVad(0.65, 30)
    result = vad(
        {
            "waveform": np.zeros(16000, dtype=np.float32),
            "sample_rate": 16000,
        }
    )
    assert result == []


def test_local_silero_vad_rejects_wrong_rate():
    vad = LocalSileroVad(0.65, 30)
    try:
        vad(
            {
                "waveform": np.zeros(8000, dtype=np.float32),
                "sample_rate": 8000,
            }
        )
    except ValueError as exc:
        assert str(exc) == "silero_vad_requires_16000hz"
        return
    raise AssertionError("wrong sample rate accepted")
