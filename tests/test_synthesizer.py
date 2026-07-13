import pytest
import torch

from app.services.synthesizer import SpeechSynthesizer


class TestAnalyzeProsody:
    def test_returns_empty_for_short_audio(self):
        wf = torch.zeros(1, 100)
        result = SpeechSynthesizer._analyze_prosody(wf, 44100, 0, 100)
        assert result == {}

    def test_returns_params_for_speech(self):
        sr = 44100
        dur_s = 1.0
        t = torch.linspace(0, dur_s, int(sr * dur_s))
        wf = torch.sin(2 * torch.pi * 200 * t).unsqueeze(0) * 0.5
        wf += torch.sin(2 * torch.pi * 400 * t).unsqueeze(0) * 0.3
        result = SpeechSynthesizer._analyze_prosody(wf, sr, 0, wf.shape[1])
        assert isinstance(result, dict)
        for key in result:
            assert 0.1 <= result[key] <= (1.0 if key != "chunk_length" else 1000)

    def test_high_energy_segment(self):
        sr = 44100
        dur_s = 2.0
        t = torch.linspace(0, dur_s, int(sr * dur_s))
        wf = torch.sin(2 * torch.pi * 300 * t).unsqueeze(0) * 0.9
        wf += torch.sin(2 * torch.pi * 500 * t).unsqueeze(0) * 0.4
        result = SpeechSynthesizer._analyze_prosody(wf, sr, 0, wf.shape[1])
        assert isinstance(result, dict)

    def test_near_silent_segment(self):
        sr = 44100
        dur_s = 1.0
        t = torch.linspace(0, dur_s, int(sr * dur_s))
        wf = torch.sin(2 * torch.pi * 300 * t).unsqueeze(0) * 1e-9
        result = SpeechSynthesizer._analyze_prosody(wf, sr, 0, wf.shape[1])
        assert result == {}

    def test_all_values_in_range(self):
        sr = 44100
        dur_s = 3.0
        t = torch.linspace(0, dur_s, int(sr * dur_s))
        wf = torch.zeros(1, int(sr * dur_s))
        for i in range(5):
            freq = 200 + i * 150
            wf += torch.sin(2 * torch.pi * freq * t).unsqueeze(0) * (0.3 + i * 0.1)
        result = SpeechSynthesizer._analyze_prosody(wf, sr, 0, wf.shape[1])
        for key, val in result.items():
            if key == "chunk_length":
                assert 100 <= val <= 1000, f"{key}={val} out of range"
            else:
                assert 0.1 <= val <= 1.0, f"{key}={val} out of range"
