import asyncio
import io
import os
import tempfile
from dataclasses import dataclass

import edge_tts
import torch
import torchaudio
import torchaudio.functional as F_audio

from app.core.storage import storage

VOICE_MAP = {
    "male_us": "en-US-GuyNeural",
    "female_us": "en-US-JennyNeural",
    "male_uk": "en-GB-RyanNeural",
    "female_uk": "en-GB-SoniaNeural",
    "male_au": "en-AU-WilliamNeural",
    "female_au": "en-AU-NatashaNeural",
}

DEFAULT_VOICE = "en-GB-RyanNeural"

@dataclass
class SynthesisResult:
    b2_key: str
    audio_url: str
    duration: float


class SpeechSynthesizer:
    TARGET_SR = 44100

    def __init__(self):
        self._loaded = False

    def load(self):
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _detect_voice(self, waveform: torch.Tensor, sr: int, start: float, end: float) -> str:
        start_sample = int(start * sr)
        end_sample = min(int(end * sr), waveform.shape[1])
        segment = waveform[:, start_sample:end_sample]

        if segment.shape[1] < sr * 0.1:
            return "male_us"

        mono = segment.mean(dim=0) if segment.shape[0] > 1 else segment[0]

        try:
            pitch = F_audio.detect_pitch_frequency(
                mono.unsqueeze(0), sr, freq_low=50, freq_high=600
            )
            voiced = pitch[pitch > 50]
            if voiced.numel() < 5:
                return "male_us"

            median_f0 = voiced.median().item()
            gender = "female" if median_f0 >= 165 else "male"

            accent = self._detect_accent(voiced)

            return f"{gender}_{accent}"
        except Exception:
            return "male_us"

    def _detect_accent(self, voiced_pitch: torch.Tensor) -> str:
        f0_mean = voiced_pitch.mean().item()
        f0_std = voiced_pitch.std().item()
        f0_range = voiced_pitch.max().item() - voiced_pitch.min().item()

        pitch_variability = f0_std / (f0_mean + 1e-8)

        n = voiced_pitch.numel()
        if n > 10:
            tail = voiced_pitch[int(n * 0.7):]
            head = voiced_pitch[:int(n * 0.3)]
            tail_mean = tail.mean().item()
            head_mean = head.mean().item()
            rising_ratio = tail_mean / (head_mean + 1e-8)
        else:
            rising_ratio = 1.0

        if rising_ratio > 1.08:
            return "au"

        if pitch_variability > 0.25 or f0_range > 120:
            return "uk"

        return "us"

    async def reconstruct_segment(
        self,
        original_audio_path: str,
        segment_start: float,
        segment_end: float,
        new_text: str,
        recording_id: str,
    ) -> SynthesisResult:
        original_waveform, orig_sr = torchaudio.load(original_audio_path)

        detected = self._detect_voice(original_waveform, orig_sr, segment_start, segment_end)
        voice_id = VOICE_MAP.get(detected, DEFAULT_VOICE)

        if orig_sr != self.TARGET_SR:
            original_waveform = F_audio.resample(original_waveform, orig_sr, self.TARGET_SR)

        start_sample = int(segment_start * self.TARGET_SR)
        end_sample = int(segment_end * self.TARGET_SR)

        tts_bytes = await self._synthesize(new_text, voice_id)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(tts_bytes)
            tts_path = tmp.name

        tts_waveform, tts_sr = torchaudio.load(tts_path)
        os.unlink(tts_path)

        if tts_sr != self.TARGET_SR:
            tts_waveform = F_audio.resample(tts_waveform, tts_sr, self.TARGET_SR)

        if tts_waveform.shape[0] != original_waveform.shape[0]:
            tts_waveform = tts_waveform.mean(dim=0, keepdim=True).expand(
                original_waveform.shape[0], -1
            )

        target_length = end_sample - start_sample
        if tts_waveform.shape[1] < target_length:
            pad = torch.zeros(tts_waveform.shape[0], target_length - tts_waveform.shape[1])
            tts_waveform = torch.cat([tts_waveform, pad], dim=1)
        elif tts_waveform.shape[1] > target_length:
            speed_factor = tts_waveform.shape[1] / target_length
            tts_waveform = F_audio.resample(
                tts_waveform,
                int(self.TARGET_SR * speed_factor),
                self.TARGET_SR,
            )[:, :target_length]

        cross_len = min(int(0.02 * self.TARGET_SR), start_sample, target_length)

        before = original_waveform[:, :start_sample]
        after = original_waveform[:, end_sample:]

        if cross_len > 0:
            fade_out = torch.linspace(1.0, 0.0, cross_len).unsqueeze(0)
            fade_in = torch.linspace(0.0, 1.0, cross_len).unsqueeze(0)
            before[:, -cross_len:] = (
                before[:, -cross_len:] * fade_out + tts_waveform[:, :cross_len] * fade_in
            )
            tts_waveform = tts_waveform[:, cross_len:]

            if after.shape[1] >= cross_len:
                end_fade_out = torch.linspace(1.0, 0.0, cross_len).unsqueeze(0)
                end_fade_in = torch.linspace(0.0, 1.0, cross_len).unsqueeze(0)
                tts_tail = tts_waveform[:, -cross_len:] * end_fade_out
                after_head = after[:, :cross_len] * end_fade_in
                after = torch.cat([tts_tail + after_head, after[:, cross_len:]], dim=1)
                tts_waveform = tts_waveform[:, :-cross_len]

        reconstructed = torch.cat([before, tts_waveform, after], dim=1)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
        torchaudio.save(out_path, reconstructed, self.TARGET_SR)

        duration = reconstructed.shape[1] / self.TARGET_SR
        b2_key = f"reconstructed/{recording_id}/{os.urandom(8).hex()}.wav"
        loop = asyncio.get_event_loop()
        audio_url = await loop.run_in_executor(None, storage.upload_file, out_path, b2_key)
        os.unlink(out_path)

        return SynthesisResult(
            b2_key=b2_key,
            audio_url=audio_url,
            duration=round(duration, 3),
        )

    async def _synthesize(self, text: str, voice_id: str) -> bytes:
        communicate = edge_tts.Communicate(text, voice_id)
        buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer.write(chunk["data"])
        return buffer.getvalue()
