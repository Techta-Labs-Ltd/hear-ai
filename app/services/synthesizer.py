import asyncio
import io
import os
import tempfile
from dataclasses import dataclass
import importlib.util

import edge_tts
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F_audio

from app.config import settings
from app.core.audio_utils import save_as_mp3
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
        track_id: str,
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

        reconstructed = self._splice_segment(original_waveform, tts_waveform, start_sample, end_sample)

        out_path = save_as_mp3(reconstructed, self.TARGET_SR)

        duration = reconstructed.shape[1] / self.TARGET_SR
        b2_key = f"reconstructed/{track_id}/{os.urandom(8).hex()}.mp3"
        loop = asyncio.get_event_loop()
        audio_url = await loop.run_in_executor(None, storage.upload_file, out_path, b2_key)
        os.unlink(out_path)

        return SynthesisResult(
            b2_key=b2_key,
            audio_url=audio_url,
            duration=round(duration, 3),
        )

    async def rebuild_track_audio(
        self,
        original_audio_path: str,
        edited_transcript: str,
        track_id: str,
        job_id: str,
        original_transcript: str = "",
    ) -> SynthesisResult:
        original_waveform, orig_sr = torchaudio.load(original_audio_path)
        if orig_sr != self.TARGET_SR:
            original_waveform = F_audio.resample(original_waveform, orig_sr, self.TARGET_SR)

        rebuilt_bytes = await self._synthesize_higgs(edited_transcript)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(rebuilt_bytes)
            rebuilt_path = tmp.name

        rebuilt_waveform, rebuilt_sr = torchaudio.load(rebuilt_path)
        os.unlink(rebuilt_path)
        if rebuilt_sr != self.TARGET_SR:
            rebuilt_waveform = F_audio.resample(rebuilt_waveform, rebuilt_sr, self.TARGET_SR)

        start_sample, end_sample = self._detect_speech_bounds(original_waveform, self.TARGET_SR, original_transcript)
        merged = self._splice_segment(original_waveform, rebuilt_waveform, start_sample, end_sample)
        peak = merged.abs().max().item()
        if peak > 0.99:
            merged = merged * (0.99 / peak)

        out_path = save_as_mp3(merged, self.TARGET_SR)
        duration = merged.shape[1] / self.TARGET_SR
        b2_key = f"rebuild/{track_id}/{job_id}.mp3"
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

    async def _synthesize_higgs(self, text: str) -> bytes:
        if not text.strip():
            return await self._synthesize(" ", DEFAULT_VOICE)
        if not settings.HIGGS_AUDIO_ENABLED:
            raise RuntimeError("Higgs audio is disabled")
        module_spec = importlib.util.find_spec("higgs_audio")
        if module_spec is None:
            raise RuntimeError("higgs_audio module is not installed for self-hosted rebuild")
        return await asyncio.get_event_loop().run_in_executor(None, self._run_local_higgs, text)

    def _run_local_higgs(self, text: str) -> bytes:
        import higgs_audio

        if hasattr(higgs_audio, "synthesize"):
            out = higgs_audio.synthesize(text=text, voice=settings.HIGGS_AUDIO_VOICE)
            if isinstance(out, bytes):
                return out
            if isinstance(out, str) and os.path.exists(out):
                with open(out, "rb") as f:
                    return f.read()
        raise RuntimeError("Local higgs_audio module did not return audio bytes")

    def _detect_speech_bounds(self, waveform: torch.Tensor, sr: int, original_transcript: str) -> tuple[int, int]:
        mono = waveform.mean(dim=0).cpu().numpy()
        if mono.size < 2:
            return 0, waveform.shape[1]
        frame = max(256, int(sr * 0.02))
        kernel = np.ones(frame) / frame
        envelope = np.convolve(np.abs(mono), kernel, mode="same")
        threshold = max(float(np.percentile(envelope, 65) * 0.45), 1e-4)
        active = np.where(envelope >= threshold)[0]
        if active.size == 0:
            return 0, waveform.shape[1]
        pad = int(sr * (0.15 if original_transcript.strip() else 0.05))
        start = max(0, int(active[0]) - pad)
        end = min(waveform.shape[1], int(active[-1]) + pad)
        if end <= start:
            return 0, waveform.shape[1]
        return start, end

    def _splice_segment(
        self,
        original_waveform: torch.Tensor,
        replacement_waveform: torch.Tensor,
        start_sample: int,
        end_sample: int,
    ) -> torch.Tensor:
        target_length = max(1, end_sample - start_sample)
        if replacement_waveform.shape[0] != original_waveform.shape[0]:
            replacement_waveform = replacement_waveform.mean(dim=0, keepdim=True).expand(original_waveform.shape[0], -1)
        if replacement_waveform.shape[1] < target_length:
            pad = torch.zeros(replacement_waveform.shape[0], target_length - replacement_waveform.shape[1])
            replacement_waveform = torch.cat([replacement_waveform, pad], dim=1)
        elif replacement_waveform.shape[1] > target_length:
            speed_factor = replacement_waveform.shape[1] / target_length
            replacement_waveform = F_audio.resample(
                replacement_waveform,
                int(self.TARGET_SR * speed_factor),
                self.TARGET_SR,
            )[:, :target_length]

        before = original_waveform[:, :start_sample]
        after = original_waveform[:, end_sample:]
        cross_len = min(int(0.02 * self.TARGET_SR), before.shape[1], replacement_waveform.shape[1], after.shape[1] if after.shape[1] else replacement_waveform.shape[1])
        if cross_len > 0:
            fade_out = torch.linspace(1.0, 0.0, cross_len).unsqueeze(0)
            fade_in = torch.linspace(0.0, 1.0, cross_len).unsqueeze(0)
            before[:, -cross_len:] = (before[:, -cross_len:] * fade_out) + (replacement_waveform[:, :cross_len] * fade_in)
            tail_out = torch.linspace(1.0, 0.0, cross_len).unsqueeze(0)
            tail_in = torch.linspace(0.0, 1.0, cross_len).unsqueeze(0)
            if after.shape[1] >= cross_len:
                replacement_waveform[:, -cross_len:] = (replacement_waveform[:, -cross_len:] * tail_out) + (after[:, :cross_len] * tail_in)
                after = after[:, cross_len:]
        return torch.cat([before, replacement_waveform, after], dim=1)
