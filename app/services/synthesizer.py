import asyncio
import io
import logging
import os
import re
import tempfile
import wave
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F_audio

from app.config import settings
from app.core.audio_utils import save_as_mp3
from app.core.hear_temp import (
    drop_temp_standalone,
    hear_temp_directory,
    hear_temp_job_dir,
    register_temp_standalone,
)
from app.core.storage import get_storage
from app.services.enhancer_utils.tts_post_processor import TTSPostProcessor
from app.services.enhancer_utils_old.noise_reducer import NoiseReducer
from app.services.fishspeech_client import FishSpeechClient
from app.services.transcriber import TranscriptionService

logger = logging.getLogger(__name__)

_transcriber_instance: TranscriptionService | None = None


def _get_transcriber() -> TranscriptionService:
    global _transcriber_instance
    if _transcriber_instance is None:
        _transcriber_instance = TranscriptionService()
    return _transcriber_instance


_recon_payload_logger = logging.getLogger("reconstruct_payload")
_recon_payload_logger.setLevel(logging.INFO)
_recon_payload_fh = logging.FileHandler("/workspace/hear-ai/logs/reconstruct_payload.log")
_recon_payload_fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
_recon_payload_logger.addHandler(_recon_payload_fh)
_recon_payload_logger.propagate = False


@dataclass
class SynthesisResult:
    b2_key: str
    audio_url: str
    duration: float


class SpeechSynthesizer:
    TARGET_SR = 44100

    TTS_SYSTEM_PROMPT = """You are a text preprocessor for the fish-speech TTS engine. The engine under-weights punctuation, but it DOES respect inline control tokens in square brackets. Rewrite the input text so pauses, emotion, and delivery are expressed through these tokens. The input can be any kind of text: news, stories, dialogue, letters, lists, transcripts.

SUPPORTED TOKENS (use only these, or free-form variants in the same style):
- Emotion: [excited], [sad], [angry], [surprised], [delight]
- Volume: [whisper], [low voice], [volume up], [loud], [shouting], [screaming]
- Pacing: [pause], [short pause], [inhale], [exhale], [sigh]
- Vocalization: [laugh], [laughing], [chuckle], [chuckling], [tsk], [clearing throat]
- Tone: [professional broadcast tone], [singing], [with strong accent]
- Expression: [moaning], [panting], [echo], [pitch up], [pitch down]
- Free-form allowed, e.g. [speaking slowly and clearly], [sarcastic tone]

RULES:
1. Split text into short sentences of max 15-20 words, breaking long sentences at clause boundaries. One sentence per line.
2. Every sentence ends with exactly one terminal mark: \".\" \"?\" or \"!\".
3. Punctuation mapping:
   - Paragraph break or topic change -> a line containing only [pause].
   - Ellipsis \"...\" -> [pause] then continue as a new sentence.
   - Exclamation -> keep \"!\" and prepend a fitting emotion token chosen from context.
   - Question -> keep \"?\"; add [pitch up] only if it would otherwise sound flat.
   - Em dash / semicolon / colon -> split into a new sentence.
4. Convert stage directions and narration cues into tokens: \"(laughs)\" -> [laugh], \"*sighs*\" -> [sigh], \"she whispered\" -> [whisper] before the whispered text.
5. Place tokens BEFORE the text they modify. Max two tokens per sentence.
6. Use tokens sparingly: at most one emotion token per 3-4 sentences unless the text clearly demands more.
7. Expand numbers, dates, times, currencies, units and abbreviations into spoken UK English words (\"\u00a312.50\" -> \"twelve pounds fifty\", \"Dr.\" -> \"Doctor\", \"3rd\" -> \"third\", \"14:30\" -> \"half past two in the afternoon\").
8. Never paraphrase, summarise, add or remove content. Only restructure and annotate.
9. Output plain text only. One sentence per line. No markdown, no numbering, no commentary, no explanation."""

    def __init__(self):
        self._loaded = False
        self._fishspeech_available = False
        self._noise = NoiseReducer()

    def load(self):
        self._fishspeech_available = settings.FISH_SPEECH_TTS_ENABLED
        if self._fishspeech_available:
            print("[STARTUP] Fish Speech client ready via HTTP API")
        else:
            print("[STARTUP] Fish Speech disabled (pipeline, categorization, discovery unaffected)")
        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def fishspeech_available(self) -> bool:
        return self._fishspeech_available

    def _compute_seed(self, job_id: str, track_id: str) -> int:
        return abs(hash(f"{job_id}:{track_id}")) % (2**31)

    @staticmethod
    def _analyze_prosody(
        waveform: torch.Tensor,
        sr: int,
        start_sample: int,
        end_sample: int,
    ) -> dict:
        """Analyze reference audio prosody and map to FishSpeech emotion parameters.

        Extracts energy, pitch variation, and speaking rate from the reference
        audio segment and maps them to temperature, top_p, and chunk_length
        values that FishSpeech understands.

        Returns a dict with optional ``temperature``, ``top_p``, ``chunk_length``
        keys.  None means "use FishSpeech defaults".
        """
        clip = waveform[:, max(0, start_sample):min(end_sample, waveform.shape[1])]
        if clip.shape[1] < int(sr * 0.1):
            return {}

        mono = clip.mean(dim=0).cpu().numpy()

        energy = float(np.sqrt(np.mean(mono ** 2)))
        if energy < 1e-8:
            return {}

        energy_db = 20.0 * np.log10(max(energy, 1e-10))

        frame_len = int(sr * 0.03)
        hop = frame_len // 2
        num_frames = max(1, (len(mono) - frame_len) // hop + 1)
        frame_rms = np.array(
            [float(np.sqrt(np.mean(mono[i * hop : i * hop + frame_len] ** 2)))
             for i in range(num_frames)]
        )
        frame_rms = frame_rms[frame_rms > 1e-10]
        rms_variation = float(np.std(frame_rms) / max(np.mean(frame_rms), 1e-10)) if len(frame_rms) > 1 else 0.0
        zero_crossings = np.sum(np.abs(np.diff(np.sign(mono)))) / len(mono)

        dur_s = clip.shape[1] / sr
        word_count = max(1, zero_crossings * sr * 0.06)
        speaking_rate = word_count / max(dur_s, 0.1)

        temperature = None
        top_p = None
        chunk_length = None

        if rms_variation > 0.6:
            temperature = 0.85
            top_p = 0.85
        elif rms_variation > 0.35:
            temperature = 0.8
            top_p = 0.8
        elif rms_variation > 0.15:
            temperature = 0.7
            top_p = 0.75

        if energy_db > -12:
            temperature = (temperature or 0.8) * 1.05
        elif energy_db < -25:
            temperature = (temperature or 0.8) * 0.9

        if speaking_rate > 3.5:
            chunk_length = 250
        elif speaking_rate > 2.5:
            chunk_length = 200
        elif speaking_rate < 1.2:
            chunk_length = 150

        result = {}
        if temperature is not None:
            result["temperature"] = max(0.1, min(1.0, temperature))
        if top_p is not None:
            result["top_p"] = max(0.1, min(1.0, top_p))
        if chunk_length is not None:
            result["chunk_length"] = max(100, min(1000, chunk_length))

        logger.debug(
            "Prosody analysis: energy=%.1fdB rms_var=%.2f rate=%.1fw/s "
            "-> temp=%s top_p=%s chunk=%s",
            energy_db, rms_variation, speaking_rate,
            result.get("temperature"), result.get("top_p"),
            result.get("chunk_length"),
        )
        return result

    async def reconstruct_segment(
        self,
        original_audio_path: str,
        segment_start: float,
        segment_end: float,
        new_text: str,
        track_id: str,
        same_speaker: bool = True,
        original_text: str | None = None,
        job_id: str | None = None,
    ) -> SynthesisResult:
        original_waveform, orig_sr = torchaudio.load(original_audio_path)
        if orig_sr != self.TARGET_SR:
            original_waveform = F_audio.resample(original_waveform, orig_sr, self.TARGET_SR)

        start_sample = int(segment_start * self.TARGET_SR)
        end_sample = int(segment_end * self.TARGET_SR)

        seed = self._compute_seed(job_id or track_id, track_id) if job_id else None

        reference_path = None
        if same_speaker:
            reference_path = self._export_reference_clip(
                original_waveform, start_sample, end_sample, track_id=track_id
            )
        try:
            tts_bytes = await self._generate_segment_groups(
                new_text, reference_audio_path=reference_path,
                original_text=original_text, track_id=track_id, seed=seed,
            )
        finally:
            if reference_path:
                drop_temp_standalone(reference_path)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir=hear_temp_directory()) as tmp:
            tmp.write(tts_bytes)
            tts_path = tmp.name
        register_temp_standalone(tts_path, purpose="tts_segment", track_id=track_id)

        tts_waveform, tts_sr = torchaudio.load(tts_path)
        drop_temp_standalone(tts_path)
        if tts_sr != self.TARGET_SR:
            tts_waveform = F_audio.resample(tts_waveform, tts_sr, self.TARGET_SR)

        ref_segment = original_waveform[:, start_sample:end_sample]
        ref_radius = int(2.0 * self.TARGET_SR)
        if ref_segment.shape[1] < int(self.TARGET_SR * 1.0):
            ref_start = max(0, start_sample - ref_radius)
            ref_end = min(original_waveform.shape[1], end_sample + ref_radius)
            ref_segment = original_waveform[:, ref_start:ref_end]

        tts_waveform = self._time_stretch_to_match(tts_waveform, ref_segment)
        tts_waveform = TTSPostProcessor.process(tts_waveform, ref_segment, self.TARGET_SR)

        reconstructed = self._splice_segment(original_waveform, tts_waveform, start_sample, end_sample)

        out_path = save_as_mp3(reconstructed, self.TARGET_SR, track_id=track_id, purpose="reconstruct_mp3")
        duration = reconstructed.shape[1] / self.TARGET_SR
        b2_key = f"reconstructed/{track_id}/{os.urandom(8).hex()}.mp3"
        loop = asyncio.get_event_loop()
        try:
            audio_url = await loop.run_in_executor(None, get_storage().upload_file, out_path, b2_key)
        finally:
            drop_temp_standalone(out_path)

        return SynthesisResult(b2_key=b2_key, audio_url=audio_url, duration=round(duration, 3))

    async def reconstruct_segments(
        self,
        original_audio_path: str,
        track_id: str,
        changes: list,
        same_speaker: bool = True,
        job_id: str | None = None,
    ) -> SynthesisResult:
        original_waveform, orig_sr = torchaudio.load(original_audio_path)
        if orig_sr != self.TARGET_SR:
            original_waveform = F_audio.resample(original_waveform, orig_sr, self.TARGET_SR)

        merged = original_waveform
        normalized = self._normalize_changes(changes)
        if not normalized:
            raise ValueError("reconstruct requires non-empty segment changes")

        seed = self._compute_seed(job_id or track_id, track_id) if job_id else None

        normalized = sorted(normalized, key=lambda c: float(c["segment_start"]), reverse=True)
        for change in normalized:
            start_sample = int(float(change["segment_start"]) * self.TARGET_SR)
            end_sample = int(float(change["segment_end"]) * self.TARGET_SR)
            start_sample = max(0, min(start_sample, merged.shape[1] - 1 if merged.shape[1] else 0))
            end_sample = max(start_sample + 1, min(end_sample, merged.shape[1]))

            if change.get("is_deletion"):
                _recon_payload_logger.info(
                    "DELETE | seg=%.1fs-%.1fs | track=%s",
                    change["segment_start"], change["segment_end"], track_id,
                )
                merged = self._splice_segment(
                    merged,
                    torch.zeros_like(merged[:, :0]),
                    start_sample, end_sample,
                )
                continue

            clip_samples = end_sample - start_sample
            min_samples = int(3.0 * self.TARGET_SR)
            if clip_samples < min_samples:
                radius = min_samples // 2
                ref_start = max(0, start_sample - radius)
                ref_end = min(merged.shape[1], end_sample + radius)
                if ref_end - ref_start < min_samples:
                    ref_start = max(0, ref_end - min_samples)
                    ref_end = min(merged.shape[1], ref_start + min_samples)
            else:
                ref_start = start_sample
                ref_end = end_sample

            ref_text_for_clone = change.get("original_text") or None
            reference_path = None
            if same_speaker:
                reference_path = self._export_reference_clip(
                    merged, start_sample, end_sample, track_id=track_id
                )

            _recon_payload_logger.info(
                "CHANGE | seg=%.1fs-%.1fs | original_text='%s' | new_text='%s' | ref_text_for_clone='%s' | ref_path=%s | track=%s",
                change["segment_start"], change["segment_end"],
                change.get("original_text", "")[:80],
                change["new_text"][:120],
                (ref_text_for_clone or "<AUTO>")[:80],
                reference_path or "<NONE>",
                track_id,
            )

            try:
                tts_bytes = await self._generate_segment_groups(
                    change["new_text"],
                    reference_audio_path=reference_path,
                    original_text=ref_text_for_clone,
                    track_id=track_id,
                    seed=seed,
                )
            finally:
                if reference_path:
                    drop_temp_standalone(reference_path)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=hear_temp_directory()) as tmp:
                tmp.write(tts_bytes)
                tts_path = tmp.name
            register_temp_standalone(tts_path, purpose="tts_segment", track_id=track_id)
            tts_waveform, tts_sr = torchaudio.load(tts_path)
            drop_temp_standalone(tts_path)
            if tts_sr != self.TARGET_SR:
                tts_waveform = F_audio.resample(tts_waveform, tts_sr, self.TARGET_SR)

            ref_segment = merged[:, start_sample:end_sample]
            ref_radius = int(2.0 * self.TARGET_SR)
            if ref_segment.shape[1] < int(self.TARGET_SR * 1.0):
                ref_start = max(0, start_sample - ref_radius)
                ref_end = min(merged.shape[1], end_sample + ref_radius)
                ref_segment = merged[:, ref_start:ref_end]

            tts_waveform = self._time_stretch_to_match(tts_waveform, ref_segment)
            tts_waveform = TTSPostProcessor.process(tts_waveform, ref_segment, self.TARGET_SR)

            orig_seg_dur = (end_sample - start_sample) / self.TARGET_SR
            tts_seg_dur = tts_waveform.shape[1] / self.TARGET_SR
            logger.info(
                "Segment [%.1f-%.1f]: orig=%.2fs tts=%.2fs delta=%+.2fs text='%s'",
                change["segment_start"], change["segment_end"],
                orig_seg_dur, tts_seg_dur, tts_seg_dur - orig_seg_dur,
                change["new_text"][:50],
            )

            merged = self._splice_segment(merged, tts_waveform, start_sample, end_sample)

        peak = merged.abs().max().item()
        if peak > 0.99:
            merged = merged * (0.99 / peak)

        out_path = save_as_mp3(merged, self.TARGET_SR, job_id=job_id, track_id=track_id, purpose="reconstruct_mp3")
        duration = merged.shape[1] / self.TARGET_SR
        b2_key = f"reconstructed/{track_id}/{os.urandom(8).hex()}.mp3"
        loop = asyncio.get_event_loop()
        try:
            audio_url = await loop.run_in_executor(None, get_storage().upload_file, out_path, b2_key)
        finally:
            drop_temp_standalone(out_path)

        return SynthesisResult(b2_key=b2_key, audio_url=audio_url, duration=round(duration, 3))


    async def reconstruct_segments_batched(
        self,
        original_audio_path: str,
        track_id: str,
        changes: list,
        same_speaker: bool = True,
        voice_reference_path: str | None = None,
        job_id: str | None = None,
    ) -> SynthesisResult:
        normalized = self._normalize_changes(changes)
        if not normalized:
            raise ValueError("reconstruct requires non-empty segment changes")

        original_waveform, orig_sr = torchaudio.load(original_audio_path)
        if orig_sr != self.TARGET_SR:
            original_waveform = F_audio.resample(original_waveform, orig_sr, self.TARGET_SR)

        seed = self._compute_seed(job_id or track_id, track_id) if job_id else None
        batches = self._split_into_batches(normalized, original_waveform)
        merged = original_waveform

        for batch_idx, batch in enumerate(batches):
            batch_sorted = sorted(batch, key=lambda c: float(c["segment_start"]), reverse=True)
            for change in batch_sorted:
                start_sample = int(float(change["segment_start"]) * self.TARGET_SR)
                end_sample = int(float(change["segment_end"]) * self.TARGET_SR)
                start_sample = max(0, min(start_sample, merged.shape[1] - 1 if merged.shape[1] else 0))
                end_sample = max(start_sample + 1, min(end_sample, merged.shape[1]))

                if change.get("is_deletion"):
                    merged = self._splice_segment(
                        merged,
                        torch.zeros_like(merged[:, :0]),
                        start_sample, end_sample,
                    )
                    continue

                ref_path = voice_reference_path
                ref_text = change.get("original_text") or None
                if same_speaker and not ref_path:
                    ref_path = self._export_reference_clip(
                        merged, start_sample, end_sample, track_id=track_id
                    )
                try:
                    tts_bytes = await self._generate_segment_groups(
                        change["new_text"],
                        reference_audio_path=ref_path,
                        original_text=ref_text,
                        track_id=track_id,
                        seed=seed,
                    )
                finally:
                    if ref_path and ref_path != voice_reference_path:
                        drop_temp_standalone(ref_path)

                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir=hear_temp_directory()) as tmp:
                    tmp.write(tts_bytes)
                    tts_path = tmp.name
                register_temp_standalone(tts_path, purpose="tts_segment", track_id=track_id)
                tts_waveform, tts_sr = torchaudio.load(tts_path)
                drop_temp_standalone(tts_path)
                if tts_sr != self.TARGET_SR:
                    tts_waveform = F_audio.resample(tts_waveform, tts_sr, self.TARGET_SR)

                ref_segment = merged[:, start_sample:end_sample]
                ref_radius = int(2.0 * self.TARGET_SR)
                if ref_segment.shape[1] < int(self.TARGET_SR * 1.0):
                    ref_start = max(0, start_sample - ref_radius)
                    ref_end = min(merged.shape[1], end_sample + ref_radius)
                    ref_segment = merged[:, ref_start:ref_end]

                tts_waveform = self._time_stretch_to_match(tts_waveform, ref_segment)
                tts_waveform = TTSPostProcessor.process(tts_waveform, ref_segment, self.TARGET_SR)
                merged = self._splice_segment(merged, tts_waveform, start_sample, end_sample)

            if batch_idx < len(batches) - 1:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                await asyncio.sleep(0.05)

        peak = merged.abs().max().item()
        if peak > 0.99:
            merged = merged * (0.99 / peak)

        out_path = save_as_mp3(merged, self.TARGET_SR, job_id=job_id, track_id=track_id, purpose="reconstruct_mp3")
        duration = merged.shape[1] / self.TARGET_SR
        b2_key = f"reconstructed/{track_id}/{os.urandom(8).hex()}.mp3"
        loop = asyncio.get_event_loop()
        try:
            audio_url = await loop.run_in_executor(None, get_storage().upload_file, out_path, b2_key)
        finally:
            drop_temp_standalone(out_path)

        return SynthesisResult(b2_key=b2_key, audio_url=audio_url, duration=round(duration, 3))

    async def generate_preview(
        self,
        original_audio_path: str,
        track_id: str,
        changes: list,
        same_speaker: bool = True,
        job_id: str | None = None,
    ) -> SynthesisResult:
        normalized = self._normalize_changes(changes)
        if not normalized:
            raise ValueError("preview requires non-empty segment changes")

        original_waveform, orig_sr = torchaudio.load(original_audio_path)
        if orig_sr != self.TARGET_SR:
            original_waveform = F_audio.resample(original_waveform, orig_sr, self.TARGET_SR)

        seed = self._compute_seed(job_id or track_id, track_id) if job_id else None
        waveforms: list[torch.Tensor] = []

        for change in normalized:
            start_sample = int(float(change["segment_start"]) * self.TARGET_SR)
            end_sample = int(float(change["segment_end"]) * self.TARGET_SR)
            text = change["new_text"]

            ref_text = change.get("original_text") or None
            reference_path = self._export_reference_clip(
                original_waveform, start_sample, end_sample, track_id=track_id,
            )

            try:
                tts_bytes = await self._generate_segment_groups(
                    text, reference_audio_path=reference_path,
                    original_text=ref_text,
                    track_id=track_id, seed=seed,
                )
            finally:
                if reference_path:
                    drop_temp_standalone(reference_path)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=hear_temp_directory()) as tmp:
                tmp.write(tts_bytes)
                tts_path = tmp.name
            register_temp_standalone(tts_path, purpose="preview_segment", track_id=track_id)
            wf, sr = torchaudio.load(tts_path)
            drop_temp_standalone(tts_path)
            if sr != self.TARGET_SR:
                wf = F_audio.resample(wf, sr, self.TARGET_SR)

            ref_segment = original_waveform[:, start_sample:end_sample]
            if ref_segment.shape[1] > 0:
                wf = self._time_stretch_to_match(wf, ref_segment)
                wf = TTSPostProcessor.process(wf, ref_segment, self.TARGET_SR)

            waveforms.append(wf)

        if len(waveforms) == 1:
            combined = waveforms[0]
        else:
            crossfade_samples = int(0.03 * self.TARGET_SR)
            combined_parts = [waveforms[0]]
            for wf in waveforms[1:]:
                prev = combined_parts[-1]
                if prev.shape[1] >= crossfade_samples and wf.shape[1] >= crossfade_samples:
                    fade_out = torch.linspace(1.0, 0.0, crossfade_samples).unsqueeze(0)
                    fade_in = torch.linspace(0.0, 1.0, crossfade_samples).unsqueeze(0)
                    prev[:, -crossfade_samples:] = prev[:, -crossfade_samples:] * fade_out + wf[:, :crossfade_samples] * fade_in
                    combined_parts[-1] = prev
                    combined_parts.append(wf[:, crossfade_samples:])
                else:
                    combined_parts.append(wf)
            combined = torch.cat(combined_parts, dim=1)

        peak = combined.abs().max().item()
        if peak > 0.99:
            combined = combined * (0.99 / peak)

        out_path = save_as_mp3(combined, self.TARGET_SR, track_id=track_id, purpose="preview_mp3")
        duration = combined.shape[1] / self.TARGET_SR
        b2_key = f"preview/{track_id}/{os.urandom(8).hex()}.mp3"
        loop = asyncio.get_event_loop()
        try:
            audio_url = await loop.run_in_executor(None, get_storage().upload_file, out_path, b2_key)
        finally:
            drop_temp_standalone(out_path)

        return SynthesisResult(b2_key=b2_key, audio_url=audio_url, duration=round(duration, 3))

    async def remove_segment(
        self,
        original_audio_path: str,
        track_id: str,
        segment_start: float,
        segment_end: float,
    ) -> SynthesisResult:
        original_waveform, orig_sr = torchaudio.load(original_audio_path)
        if orig_sr != self.TARGET_SR:
            original_waveform = F_audio.resample(original_waveform, orig_sr, self.TARGET_SR)

        start_sample = int(segment_start * self.TARGET_SR)
        end_sample = int(segment_end * self.TARGET_SR)
        start_sample = max(0, min(start_sample, original_waveform.shape[1]))
        end_sample = max(start_sample, min(end_sample, original_waveform.shape[1]))

        before = original_waveform[:, :start_sample]
        after = original_waveform[:, end_sample:]

        cross_len = min(int(0.05 * self.TARGET_SR), before.shape[1], after.shape[1])
        if cross_len > 0 and before.shape[1] > 0 and after.shape[1] > 0:
            fade_out = torch.linspace(1.0, 0.0, cross_len).unsqueeze(0)
            fade_in = torch.linspace(0.0, 1.0, cross_len).unsqueeze(0)
            before[:, -cross_len:] = before[:, -cross_len:] * fade_out + after[:, :cross_len] * fade_in
            after = after[:, cross_len:]

        merged = torch.cat([before, after], dim=1) if after.shape[1] > 0 else before

        peak = merged.abs().max().item()
        if peak > 0.99:
            merged = merged * (0.99 / peak)

        out_path = save_as_mp3(merged, self.TARGET_SR, track_id=track_id, purpose="remove_mp3")
        duration = merged.shape[1] / self.TARGET_SR
        b2_key = f"reconstructed/{track_id}/{os.urandom(8).hex()}.mp3"
        loop = asyncio.get_event_loop()
        try:
            audio_url = await loop.run_in_executor(None, get_storage().upload_file, out_path, b2_key)
        finally:
            drop_temp_standalone(out_path)

        return SynthesisResult(b2_key=b2_key, audio_url=audio_url, duration=round(duration, 3))

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

        seed = self._compute_seed(job_id, track_id)
        rebuilt_bytes = await self._generate_segment_groups(
            edited_transcript,
            reference_audio_path=original_audio_path,
            track_id=track_id,
            seed=seed,
        )
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir=hear_temp_directory()) as tmp:
            tmp.write(rebuilt_bytes)
            rebuilt_path = tmp.name
        register_temp_standalone(rebuilt_path, purpose="rebuilt_track_intermediate", job_id=job_id, track_id=track_id)

        rebuilt_waveform, rebuilt_sr = torchaudio.load(rebuilt_path)
        drop_temp_standalone(rebuilt_path)
        if rebuilt_sr != self.TARGET_SR:
            rebuilt_waveform = F_audio.resample(rebuilt_waveform, rebuilt_sr, self.TARGET_SR)

        start_sample, end_sample = self._detect_speech_bounds(original_waveform, self.TARGET_SR, original_transcript)
        ref_segment = original_waveform[:, start_sample:end_sample]
        if ref_segment.shape[1] < int(self.TARGET_SR * 0.05):
            ref_radius = int(2.0 * self.TARGET_SR)
            ref_start = max(0, start_sample - ref_radius)
            ref_end = min(original_waveform.shape[1], end_sample + ref_radius)
            ref_segment = original_waveform[:, ref_start:ref_end]

        rebuilt_waveform = self._time_stretch_to_match(rebuilt_waveform, ref_segment)
        rebuilt_waveform = TTSPostProcessor.process(rebuilt_waveform, ref_segment, self.TARGET_SR)

        merged = self._splice_segment(original_waveform, rebuilt_waveform, start_sample, end_sample)
        peak = merged.abs().max().item()
        if peak > 0.99:
            merged = merged * (0.99 / peak)

        out_path = save_as_mp3(merged, self.TARGET_SR, job_id=job_id, track_id=track_id, purpose="rebuilt_track_mp3")
        duration = merged.shape[1] / self.TARGET_SR
        b2_key = f"rebuild/{track_id}/{job_id}.mp3"
        loop = asyncio.get_event_loop()
        try:
            audio_url = await loop.run_in_executor(None, get_storage().upload_file, out_path, b2_key)
        finally:
            drop_temp_standalone(out_path)

        return SynthesisResult(b2_key=b2_key, audio_url=audio_url, duration=round(duration, 3))

    def _split_into_batches(
        self,
        changes: list[dict],
        waveform: torch.Tensor,
    ) -> list[list[dict]]:
        max_words = max(1, settings.EDIT_MAX_BATCH_WORDS)
        max_duration = max(1.0, settings.EDIT_MAX_BATCH_DURATION)

        sorted_changes = sorted(changes, key=lambda c: float(c["segment_start"]))
        batches: list[list[dict]] = []
        current_batch: list[dict] = []
        current_words = 0
        current_duration = 0.0

        for change in sorted_changes:
            words = len(change["new_text"].split())
            dur = float(change["segment_end"]) - float(change["segment_start"])
            if current_batch and (
                current_words + words > max_words or current_duration + dur > max_duration
            ):
                batches.append(current_batch)
                current_batch = []
                current_words = 0
                current_duration = 0.0
            current_batch.append(change)
            current_words += words
            current_duration += dur

        if current_batch:
            batches.append(current_batch)
        return batches if batches else [sorted_changes]

    def _normalize_changes(self, changes: list) -> list[dict]:
        normalized: list[dict] = []
        for item in changes or []:
            if isinstance(item, dict):
                start_raw = item.get("segment_start", item.get("start"))
                end_raw = item.get("segment_end", item.get("end"))
                text_raw = item.get("new_text", item.get("text"))
                original_text_raw = item.get("original_text")
                is_deletion = item.get("is_deletion", False)
            else:
                start_raw = getattr(item, "segment_start", getattr(item, "start", None))
                end_raw = getattr(item, "segment_end", getattr(item, "end", None))
                text_raw = getattr(item, "new_text", getattr(item, "text", None))
                original_text_raw = getattr(item, "original_text", None)
                is_deletion = getattr(item, "is_deletion", False)
            try:
                start = float(start_raw)
                end = float(end_raw)
            except Exception:
                continue
            text = str(text_raw or "").strip()
            if end < start:
                continue
            is_del = is_deletion or not text
            if is_del:
                if end <= start:
                    continue
                normalized.append({
                    "segment_start": start,
                    "segment_end": end,
                    "new_text": "",
                    "original_text": original_text_raw,
                    "is_deletion": True,
                })
            else:
                if not text:
                    continue
                normalized.append({
                    "segment_start": start,
                    "segment_end": end,
                    "new_text": text,
                    "original_text": original_text_raw,
                })
        return normalized

    async def _synthesize_fishspeech(
        self,
        text: str,
        reference_audio_path: str | None = None,
        original_text: str | None = None,
        seed: int | None = None,
        emotion_params: dict | None = None,
    ) -> bytes:
        if not settings.FISH_SPEECH_TTS_ENABLED:
            raise RuntimeError("Audio reconstruction is currently unavailable")

        processed_text = await self._preprocess_for_s2(text)

        _recon_payload_logger.info(
            "FISHSPEECH_REQ | text='%s' | processed='%s'",
            text[:120], processed_text[:120],
        )

        client = FishSpeechClient()
        refs = None
        if reference_audio_path and os.path.isfile(reference_audio_path):
            with open(reference_audio_path, "rb") as _rf:
                refs = [{"audio": _rf.read(), "text": original_text or ""}]
            _recon_payload_logger.info(
                "FISHSPEECH_REF | path='%s' size=%d text='%s'",
                reference_audio_path, len(refs[0]["audio"]) if refs else 0,
                (original_text or "")[:80],
            )
        return await client.generate_speech(text=processed_text, max_new_tokens=1024, references=refs)

    async def _preprocess_for_s2(self, text: str) -> str:
        """Add Fish Speech control tokens at punctuation and paragraph breaks.
        
        Token placement rules:
          [pause]        — after . ! ? and at paragraph breaks
          [short pause]  — after , ; : —
        """
        import re

        # Split into paragraphs
        paragraphs = re.split(r"\n\s*\n", text.strip())
        result_parts = []

        for pi, para in enumerate(paragraphs):
            if not para.strip():
                continue
            # Add [pause] at paragraph breaks
            if pi > 0:
                result_parts.append("[pause]")

            # Process sentence by sentence within paragraph
            lines = para.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Add [short pause] after commas, semicolons, colons, em-dashes
                line = re.sub(r",\s*", ", [short pause] ", line)
                line = re.sub(r";\s*", "; [short pause] ", line)
                line = re.sub(r":\s*", ": [short pause] ", line)
                line = re.sub(r"\u2014\s*", "\u2014 [short pause] ", line)

                # Add [pause] after periods, exclamation marks, question marks
                line = re.sub(r"\.\s+", ". [pause] ", line)
                line = re.sub(r"!\s+", "! [pause] ", line)
                line = re.sub(r"\?\s+", "? [pause] ", line)

                # Clean up: no double tags
                line = re.sub(r"\[pause\]\s*\[pause\]", "[pause]", line)
                line = re.sub(r"\[short pause\]\s*\[short pause\]", "[short pause]", line)

                result_parts.append(line)

        processed = " ".join(result_parts)
        return processed

    async def _generate_segment_groups(
        self,
        text: str,
        reference_audio_path: str | None = None,
        original_text: str | None = None,
        track_id: str | None = None,
        seed: int | None = None,
        emotion_params: dict | None = None,
    ) -> bytes:
        return await self._synthesize_fishspeech(
            text, reference_audio_path, original_text, seed=seed,
            emotion_params=emotion_params,
        )

    def _export_reference_clip(
        self,
        waveform: torch.Tensor,
        start_sample: int,
        end_sample: int,
        *,
        track_id: str | None = None,
    ) -> str:
        start_sample = max(0, start_sample)
        end_sample = max(start_sample, min(end_sample, waveform.shape[1]))

        max_ref_samples = int(10.0 * self.TARGET_SR)
        clip_samples = end_sample - start_sample
        if clip_samples > max_ref_samples:
            end_sample = start_sample + max_ref_samples

        clip = waveform[:, start_sample:end_sample].detach().cpu()
        dur_s = clip.shape[1] / self.TARGET_SR

        _recon_payload_logger.info(
            "REFERENCE_CLIP | segment=%ss-%ss | ref_dur=%.1fs | track=%s",
            round(start_sample / self.TARGET_SR, 2),
            round(end_sample / self.TARGET_SR, 2),
            round(dur_s, 2),
            track_id or "?",
        )

        # Clean reference clip for better FishSpeech voice cloning
        clip = self._noise.noise_gate(clip, self.TARGET_SR, threshold_db=-45.0)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=hear_temp_directory()) as tmp:
            ref_path = tmp.name
        torchaudio.save(ref_path, clip, self.TARGET_SR)
        register_temp_standalone(ref_path, purpose="reference_clip", track_id=track_id)
        return ref_path

    def _wav_bytes_from_audio(self, audio: np.ndarray, sampling_rate: int) -> bytes:
        pcm = np.clip(audio.astype(np.float32), -1.0, 1.0)
        pcm_i16 = (pcm * 32767.0).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sampling_rate))
            wf.writeframes(pcm_i16.tobytes())
        return buffer.getvalue()

    def _detect_speech_bounds(
        self, waveform: torch.Tensor, sr: int, original_transcript: str
    ) -> tuple[int, int]:
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

    def _time_stretch_to_match(
        self,
        tts_waveform: torch.Tensor,
        ref_waveform: torch.Tensor,
    ) -> torch.Tensor:
        target_dur = ref_waveform.shape[1] / self.TARGET_SR
        current_dur = tts_waveform.shape[1] / self.TARGET_SR
        if current_dur < 0.01:
            return tts_waveform
        ratio = target_dur / current_dur
        ratio = max(0.9, min(1.1, ratio))
        if abs(ratio - 1.0) < 0.05:
            return tts_waveform
        logger.info("Time-stretching TTS: %.2fs -> %.2fs (ratio=%.3f)", current_dur, target_dur, ratio)
        try:
            n_fft = 2048
            hop = n_fft // 4
            spec = torch.stft(
                tts_waveform.squeeze(0),
                n_fft=n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=torch.hann_window(n_fft, device=tts_waveform.device),
                return_complex=True,
            )
            phase_advance = torch.linspace(
                0, torch.pi * hop, spec.size(0), device=spec.device
            ).unsqueeze(1)
            stretched_spec = torchaudio.functional.phase_vocoder(
                spec.unsqueeze(0), ratio, phase_advance
            ).squeeze(0)
            stretched = torch.istft(
                stretched_spec,
                n_fft=n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=torch.hann_window(n_fft, device=tts_waveform.device),
            )
            return stretched.unsqueeze(0)
        except Exception as e:
            logger.warning("Time-stretch failed: %s, using original", e)
            return tts_waveform

    def _splice_segment(
        self,
        original_waveform: torch.Tensor,
        replacement_waveform: torch.Tensor,
        start_sample: int,
        end_sample: int,
    ) -> torch.Tensor:
        if replacement_waveform.shape[0] != original_waveform.shape[0]:
            replacement_waveform = replacement_waveform.mean(dim=0, keepdim=True).expand(
                original_waveform.shape[0], -1
            )

        before = original_waveform[:, :start_sample]
        after = original_waveform[:, end_sample:]

        is_removal = replacement_waveform.shape[1] == 0

        if is_removal:
            cross_len = min(int(0.03 * self.TARGET_SR), before.shape[1], after.shape[1])
            if cross_len > 0 and before.shape[1] > 0 and after.shape[1] > 0:
                fade_out = torch.linspace(1.0, 0.0, cross_len).unsqueeze(0)
                fade_in = torch.linspace(0.0, 1.0, cross_len).unsqueeze(0)
                before[:, -cross_len:] = before[:, -cross_len:] * fade_out + after[:, :cross_len] * fade_in
                after = after[:, cross_len:]
            return torch.cat([before, after], dim=1) if after.shape[1] > 0 else before

        if replacement_waveform.shape[1] > int(0.3 * self.TARGET_SR) and after.shape[1] > int(0.3 * self.TARGET_SR):
            tail_len = min(int(0.2 * self.TARGET_SR), replacement_waveform.shape[1])
            head_len = min(int(0.2 * self.TARGET_SR), after.shape[1])
            if tail_len > 0 and head_len > 0:
                tail = replacement_waveform[0, -tail_len:].float()
                head = after[0, :head_len].float()
                tail_norm = tail / (tail.norm() + 1e-10)
                head_norm = head / (head.norm() + 1e-10)
                correlation = torch.nn.functional.conv1d(
                    head_norm.view(1, 1, -1),
                    tail_norm.flip(0).view(1, 1, -1),
                    padding=tail_len - 1,
                ).squeeze()
                max_corr = correlation.max().item()
                if max_corr > 0.35:
                    best_offset = correlation.argmax().item() - tail_len + 1
                    if best_offset > 0:
                        after = after[:, best_offset:]

        is_removal = replacement_waveform.shape[1] == 0

        if is_removal:
            cross_len = min(int(0.03 * self.TARGET_SR), before.shape[1], after.shape[1])
            if cross_len > 0 and before.shape[1] > 0 and after.shape[1] > 0:
                fade_out = torch.linspace(1.0, 0.0, cross_len).unsqueeze(0)
                fade_in = torch.linspace(0.0, 1.0, cross_len).unsqueeze(0)
                before[:, -cross_len:] = before[:, -cross_len:] * fade_out + after[:, :cross_len] * fade_in
                after = after[:, cross_len:]
            return torch.cat([before, after], dim=1) if after.shape[1] > 0 else before

        cross_len = min(
            int(0.03 * self.TARGET_SR),
            before.shape[1],
            replacement_waveform.shape[1],
        )

        if cross_len > 0 and before.shape[1] > 0 and replacement_waveform.shape[1] > 0:
            fade_out = torch.linspace(1.0, 0.0, cross_len).unsqueeze(0)
            fade_in = torch.linspace(0.0, 1.0, cross_len).unsqueeze(0)
            before[:, -cross_len:] = before[:, -cross_len:] * fade_out + replacement_waveform[:, :cross_len] * fade_in
            replacement_waveform = replacement_waveform[:, cross_len:]

        cross_end = min(cross_len, replacement_waveform.shape[1], after.shape[1])
        if cross_end > 0 and after.shape[1] > 0:
            tail_out = torch.linspace(1.0, 0.0, cross_end).unsqueeze(0)
            tail_in = torch.linspace(0.0, 1.0, cross_end).unsqueeze(0)
            replacement_waveform[:, -cross_end:] = (
                replacement_waveform[:, -cross_end:] * tail_out
                + after[:, :cross_end] * tail_in
            )
            after = after[:, cross_end:]

        if after.shape[1] == 0:
            return torch.cat([before, replacement_waveform], dim=1)
        return torch.cat([before, replacement_waveform, after], dim=1)
