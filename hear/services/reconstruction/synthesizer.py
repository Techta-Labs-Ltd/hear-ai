import asyncio
import hashlib
import io
import logging
import os
import re
import tempfile
import warnings
import wave
from dataclasses import dataclass, field

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F_audio

from hear.config import settings
from hear.core.audio_utils import save_as_mp3
from hear.core.hear_temp import (
    drop_temp_standalone,
    hear_temp_directory,
)
from hear.core.storage import B2Storage
from hear.core.noise import NoiseReducer
from hear.services.model_client import RayModelClient
from hear.services.reconstruction.tts_post_processor import TTSPostProcessor
from hear.services.transcription.service import TranscriptionService

logger = logging.getLogger(__name__)

_recon_payload_logger = logging.getLogger("reconstruct_payload")


@dataclass
class SegmentAudioResult:
    segment_start: float
    segment_end: float
    b2_key: str = ""
    audio_url: str = ""
    duration: float = 0.0
    is_deletion: bool = False
    bucket_name: str = ""
    backend_id: str = ""


@dataclass
class SynthesisResult:
    b2_key: str
    audio_url: str
    duration: float
    segments: list[SegmentAudioResult] = field(default_factory=list)
    bucket_name: str = ""
    backend_id: str = ""


class SpeechSynthesizer:
    TARGET_SR = 44100
    VOICE_REFERENCE_SECONDS = 10.0
    VOICE_REFERENCE_GUARD_SECONDS = 0.25
    MIN_VOICE_REFERENCE_SECONDS = 1.0
    MIN_PACING_TARGET_SECONDS = 0.15
    MAX_PACING_REFERENCE_SECONDS = 30.0
    PACING_ACTIVITY_FRAME_MS = 20
    MIN_PLAUSIBLE_SPEAKING_RATE = 0.5
    MAX_PLAUSIBLE_SPEAKING_RATE = 8.0
    MIN_SAFE_TEMPO_FACTOR = 0.75
    MAX_SAFE_TEMPO_FACTOR = 1.35

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

    def __init__(self, model_client: RayModelClient, transcriber: TranscriptionService):
        self._model_client = model_client
        self._transcriber = transcriber
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

    def _compute_seed(self, _job_id: str, track_id: str) -> int:
        """Keep Fish sampling stable when the same track is regenerated again."""
        digest = hashlib.sha256(
            f"reconstruction:{track_id}".encode()
        ).digest()
        return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF

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
        storage: B2Storage,
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
        reference_text = None
        reference_speaking_rate = None
        try:
            if same_speaker:
                (
                    reference_path,
                    reference_text,
                    reference_speaking_rate,
                ) = await self._prepare_voice_reference(
                    original_waveform,
                    start_sample,
                    end_sample,
                    track_id=track_id,
                    original_text=original_text,
                )
            tts_bytes = await self._generate_segment_groups(
                new_text, reference_audio_path=reference_path,
                original_text=reference_text, track_id=track_id, seed=seed,
            )
        finally:
            if reference_path:
                drop_temp_standalone(reference_path)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir=hear_temp_directory()) as tmp:
            tmp.write(tts_bytes)
            tts_path = tmp.name

        tts_waveform, tts_sr = torchaudio.load(tts_path)
        drop_temp_standalone(tts_path)
        if tts_sr != self.TARGET_SR:
            tts_waveform = F_audio.resample(tts_waveform, tts_sr, self.TARGET_SR)

        duration_reference = original_waveform[:, start_sample:end_sample]
        ref_segment = duration_reference
        ref_radius = int(2.0 * self.TARGET_SR)
        if ref_segment.shape[1] < int(self.TARGET_SR * 1.0):
            ref_start = max(0, start_sample - ref_radius)
            ref_end = min(original_waveform.shape[1], end_sample + ref_radius)
            ref_segment = original_waveform[:, ref_start:ref_end]

        tts_waveform = await asyncio.to_thread(
            TTSPostProcessor.process,
            tts_waveform,
            ref_segment,
            self.TARGET_SR,
            match_reference_pitch=same_speaker,
        )
        pacing_reference, pacing_text = self._pacing_reference(
            original_waveform,
            start_sample,
            end_sample,
            original_text=original_text,
            reference_text=reference_text,
        )
        tts_waveform = self._time_stretch_to_match(
            tts_waveform,
            pacing_reference,
            new_text=new_text,
            original_text=pacing_text,
            source_speaking_rate=reference_speaking_rate,
        )

        reconstructed = self._splice_segment(original_waveform, tts_waveform, start_sample, end_sample)

        out_path = save_as_mp3(reconstructed, self.TARGET_SR, track_id=track_id, purpose="reconstruct_mp3")
        duration = reconstructed.shape[1] / self.TARGET_SR
        b2_key = storage.key("reconstructed", f"{job_id or track_id}.mp3")
        loop = asyncio.get_event_loop()
        try:
            audio_url = await loop.run_in_executor(
                None, storage.upload_file, out_path, b2_key, "audio/mpeg"
            )
        finally:
            drop_temp_standalone(out_path)

        return SynthesisResult(b2_key=b2_key, audio_url=audio_url, duration=round(duration, 3), bucket_name=storage.bucket_name)

    async def reconstruct_segments(
        self,
        original_audio_path: str,
        track_id: str,
        changes: list,
        storage: B2Storage,
        same_speaker: bool = True,
        voice_reference_audio_path: str | None = None,
        job_id: str | None = None,
        run_id: str | None = None,
    ) -> SynthesisResult:
        original_waveform, orig_sr = torchaudio.load(original_audio_path)
        if orig_sr != self.TARGET_SR:
            original_waveform = F_audio.resample(original_waveform, orig_sr, self.TARGET_SR)
        reference_waveform = original_waveform
        if (
            voice_reference_audio_path
            and voice_reference_audio_path != original_audio_path
        ):
            reference_waveform, reference_sr = torchaudio.load(
                voice_reference_audio_path
            )
            if reference_sr != self.TARGET_SR:
                reference_waveform = F_audio.resample(
                    reference_waveform,
                    reference_sr,
                    self.TARGET_SR,
                )

        merged = original_waveform.clone()
        normalized = self._normalize_changes(changes)
        if not normalized:
            raise ValueError("reconstruct requires non-empty segment changes")
        segment_results: list[SegmentAudioResult] = []

        seed = self._compute_seed(job_id or track_id, track_id) if job_id else None

        normalized = sorted(normalized, key=lambda c: float(c["segment_start"]), reverse=True)
        for change in normalized:
            timeline_start = int(
                float(change["segment_start"]) * self.TARGET_SR
            )
            timeline_end = int(
                float(change["segment_end"]) * self.TARGET_SR
            )
            start_sample = max(
                0,
                min(
                    timeline_start,
                    merged.shape[1] - 1 if merged.shape[1] else 0,
                ),
            )
            end_sample = max(
                start_sample + 1,
                min(timeline_end, merged.shape[1]),
            )
            reference_start = max(
                0,
                min(timeline_start, reference_waveform.shape[1]),
            )
            reference_end = max(
                reference_start,
                min(timeline_end, reference_waveform.shape[1]),
            )

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
                segment_results.append(SegmentAudioResult(
                    segment_start=float(change["segment_start"]),
                    segment_end=float(change["segment_end"]),
                    is_deletion=True,
                ))
                continue

            original_text = str(change.get("original_text") or "").strip()
            ref_text_for_clone = original_text or None
            reference_path = None
            reference_speaking_rate = None
            try:
                if same_speaker:
                    (
                        reference_path,
                        ref_text_for_clone,
                        reference_speaking_rate,
                    ) = await self._prepare_voice_reference(
                        reference_waveform,
                        reference_start,
                        reference_end,
                        track_id=track_id,
                        original_text=original_text,
                    )

                _recon_payload_logger.info(
                    "CHANGE | seg=%.1fs-%.1fs | original_text='%s' | new_text='%s' "
                    "| ref_text_for_clone='%s' | ref_path=%s | track=%s",
                    change["segment_start"], change["segment_end"],
                    change.get("original_text", "")[:80],
                    change["new_text"][:120],
                    (ref_text_for_clone or "<AUTO>")[:80],
                    reference_path or "<NONE>",
                    track_id,
                )

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
            tts_waveform, tts_sr = torchaudio.load(tts_path)
            drop_temp_standalone(tts_path)
            if tts_sr != self.TARGET_SR:
                tts_waveform = F_audio.resample(tts_waveform, tts_sr, self.TARGET_SR)

            ref_segment = reference_waveform[
                :, reference_start:reference_end
            ]
            ref_radius = int(2.0 * self.TARGET_SR)
            if ref_segment.shape[1] < int(self.TARGET_SR * 1.0):
                ref_start = max(0, reference_start - ref_radius)
                ref_end = min(
                    reference_waveform.shape[1],
                    reference_end + ref_radius,
                )
                ref_segment = reference_waveform[:, ref_start:ref_end]

            tts_waveform = await asyncio.to_thread(
                TTSPostProcessor.process,
                tts_waveform,
                ref_segment,
                self.TARGET_SR,
                match_reference_pitch=same_speaker,
            )
            pacing_reference, pacing_text = self._pacing_reference(
                reference_waveform,
                reference_start,
                reference_end,
                original_text=original_text,
                reference_text=ref_text_for_clone,
            )
            tts_waveform = self._time_stretch_to_match(
                tts_waveform,
                pacing_reference,
                new_text=change["new_text"],
                original_text=pacing_text,
                source_speaking_rate=reference_speaking_rate,
            )

            orig_seg_dur = (end_sample - start_sample) / self.TARGET_SR
            tts_seg_dur = tts_waveform.shape[1] / self.TARGET_SR
            logger.info(
                "Segment [%.1f-%.1f]: orig=%.2fs tts=%.2fs delta=%+.2fs text='%s'",
                change["segment_start"], change["segment_end"],
                orig_seg_dur, tts_seg_dur, tts_seg_dur - orig_seg_dur,
                change["new_text"][:50],
            )

            segment_results.append(await self._upload_segment_audio(
                tts_waveform,
                track_id=track_id,
                segment_start=float(change["segment_start"]),
                segment_end=float(change["segment_end"]),
                purpose="reconstruct_segment_mp3",
                storage=storage,
                job_id=job_id or track_id,
            ))
            merged = self._splice_segment(merged, tts_waveform, start_sample, end_sample)

        peak = merged.abs().max().item()
        if peak > 0.99:
            merged = merged * (0.99 / peak)

        out_path = save_as_mp3(
            merged,
            self.TARGET_SR,
            job_id=job_id,
            run_id=run_id,
            track_id=track_id,
            purpose="reconstruct_mp3",
        )
        duration = merged.shape[1] / self.TARGET_SR
        b2_key = storage.key("reconstructed", f"{job_id or track_id}.mp3")
        loop = asyncio.get_event_loop()
        try:
            audio_url = await loop.run_in_executor(
                None, storage.upload_file, out_path, b2_key, "audio/mpeg"
            )
        finally:
            drop_temp_standalone(out_path)

        return SynthesisResult(
            b2_key=b2_key,
            audio_url=audio_url,
            duration=round(duration, 3),
            segments=sorted(segment_results, key=lambda item: item.segment_start),
        )


    async def reconstruct_segments_batched(
        self,
        original_audio_path: str,
        track_id: str,
        changes: list,
        storage: B2Storage,
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

                original_text = str(change.get("original_text") or "").strip()
                ref_path = voice_reference_path if same_speaker else None
                ref_text = original_text or None
                reference_speaking_rate = None
                owns_reference = False
                if same_speaker and not ref_path:
                    (
                        ref_path,
                        ref_text,
                        reference_speaking_rate,
                    ) = await self._prepare_voice_reference(
                        original_waveform,
                        start_sample,
                        end_sample,
                        track_id=track_id,
                        original_text=original_text,
                    )
                    owns_reference = True
                try:
                    tts_bytes = await self._generate_segment_groups(
                        change["new_text"],
                        reference_audio_path=ref_path,
                        original_text=ref_text,
                        track_id=track_id,
                        seed=seed,
                    )
                finally:
                    if ref_path and owns_reference:
                        drop_temp_standalone(ref_path)

                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir=hear_temp_directory()) as tmp:
                    tmp.write(tts_bytes)
                    tts_path = tmp.name
                tts_waveform, tts_sr = torchaudio.load(tts_path)
                drop_temp_standalone(tts_path)
                if tts_sr != self.TARGET_SR:
                    tts_waveform = F_audio.resample(tts_waveform, tts_sr, self.TARGET_SR)

                duration_reference = merged[:, start_sample:end_sample]
                ref_segment = duration_reference
                ref_radius = int(2.0 * self.TARGET_SR)
                if ref_segment.shape[1] < int(self.TARGET_SR * 1.0):
                    ref_start = max(0, start_sample - ref_radius)
                    ref_end = min(merged.shape[1], end_sample + ref_radius)
                    ref_segment = merged[:, ref_start:ref_end]

                tts_waveform = await asyncio.to_thread(
                    TTSPostProcessor.process,
                    tts_waveform,
                    ref_segment,
                    self.TARGET_SR,
                    match_reference_pitch=same_speaker,
                )
                pacing_reference, pacing_text = self._pacing_reference(
                    original_waveform,
                    start_sample,
                    end_sample,
                    original_text=original_text,
                    reference_text=ref_text,
                )
                tts_waveform = self._time_stretch_to_match(
                    tts_waveform,
                    pacing_reference,
                    new_text=change["new_text"],
                    original_text=pacing_text,
                    source_speaking_rate=reference_speaking_rate,
                )
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
        b2_key = storage.key("reconstructed", f"{job_id or track_id}.mp3")
        loop = asyncio.get_event_loop()
        try:
            audio_url = await loop.run_in_executor(
                None, storage.upload_file, out_path, b2_key, "audio/mpeg"
            )
        finally:
            drop_temp_standalone(out_path)

        return SynthesisResult(b2_key=b2_key, audio_url=audio_url, duration=round(duration, 3), bucket_name=storage.bucket_name)

    async def generate_preview(
        self,
        original_audio_path: str,
        track_id: str,
        changes: list,
        storage: B2Storage,
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
        segment_results: list[SegmentAudioResult] = []

        for change in normalized:
            start_sample = int(float(change["segment_start"]) * self.TARGET_SR)
            end_sample = int(float(change["segment_end"]) * self.TARGET_SR)
            text = change["new_text"]

            if change.get("is_deletion"):
                segment_results.append(SegmentAudioResult(
                    segment_start=float(change["segment_start"]),
                    segment_end=float(change["segment_end"]),
                    is_deletion=True,
                ))
                continue

            original_text = str(change.get("original_text") or "").strip()
            ref_text = original_text or None
            reference_path = None
            reference_speaking_rate = None
            try:
                if same_speaker:
                    (
                        reference_path,
                        ref_text,
                        reference_speaking_rate,
                    ) = await self._prepare_voice_reference(
                        original_waveform,
                        start_sample,
                        end_sample,
                        track_id=track_id,
                        original_text=original_text,
                    )
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
            wf, sr = torchaudio.load(tts_path)
            drop_temp_standalone(tts_path)
            if sr != self.TARGET_SR:
                wf = F_audio.resample(wf, sr, self.TARGET_SR)

            duration_reference = original_waveform[:, start_sample:end_sample]
            ref_segment = duration_reference
            if ref_segment.shape[1] > 0:
                wf = await asyncio.to_thread(
                    TTSPostProcessor.process,
                    wf,
                    ref_segment,
                    self.TARGET_SR,
                    match_reference_pitch=same_speaker,
                )
                pacing_reference, pacing_text = self._pacing_reference(
                    original_waveform,
                    start_sample,
                    end_sample,
                    original_text=original_text,
                    reference_text=ref_text,
                )
                wf = self._time_stretch_to_match(
                    wf,
                    pacing_reference,
                    new_text=text,
                    original_text=pacing_text,
                    source_speaking_rate=reference_speaking_rate,
                )

            segment_results.append(await self._upload_segment_audio(
                wf,
                track_id=track_id,
                segment_start=float(change["segment_start"]),
                segment_end=float(change["segment_end"]),
                purpose="preview_segment_mp3",
                storage=storage,
                job_id=job_id or track_id,
            ))
            waveforms.append(wf)

        if not waveforms:
            combined = torch.zeros((1, 1), dtype=original_waveform.dtype)
        elif len(waveforms) == 1:
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
        b2_key = storage.key("previews", f"{job_id or track_id}.mp3")
        loop = asyncio.get_event_loop()
        try:
            audio_url = await loop.run_in_executor(
                None, storage.upload_file, out_path, b2_key, "audio/mpeg"
            )
        finally:
            drop_temp_standalone(out_path)

        return SynthesisResult(
            b2_key=b2_key,
            audio_url=audio_url,
            duration=round(duration, 3),
            segments=segment_results,
            bucket_name=storage.bucket_name,
        )

    async def _upload_segment_audio(
        self,
        waveform: torch.Tensor,
        *,
        track_id: str,
        segment_start: float,
        segment_end: float,
        purpose: str,
        storage: B2Storage,
        job_id: str,
    ) -> SegmentAudioResult:
        out_path = save_as_mp3(
            waveform,
            self.TARGET_SR,
            track_id=track_id,
            purpose=purpose,
        )
        b2_key = storage.key("segments", job_id, f"{os.urandom(8).hex()}.mp3")
        loop = asyncio.get_event_loop()
        try:
            audio_url = await loop.run_in_executor(
                None, storage.upload_file, out_path, b2_key, "audio/mpeg"
            )
        finally:
            drop_temp_standalone(out_path)
        return SegmentAudioResult(
            segment_start=segment_start,
            segment_end=segment_end,
            b2_key=b2_key,
            audio_url=audio_url,
            duration=round(waveform.shape[1] / self.TARGET_SR, 3),
            bucket_name=storage.bucket_name,
        )

    async def remove_segment(
        self,
        original_audio_path: str,
        track_id: str,
        segment_start: float,
        segment_end: float,
        storage: B2Storage,
        job_id: str,
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
        b2_key = storage.key("reconstructed", f"{job_id}.mp3")
        loop = asyncio.get_event_loop()
        try:
            audio_url = await loop.run_in_executor(
                None, storage.upload_file, out_path, b2_key, "audio/mpeg"
            )
        finally:
            drop_temp_standalone(out_path)

        return SynthesisResult(b2_key=b2_key, audio_url=audio_url, duration=round(duration, 3), bucket_name=storage.bucket_name)

    async def rebuild_track_audio(
        self,
        original_audio_path: str,
        edited_transcript: str,
        track_id: str,
        job_id: str,
        storage: B2Storage,
        original_transcript: str = "",
    ) -> SynthesisResult:
        original_waveform, orig_sr = torchaudio.load(original_audio_path)
        if orig_sr != self.TARGET_SR:
            original_waveform = F_audio.resample(original_waveform, orig_sr, self.TARGET_SR)

        start_sample, end_sample = self._detect_speech_bounds(
            original_waveform, self.TARGET_SR, original_transcript
        )
        seed = self._compute_seed(job_id, track_id)
        reference_path = None
        reference_text = None
        reference_speaking_rate = None
        try:
            (
                reference_path,
                reference_text,
                reference_speaking_rate,
            ) = await self._prepare_voice_reference(
                original_waveform,
                start_sample,
                end_sample,
                track_id=track_id,
                original_text=original_transcript,
            )
            rebuilt_bytes = await self._generate_segment_groups(
                edited_transcript,
                reference_audio_path=reference_path,
                original_text=reference_text,
                track_id=track_id,
                seed=seed,
            )
        finally:
            if reference_path:
                drop_temp_standalone(reference_path)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir=hear_temp_directory()) as tmp:
            tmp.write(rebuilt_bytes)
            rebuilt_path = tmp.name

        rebuilt_waveform, rebuilt_sr = torchaudio.load(rebuilt_path)
        drop_temp_standalone(rebuilt_path)
        if rebuilt_sr != self.TARGET_SR:
            rebuilt_waveform = F_audio.resample(rebuilt_waveform, rebuilt_sr, self.TARGET_SR)

        ref_segment = original_waveform[:, start_sample:end_sample]
        if ref_segment.shape[1] < int(self.TARGET_SR * 0.05):
            ref_radius = int(2.0 * self.TARGET_SR)
            ref_start = max(0, start_sample - ref_radius)
            ref_end = min(original_waveform.shape[1], end_sample + ref_radius)
            ref_segment = original_waveform[:, ref_start:ref_end]

        rebuilt_waveform = await asyncio.to_thread(
            TTSPostProcessor.process,
            rebuilt_waveform,
            ref_segment,
            self.TARGET_SR,
        )
        rebuilt_waveform = self._time_stretch_to_match(
            rebuilt_waveform,
            ref_segment,
            new_text=edited_transcript,
            original_text=original_transcript,
            source_speaking_rate=reference_speaking_rate,
        )

        merged = self._splice_segment(original_waveform, rebuilt_waveform, start_sample, end_sample)
        peak = merged.abs().max().item()
        if peak > 0.99:
            merged = merged * (0.99 / peak)

        out_path = save_as_mp3(merged, self.TARGET_SR, job_id=job_id, track_id=track_id, purpose="rebuilt_track_mp3")
        duration = merged.shape[1] / self.TARGET_SR
        b2_key = storage.key("reconstructed", f"{job_id}.mp3")
        loop = asyncio.get_event_loop()
        try:
            audio_url = await loop.run_in_executor(
                None, storage.upload_file, out_path, b2_key, "audio/mpeg"
            )
        finally:
            drop_temp_standalone(out_path)

        return SynthesisResult(b2_key=b2_key, audio_url=audio_url, duration=round(duration, 3), bucket_name=storage.bucket_name)

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

        refs = None
        if reference_audio_path:
            try:
                reference_bytes = await asyncio.to_thread(
                    self._read_file_bytes, reference_audio_path
                )
                refs = [{"audio": reference_bytes, "text": original_text or ""}]
                _recon_payload_logger.info(
                    "FISHSPEECH_REF | path='%s' size=%d text='%s'",
                    reference_audio_path,
                    len(reference_bytes),
                    (original_text or "")[:80],
                )
            except OSError as exc:
                logger.warning("Unable to read voice reference %s: %s", reference_audio_path, exc)
        return await self._model_client.generate_speech(
            text=processed_text,
            max_new_tokens=1024,
            references=refs,
            seed=seed,
        )

    async def _preprocess_for_s2(self, text: str) -> str:
        """Preserve the requested delivery and mark only explicit paragraph breaks."""
        paragraphs = re.split(r"\n\s*\n", text.strip())
        normalized = [re.sub(r"\s+", " ", paragraph).strip() for paragraph in paragraphs]
        return " [pause] ".join(paragraph for paragraph in normalized if paragraph)

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
        return ref_path

    def _reference_clip_bounds(
        self,
        waveform: torch.Tensor,
        start_sample: int,
        end_sample: int,
    ) -> tuple[int, int]:
        """Return a clean reference window that never contains the edited audio."""
        total_samples = waveform.shape[1]
        if total_samples <= 0:
            return 0, 0

        start_sample = max(0, min(start_sample, total_samples))
        end_sample = max(start_sample, min(end_sample, total_samples))
        target_samples = int(self.VOICE_REFERENCE_SECONDS * self.TARGET_SR)
        minimum_samples = int(self.MIN_VOICE_REFERENCE_SECONDS * self.TARGET_SR)
        guard_samples = int(self.VOICE_REFERENCE_GUARD_SECONDS * self.TARGET_SR)

        excluded_start = max(0, start_sample - guard_samples)
        excluded_end = min(total_samples, end_sample + guard_samples)
        left_samples = min(target_samples, excluded_start)
        right_samples = min(target_samples, total_samples - excluded_end)

        if max(left_samples, right_samples) < minimum_samples:
            return 0, 0

        # Prefer preceding speech when both sides offer the same amount. It is
        # normally the closest causal context and avoids crossing a speaker
        # change immediately after the edited interval.
        if left_samples >= right_samples:
            return excluded_start - left_samples, excluded_start
        return excluded_end, excluded_end + right_samples

    @classmethod
    def _speaking_rate_from_transcription(
        cls,
        transcription: dict,
    ) -> float | None:
        """Measure delivery rate across the complete aligned ASR speech span."""
        intervals: list[tuple[float, float]] = []
        units = 0

        for segment in transcription.get("segments") or []:
            segment_has_aligned_words = False
            for word in segment.get("words") or []:
                word_units = cls._speech_units(word.get("word"))
                try:
                    word_start = float(word.get("start"))
                    word_end = float(word.get("end"))
                except (TypeError, ValueError):
                    continue
                if (
                    word_units <= 0
                    or not np.isfinite(word_start)
                    or not np.isfinite(word_end)
                    or word_end <= word_start
                ):
                    continue
                intervals.append((word_start, word_end))
                units += word_units
                segment_has_aligned_words = True

            if segment_has_aligned_words:
                continue

            segment_units = cls._speech_units(segment.get("text"))
            try:
                segment_start = float(segment.get("start"))
                segment_end = float(segment.get("end"))
            except (TypeError, ValueError):
                continue
            if (
                segment_units > 0
                and np.isfinite(segment_start)
                and np.isfinite(segment_end)
                and segment_end > segment_start
            ):
                intervals.append((segment_start, segment_end))
                units += segment_units

        if units <= 0 or not intervals:
            return None

        # Source and generated audio must use the same clock. The full aligned
        # span includes natural internal pauses while excluding leading and
        # trailing audio outside the first and last recognized words.
        speech_start = min(start for start, _end in intervals)
        speech_end = max(end for _start, end in intervals)
        span_seconds = speech_end - speech_start
        if span_seconds <= 0.0:
            return None
        speaking_rate = units / span_seconds
        if not (
            np.isfinite(speaking_rate)
            and cls.MIN_PLAUSIBLE_SPEAKING_RATE
            <= speaking_rate
            <= cls.MAX_PLAUSIBLE_SPEAKING_RATE
        ):
            return None
        return float(speaking_rate)

    @classmethod
    def _transcription_text_for_window(
        cls,
        transcription: dict,
        window_start: float,
        window_end: float,
    ) -> str:
        """Return ASR text aligned to a clipped reference-audio window."""
        if window_end <= window_start:
            return ""

        timed_words: list[tuple[float, float, str, int]] = []
        segment_bounds: list[tuple[float, float, str]] = []
        for segment in transcription.get("segments") or []:
            try:
                segment_start = float(segment.get("start"))
                segment_end = float(segment.get("end"))
            except (TypeError, ValueError):
                segment_start = segment_end = 0.0
            segment_text = str(segment.get("text") or "").strip()
            if segment_text and segment_end > segment_start:
                segment_bounds.append((segment_start, segment_end, segment_text))

            for word in segment.get("words") or []:
                word_text = str(word.get("word") or "").strip()
                word_units = cls._speech_units(word_text)
                try:
                    word_start = float(word.get("start"))
                    word_end = float(word.get("end"))
                except (TypeError, ValueError):
                    continue
                if (
                    word_units <= 0
                    or not np.isfinite(word_start)
                    or not np.isfinite(word_end)
                    or word_end <= word_start
                ):
                    continue
                timed_words.append((word_start, word_end, word_text, word_units))

        all_bounds = [(start, end) for start, end, *_rest in timed_words]
        all_bounds.extend((start, end) for start, end, _text in segment_bounds)
        if all_bounds:
            speech_start = min(start for start, _end in all_bounds)
            speech_end = max(end for _start, end in all_bounds)
            if window_start <= speech_start and window_end >= speech_end:
                return str(transcription.get("transcript") or "").strip()

        selected_words = [
            word_text
            for word_start, word_end, word_text, word_units in timed_words
            if word_units <= 2
            and window_start <= (word_start + word_end) / 2 < window_end
        ]
        if selected_words:
            return " ".join(selected_words)

        # A provider may omit word timestamps. Only use whole segments that
        # fit inside the clip; partial segment text would mislabel Fish's
        # reference audio and can leak source words into the output.
        selected_segments = [
            segment_text
            for segment_start, segment_end, segment_text in segment_bounds
            if segment_start >= window_start and segment_end <= window_end
        ]
        return " ".join(selected_segments)

    @classmethod
    def _complete_word_reference_window(
        cls,
        transcription: dict,
        window_start: float,
        window_end: float,
    ) -> tuple[float, float, str] | None:
        """Align reference audio and text to the same complete ASR words."""
        if window_end <= window_start:
            return None

        selected: list[tuple[float, float, str]] = []
        for segment in transcription.get("segments") or []:
            for word in segment.get("words") or []:
                word_text = str(word.get("word") or "").strip()
                word_units = cls._speech_units(word_text)
                try:
                    word_start = float(word.get("start"))
                    word_end = float(word.get("end"))
                except (TypeError, ValueError):
                    continue
                if (
                    0 < word_units <= 2
                    and np.isfinite(word_start)
                    and np.isfinite(word_end)
                    and word_end > word_start
                    and word_start >= window_start
                    and word_end <= window_end
                ):
                    selected.append((word_start, word_end, word_text))

        if not selected:
            return None
        selected.sort(key=lambda item: (item[0], item[1]))
        return selected[0][0], selected[-1][1], " ".join(
            word_text for _start, _end, word_text in selected
        )

    async def _prepare_voice_reference(
        self,
        waveform: torch.Tensor,
        start_sample: int,
        end_sample: int,
        *,
        track_id: str | None,
        original_text: str | None = None,
    ) -> tuple[str | None, str, float | None]:
        """Export word-aligned speaker audio, transcript, and delivery rate."""
        total_samples = waveform.shape[1]
        edit_start = max(0, min(start_sample, total_samples))
        edit_end = max(edit_start, min(end_sample, total_samples))
        target_samples = int(self.VOICE_REFERENCE_SECONDS * self.TARGET_SR)
        minimum_samples = int(self.MIN_VOICE_REFERENCE_SECONDS * self.TARGET_SR)

        # A supplied source transcript makes the edited interval a trusted
        # speaker and pacing reference. Prefer it over adjacent audio, which
        # can cross a speaker change.
        use_edited_interval = (
            self._speech_units(original_text) > 0
            and edit_end - edit_start >= minimum_samples
        )
        if use_edited_interval:
            clip_samples = min(target_samples, edit_end - edit_start)
            ref_start = edit_start + (edit_end - edit_start - clip_samples) // 2
            ref_end = ref_start + clip_samples

            max_pacing_samples = int(
                self.MAX_PACING_REFERENCE_SECONDS * self.TARGET_SR
            )
            pacing_samples = min(max_pacing_samples, edit_end - edit_start)
            pacing_start = edit_start + (
                edit_end - edit_start - pacing_samples
            ) // 2
            pacing_end = pacing_start + pacing_samples
        else:
            ref_start, ref_end = self._reference_clip_bounds(
                waveform, edit_start, edit_end
            )
            pacing_start, pacing_end = ref_start, ref_end

        if ref_end <= ref_start:
            logger.warning(
                "No usable voice reference for track=%s",
                track_id,
            )
            return None, "", None

        reference_path = None
        try:
            if use_edited_interval:
                pacing_audio = (
                    waveform[:, pacing_start:pacing_end]
                    .detach()
                    .float()
                    .mean(dim=0)
                    .cpu()
                    .numpy()
                )
                audio_bytes = await asyncio.to_thread(
                    self._wav_bytes_from_audio,
                    pacing_audio,
                    self.TARGET_SR,
                )
            else:
                reference_path = self._export_reference_clip(
                    waveform,
                    ref_start,
                    ref_end,
                    track_id=track_id,
                )
                audio_bytes = await asyncio.to_thread(
                    self._read_file_bytes,
                    reference_path,
                )

            pacing_duration = (pacing_end - pacing_start) / self.TARGET_SR
            transcript = await self._transcriber.transcribe(
                audio_bytes,
                track_id=track_id,
                short_utterance=pacing_duration <= self.VOICE_REFERENCE_SECONDS,
            )
            speaking_rate = self._speaking_rate_from_transcription(transcript)
            if use_edited_interval:
                window_start = (ref_start - pacing_start) / self.TARGET_SR
                window_end = (ref_end - pacing_start) / self.TARGET_SR
                aligned_window = self._complete_word_reference_window(
                    transcript,
                    window_start,
                    window_end,
                )
                if aligned_window is not None:
                    aligned_start, aligned_end, reference_text = aligned_window
                    aligned_samples = round(
                        (aligned_end - aligned_start) * self.TARGET_SR
                    )
                    if aligned_samples >= minimum_samples:
                        ref_start = pacing_start + round(
                            aligned_start * self.TARGET_SR
                        )
                        ref_end = pacing_start + round(aligned_end * self.TARGET_SR)
                    else:
                        reference_text = ""
                else:
                    reference_text = self._transcription_text_for_window(
                        transcript,
                        window_start,
                        window_end,
                    )
            else:
                reference_text = str(transcript.get("transcript") or "").strip()

            if reference_text:
                if reference_path is None:
                    reference_path = self._export_reference_clip(
                        waveform,
                        ref_start,
                        ref_end,
                        track_id=track_id,
                    )
                return reference_path, reference_text, speaking_rate

            if speaking_rate is not None:
                logger.warning(
                    "Voice clone reference text was not aligned for track=%s; "
                    "using measured speaking rate without voice reference",
                    track_id,
                )
                if reference_path:
                    drop_temp_standalone(reference_path)
                return None, "", speaking_rate
            logger.warning("Voice reference transcription was empty for track=%s", track_id)
        except Exception as exc:
            logger.warning("Voice reference transcription failed for track=%s: %s", track_id, exc)

        # Never pair reference audio with missing or mismatched text. Fish can
        # otherwise treat unlabelled reference words as target content.
        if reference_path:
            drop_temp_standalone(reference_path)
        return None, "", None

    @staticmethod
    def _read_file_bytes(path: str) -> bytes:
        with open(path, "rb") as file_obj:
            return file_obj.read()

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

    def _pacing_reference(
        self,
        waveform: torch.Tensor,
        start_sample: int,
        end_sample: int,
        *,
        original_text: str | None,
        reference_text: str | None,
    ) -> tuple[torch.Tensor, str]:
        """Choose audio and aligned text from which to learn speaking rate."""
        total_samples = waveform.shape[1]
        start_sample = max(0, min(start_sample, total_samples))
        end_sample = max(start_sample, min(end_sample, total_samples))
        edited_segment = waveform[:, start_sample:end_sample]

        aligned_original = str(original_text or "").strip()
        if self._speech_units(aligned_original) > 0:
            return edited_segment, aligned_original

        clean_reference_text = str(reference_text or "").strip()
        if self._speech_units(clean_reference_text) > 0:
            ref_start, ref_end = self._reference_clip_bounds(
                waveform,
                start_sample,
                end_sample,
            )
            if ref_end > ref_start:
                return waveform[:, ref_start:ref_end], clean_reference_text

        return edited_segment, ""

    @staticmethod
    def _speech_units(text: str | None) -> int:
        """Count spoken word units while ignoring Fish Speech control tokens."""
        without_controls = re.sub(r"\[[^\]]+\]", " ", str(text or ""))
        return len(re.findall(r"[^\W_]+(?:['’][^\W_]+)?", without_controls, re.UNICODE))

    @classmethod
    def _active_speech_duration(
        cls,
        waveform: torch.Tensor,
        sr: int,
        *,
        allow_uniform_activity: bool = True,
    ) -> float:
        """Estimate voiced/activity time without counting long silent gaps.

        Uniform energy is accepted for clean generated speech, but callers can
        reject it for source recordings where stationary music/noise otherwise
        looks like speech across the entire clip.
        """
        if waveform.numel() == 0 or sr <= 0:
            return 0.0

        mono = waveform.detach().float().mean(dim=0).cpu().numpy()
        frame_samples = max(1, int(sr * cls.PACING_ACTIVITY_FRAME_MS / 1000))
        frame_count = len(mono) // frame_samples
        if frame_count < 2:
            peak = float(np.max(np.abs(mono))) if mono.size else 0.0
            return len(mono) / sr if peak >= 1e-5 else 0.0

        framed = mono[: frame_count * frame_samples].reshape(frame_count, frame_samples)
        rms = np.sqrt(np.mean(framed * framed, axis=1))
        low_energy = float(np.percentile(rms, 20))
        high_energy = float(np.percentile(rms, 95))
        if high_energy < 1e-5:
            return 0.0

        if low_energy / high_energy >= 0.65:
            if not allow_uniform_activity:
                return 0.0
            active = np.ones(frame_count, dtype=bool)
        else:
            threshold = max(
                1e-5,
                high_energy * 0.08,
                float(np.sqrt(max(low_energy, 0.0) * high_energy)),
            )
            active = rms > threshold
            # Bridge isolated 20 ms detector gaps inside otherwise continuous speech.
            active = np.convolve(active.astype(np.int8), np.ones(3, dtype=np.int8), mode="same") > 0

        return float(np.count_nonzero(active) * frame_samples / sr)

    def _time_stretch_to_match(
        self,
        tts_waveform: torch.Tensor,
        ref_waveform: torch.Tensor,
        *,
        new_text: str | None = None,
        original_text: str | None = None,
        source_speaking_rate: float | None = None,
    ) -> torch.Tensor:
        current_samples = tts_waveform.shape[1]
        current_dur = current_samples / self.TARGET_SR
        if current_samples <= 0:
            return tts_waveform
        source_units = self._speech_units(original_text)
        replacement_units = self._speech_units(new_text)
        if replacement_units <= 0:
            logger.info("Speech-rate match skipped because replacement text is unavailable")
            return tts_waveform

        source_rate = None
        source_measurement = "aligned-asr-span"
        try:
            candidate_rate = float(source_speaking_rate)
        except (TypeError, ValueError):
            candidate_rate = 0.0
        if (
            np.isfinite(candidate_rate)
            and self.MIN_PLAUSIBLE_SPEAKING_RATE
            <= candidate_rate
            <= self.MAX_PLAUSIBLE_SPEAKING_RATE
        ):
            source_rate = candidate_rate

        if source_rate is None:
            source_measurement = "aligned-source-span"
            if source_units <= 0:
                logger.info(
                    "Speech-rate match skipped because aligned source timing is unavailable"
                )
                return tts_waveform
            source_span_dur = ref_waveform.shape[1] / self.TARGET_SR
            if source_span_dur < 0.05:
                logger.warning(
                    "Speech-rate match skipped because source span is too short"
                )
                return tts_waveform
            source_rate = source_units / source_span_dur

        if current_dur < 0.05:
            logger.warning("Speech-rate match skipped because generated span is too short")
            return tts_waveform

        raw_tts_rate = replacement_units / current_dur
        if not (
            np.isfinite(source_rate)
            and np.isfinite(raw_tts_rate)
            and self.MIN_PLAUSIBLE_SPEAKING_RATE
            <= source_rate
            <= self.MAX_PLAUSIBLE_SPEAKING_RATE
            and self.MIN_PLAUSIBLE_SPEAKING_RATE
            <= raw_tts_rate
            <= self.MAX_PLAUSIBLE_SPEAKING_RATE
        ):
            logger.warning(
                "Speech-rate match skipped because source/raw rates are unreliable "
                "(source=%.3f raw_tts=%.3f)",
                source_rate,
                raw_tts_rate,
            )
            return tts_waveform

        requested_rate = source_rate / raw_tts_rate
        applied_rate = float(np.clip(
            requested_rate,
            self.MIN_SAFE_TEMPO_FACTOR,
            self.MAX_SAFE_TEMPO_FACTOR,
        ))
        target_samples = max(1, round(current_samples / applied_rate))
        target_dur = target_samples / self.TARGET_SR
        logger.info(
            "Speech-rate profile: source=%.2f words/s raw_tts=%.2f words/s "
            "requested=%.3f applied=%.3f measurement=%s duration=%.2fs->%.2fs",
            source_rate,
            raw_tts_rate,
            requested_rate,
            applied_rate,
            source_measurement,
            current_dur,
            target_dur,
        )
        _recon_payload_logger.info(
            "PACING_PROFILE | source=%.3f words/s | raw_tts=%.3f words/s "
            "| requested=%.4f | applied=%.4f | measurement=%s "
            "| duration=%.3fs->%.3fs",
            source_rate,
            raw_tts_rate,
            requested_rate,
            applied_rate,
            source_measurement,
            current_dur,
            target_dur,
        )
        if not np.isclose(requested_rate, applied_rate):
            residual_percent = (applied_rate / requested_rate - 1.0) * 100.0
            logger.warning(
                "Speech-rate correction clamped: requested=%.3f applied=%.3f "
                "residual=%+.1f%%",
                requested_rate,
                applied_rate,
                residual_percent,
            )
        if target_dur < self.MIN_PACING_TARGET_SECONDS:
            return tts_waveform
        if abs(applied_rate - 1.0) < 0.01:
            return tts_waveform
        try:
            sox_input = tts_waveform.detach().to(device="cpu", dtype=torch.float32)
            effects = [["tempo", "-s", f"{applied_rate:.8f}"]]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                stretched, stretched_sr = torchaudio.sox_effects.apply_effects_tensor(
                    sox_input,
                    self.TARGET_SR,
                    effects,
                )
            if stretched_sr != self.TARGET_SR:
                stretched = F_audio.resample(
                    stretched,
                    stretched_sr,
                    self.TARGET_SR,
                )
            stretched = stretched.to(
                device=tts_waveform.device,
                dtype=tts_waveform.dtype,
            )
            return self._fit_waveform_length(stretched, target_samples)
        except Exception as e:
            logger.warning(
                "Speech tempo matching failed: %s, preserving natural TTS speed",
                e,
            )
            return tts_waveform

    @staticmethod
    def _fit_waveform_length(waveform: torch.Tensor, target_samples: int) -> torch.Tensor:
        """Keep reconstruction timestamps stable even when time-stretching fails."""
        current_samples = waveform.shape[1]
        if current_samples >= target_samples:
            return waveform[:, :target_samples]
        return torch.nn.functional.pad(waveform, (0, target_samples - current_samples))

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
