"""Bounded, job-scoped Fish TTS generation and lossless timeline assembly."""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from hear.contracts.reconstruction import ReconstructionOptions, SpeechEdit, VoiceReference
from hear.execution.native import NativeExecutor
from hear.runtime.cleaner.resource_guard import ResourceGuard
from hear.runtime.cleaner.subprocesses import CancellableProcessRunner
from hear.services.magic_clean.mastering import AudioMasteringService, MasteringSettings
from hear.services.magic_clean.profile_dsp import ProfileDspService


@dataclass
class RenderedSpeechEdit:
    edit: SpeechEdit
    master: Path | None
    delivery: Path | None
    frames: int
    reference_sha256: str | None
    gain_db: float = 0


class FishReconstructionRenderer:
    RATE = 48000
    BLOCK = 48000
    MAX_CHUNK_CHARACTERS = 400
    MAX_GENERATED_SECONDS = 3600

    def __init__(self, fish, native: NativeExecutor):
        self.fish = fish
        self.native = native
        self.runner = CancellableProcessRunner()
        self.dsp = ProfileDspService(self.runner)
        self.mastering = AudioMasteringService(self.runner)

    @staticmethod
    def text_chunks(text: str) -> list[str]:
        # Preserve tokens and punctuation: never paraphrase or add style tags.
        words = text.strip().split()
        chunks = []
        current: list[str] = []
        length = 0
        for word in words:
            if len(word) > 400:
                raise ValueError("reconstruction_text_token_too_long")
            if current and length + len(word) + 1 > 400:
                chunks.append(" ".join(current))
                current = []
                length = 0
            current.append(word)
            length += len(word) + 1
        if current:
            chunks.append(" ".join(current))
        if not chunks:
            raise ValueError("new_text_required")
        return chunks

    @classmethod
    def reference_bytes(
        cls, source: Path, reference: VoiceReference, guard: ResourceGuard
    ) -> bytes:
        with sf.SoundFile(source) as audio:
            start = round(reference.start_seconds * cls.RATE)
            count = round((reference.end_seconds - reference.start_seconds) * cls.RATE)
            if start + count > audio.frames:
                raise ValueError("voice_reference_exceeds_source")
            guard.check()
            audio.seek(start)
            values = audio.read(count, dtype="float32", always_2d=True)
        # Choose the strongest original channel instead of anti-phase downmix.
        energy = np.mean(values.astype("float64") ** 2, axis=0)
        if reference.channel is not None and reference.channel >= values.shape[1]:
            raise ValueError("reference_channel_exceeds_source")
        if reference.channel is None and values.shape[1] == 2 and min(energy) > max(energy) * 0.1:
            correlation = float(
                np.dot(values[:, 0].astype("float64"), values[:, 1].astype("float64"))
                / max(
                    np.sqrt(
                        np.sum(values[:, 0].astype("float64") ** 2)
                        * np.sum(values[:, 1].astype("float64") ** 2)
                    ),
                    1e-15,
                )
            )
            if abs(correlation) < 0.8:
                raise ValueError("independent_stereo_voices_require_reference_channel")
        channel = reference.channel if reference.channel is not None else int(np.argmax(energy))
        if float(energy[channel]) < 1e-8:
            raise ValueError("voice_reference_is_silent")
        stream = io.BytesIO()
        voice = values[:, channel]
        peak = float(np.max(np.abs(voice)))
        if peak > 0.95:
            voice = voice * (0.95 / peak)
        sf.write(stream, voice, cls.RATE, format="WAV", subtype="PCM_16")
        return stream.getvalue()

    @classmethod
    def append_generated(cls, payload: bytes, output, channels: int, guard: ResourceGuard) -> int:
        guard.check()
        if not isinstance(payload, bytes) or len(payload) > 25_000_000:
            raise ValueError("invalid_fish_audio_response")
        with sf.SoundFile(io.BytesIO(payload)) as audio:
            if (
                audio.channels != 1
                or not 8000 <= audio.samplerate <= 96000
                or not 0 < audio.frames <= audio.samplerate * 120
            ):
                raise ValueError("invalid_fish_audio_layout_or_duration")
            values = audio.read(dtype="float32", always_2d=True)
            rate = audio.samplerate
        if not np.isfinite(values).all():
            raise ValueError("nonfinite_fish_audio")
        factor = math.gcd(rate, cls.RATE)
        values = resample_poly(values, cls.RATE // factor, rate // factor, axis=0)
        if channels == 2:
            values = np.repeat(values, 2, axis=1)
        output.write(values)
        return len(values)

    @classmethod
    def assemble(
        cls, source: Path, target: Path, edits: list[RenderedSpeechEdit], guard: ResourceGuard
    ) -> list[dict]:
        timeline = []
        with sf.SoundFile(source) as original, target.open("xb") as stream:
            with sf.SoundFile(
                stream,
                "w",
                format="WAV",
                samplerate=cls.RATE,
                channels=original.channels,
                subtype="FLOAT",
            ) as out:
                cursor = 0
                output_cursor = 0
                for item in edits:
                    start = round(item.edit.segment_start * cls.RATE)
                    end = round(item.edit.segment_end * cls.RATE)
                    original.seek(cursor)
                    remaining = start - cursor
                    while remaining:
                        guard.check_scratch()
                        data = original.read(
                            min(cls.BLOCK, remaining), dtype="float32", always_2d=True
                        )
                        if not len(data):
                            raise ValueError("truncated_reconstruction_source")
                        out.write(data)
                        remaining -= len(data)
                        output_cursor += len(data)
                    output_start = output_cursor
                    if item.master is not None:
                        with sf.SoundFile(item.master) as replacement:
                            position = 0
                            while True:
                                guard.check_scratch()
                                block = replacement.read(cls.BLOCK, dtype="float32", always_2d=True)
                                if not len(block):
                                    break
                                # Blend within the generated segment, never consume
                                # neighbouring source frames or alter its timestamp map.
                                edge = min(240, item.frames // 4, (end - start) // 2)
                                if edge and position == 0:
                                    original.seek(start)
                                    old = original.read(edge, dtype="float32", always_2d=True)
                                    w = np.linspace(0, 1, edge, dtype="float32")[:, None]
                                    block[:edge] = old * (1 - w) + block[:edge] * w
                                if edge and position + len(block) == item.frames:
                                    original.seek(end - edge)
                                    old = original.read(edge, dtype="float32", always_2d=True)
                                    w = np.linspace(1, 0, edge, dtype="float32")[:, None]
                                    block[-edge:] = old * (1 - w) + block[-edge:] * w
                                out.write(block)
                                position += len(block)
                                output_cursor += len(block)
                    timeline.append(
                        {
                            "segment_start": item.edit.segment_start,
                            "segment_end": item.edit.segment_end,
                            "source_start_frame": start,
                            "source_end_frame": end,
                            "output_start_frame": output_start,
                            "output_end_frame": output_cursor,
                            "output_start_seconds": output_start / cls.RATE,
                            "output_end_seconds": output_cursor / cls.RATE,
                            "duration": item.frames / cls.RATE,
                            "duration_delta_seconds": (item.frames - (end - start)) / cls.RATE,
                            "new_text": item.edit.new_text,
                            "is_deletion": item.edit.is_deletion,
                            "reference_sha256": item.reference_sha256,
                            "segment_gain_db": item.gain_db,
                        }
                    )
                    cursor = end
                original.seek(cursor)
                while True:
                    guard.check_scratch()
                    data = original.read(cls.BLOCK, dtype="float32", always_2d=True)
                    if not len(data):
                        break
                    out.write(data)
        return timeline

    async def render(
        self,
        source: Path,
        operation: str,
        options: ReconstructionOptions,
        guard: ResourceGuard,
        job_identity: str,
        progress=None,
    ) -> dict[str, Any]:
        prepared = guard.workspace / "reconstruction_source.wav"
        await self.native.run(self.dsp.render, source, prepared, [], guard, decode=True)
        frames, channels = await self.native.run(self.dsp.validate, prepared, guard)
        guard.preflight_pcm(frames, channels, copies=3, output_bytes=frames * channels * 4)
        if options.reference and round(options.reference.end_seconds * self.RATE) > frames:
            raise ValueError("voice_reference_exceeds_source")
        if operation == "rebuild":
            edits = [
                SpeechEdit(
                    segment_start=0,
                    segment_end=frames / self.RATE,
                    new_text=options.edited_transcript or "",
                )
            ]
        elif operation == "remove_segments":
            if options.segment_start is None or options.segment_end is None:
                raise ValueError("deletion_interval_required")
            edits = [
                SpeechEdit(
                    segment_start=options.segment_start,
                    segment_end=options.segment_end,
                    is_deletion=True,
                )
            ]
        else:
            edits = sorted(options.changes, key=lambda x: x.segment_start)
        if any(round(e.segment_end * self.RATE) > frames for e in edits):
            raise ValueError("reconstruction_interval_exceeds_source")
        if (
            sum(
                round((e.segment_end - e.segment_start) * self.RATE) for e in edits if e.is_deletion
            )
            >= frames
        ):
            raise ValueError("reconstruction_would_remove_all_audio")
        # Validate every reference/text before invoking Fish for the first edit.
        references = {}
        chunks = {}
        for index, edit in enumerate(edits):
            if edit.is_deletion:
                continue
            chunks[index] = self.text_chunks(edit.new_text)
            if options.same_speaker:
                reference = options.reference or VoiceReference(
                    start_seconds=edit.segment_start,
                    end_seconds=edit.segment_end,
                    text=edit.original_text,
                )
                payload = await self.native.run(self.reference_bytes, prepared, reference, guard)
                references[index] = [{"audio": payload, "text": reference.text}]
        rendered = []
        total_generated = 0
        for index, edit in enumerate(edits):
            guard.check()
            if edit.is_deletion:
                rendered.append(RenderedSpeechEdit(edit, None, None, 0, None))
                continue
            raw = guard.workspace / f"fish-segment-{index:03d}.wav"
            count = 0
            with (
                raw.open("xb") as stream,
                sf.SoundFile(
                    stream,
                    "w",
                    format="WAV",
                    samplerate=self.RATE,
                    channels=channels,
                    subtype="FLOAT",
                ) as output,
            ):
                for part, text in enumerate(chunks[index]):
                    guard.check()
                    seed = (
                        int.from_bytes(
                            hashlib.sha256(f"{job_identity}:{index}:{part}".encode()).digest()[:4],
                            "big",
                        )
                        & 0x7FFFFFFF
                    )
                    payload = await self.fish.generate_speech(
                        text=text,
                        references=references.get(index),
                        language=options.language,
                        seed=seed,
                        max_new_tokens=1024,
                    )
                    count += await self.native.run(
                        self.append_generated, payload, output, channels, guard
                    )
                    if count + total_generated > self.RATE * self.MAX_GENERATED_SECONDS:
                        raise ValueError("generated_speech_budget_exceeded")
            total_generated += count
            mastered = await self.native.run(self.mastering.master, raw, MasteringSettings(), guard)
            master = guard.workspace / f"segment-{index:03d}.flac"
            delivery = guard.workspace / f"segment-{index:03d}.mp3"
            mastered.master.rename(master)
            mastered.delivery.rename(delivery)
            ref = references.get(index)
            rendered.append(
                RenderedSpeechEdit(
                    edit,
                    master,
                    delivery,
                    count,
                    hashlib.sha256(ref[0]["audio"]).hexdigest() if ref else None,
                    mastered.gain_db,
                )
            )
            raw.unlink()
            if progress:
                progress(20 + 55 * (index + 1) / len(edits))
        target = guard.workspace / "reconstructed_float.wav"
        timeline = await self.native.run(self.assemble, prepared, target, rendered, guard)
        expected = frames + sum(
            item.frames
            - (
                round(item.edit.segment_end * self.RATE)
                - round(item.edit.segment_start * self.RATE)
            )
            for item in rendered
        )
        await self.native.run(
            AudioMasteringService.scan,
            target,
            guard,
            rate=self.RATE,
            channels=channels,
            frames=expected,
        )
        mastered = await self.native.run(self.mastering.master, target, MasteringSettings(), guard)
        return {
            "master": mastered.master,
            "delivery": mastered.delivery,
            "segments": rendered,
            "timeline": timeline,
            "engine": "fish_speech_s2_pro",
            "sample_rate": self.RATE,
            "source_frames": frames,
            "output_frames": mastered.frames,
            "channels": channels,
            "duration": mastered.frames / self.RATE,
            "duration_delta_seconds": (mastered.frames - frames) / self.RATE,
            "timeline_policy": "natural_tts_duration_with_explicit_source_output_map",
            "generated_stereo_policy": "dual_mono_inside_edited_regions"
            if channels == 2
            else "mono",
            "master_measurement": asdict(mastered.master_measurement),
            "delivery_measurement": asdict(mastered.delivery_measurement),
            "final_gain_db": mastered.gain_db,
            "word_accuracy_verified": False,
            "requires_approval": True,
        }
