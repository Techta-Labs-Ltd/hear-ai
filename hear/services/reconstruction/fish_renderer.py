"""Bounded, job-scoped Fish TTS generation and lossless timeline assembly."""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import asdict, dataclass, field
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
from hear.services.reconstruction.acoustic_match import AcousticMatcher
from hear.services.reconstruction.verification import (
    CtcAligner,
    SplicePlan,
    SplicePlanner,
    VerificationResult,
    WordMatcher,
)


@dataclass
class RenderedSpeechEdit:
    edit: SpeechEdit
    audio: Path | None
    delivery: Path | None
    frames: int
    reference_sha256: str | None
    gain_db: float = 0
    seed: int | None = None
    voice_reference: dict | None = None
    acoustic: dict | None = None
    chunks: int = 0
    takes: list[dict] = field(default_factory=list)
    selected_take: int | None = None
    similarity: float | None = None
    words_verified: bool | None = None
    splice: dict | None = None

    def row(self, start: int, end: int, output_start: int, output_end: int) -> dict:
        rate = FishReconstructionRenderer.RATE
        return {
            "segment_start": self.edit.segment_start,
            "segment_end": self.edit.segment_end,
            "source_start_frame": start,
            "source_end_frame": end,
            "output_start_frame": output_start,
            "output_end_frame": output_end,
            "output_start_seconds": output_start / rate,
            "output_end_seconds": output_end / rate,
            "duration": self.frames / rate,
            "duration_delta_seconds": (self.frames - (end - start)) / rate,
            "new_text": self.edit.new_text,
            "is_deletion": self.edit.is_deletion,
            "reference_sha256": self.reference_sha256,
            "segment_gain_db": self.gain_db,
            "seed": self.seed,
            "text_chunks": self.chunks,
            "voice_reference": self.voice_reference,
            "acoustic_match": self.acoustic,
            "crossfade_frames": FishReconstructionRenderer.crossfade(self, start, end),
            "takes": self.takes,
            "selected_take": self.selected_take,
            "voice_similarity": self.similarity,
            "words_verified": self.words_verified,
            "splice": self.splice,
        }


@dataclass
class GeneratedTake:
    take: int
    seed: int
    audio: Path
    frames: int
    verification: VerificationResult | None
    words_verified: bool | None
    word_match_ratio: float | None

    @property
    def similarity(self) -> float | None:
        return None if self.verification is None else self.verification.similarity

    @property
    def similarity_reliable(self) -> bool:
        """x-vectors need a couple of seconds of speech; a one-word take cannot be judged."""
        return (
            self.similarity is not None
            and self.frames
            >= FishReconstructionRenderer.MIN_SIMILARITY_SECONDS * FishReconstructionRenderer.RATE
        )

    @property
    def acceptable(self) -> bool:
        return self.words_verified is not False

    def summary(self) -> dict:
        return {
            "take": self.take,
            "seed": self.seed,
            "duration": self.frames / FishReconstructionRenderer.RATE,
            "similarity": None if self.similarity is None else round(self.similarity, 4),
            "similarity_reliable": self.similarity_reliable,
            "words_verified": self.words_verified,
            "word_match_ratio": self.word_match_ratio,
            "decoded": None if self.verification is None else " ".join(self.verification.decoded),
        }


@dataclass(frozen=True)
class ReferenceQuality:
    duration_seconds: float
    channel: int
    silence_ratio: float
    clipped_ratio: float
    chars_per_second: float
    peak: float
    warnings: tuple[str, ...] = field(default_factory=tuple)


class FishReconstructionRenderer:
    RATE = 48000
    BLOCK = 48000
    # Keep each Fish request well inside its 1,024 new-token budget whatever the
    # codec frame rate turns out to be; roughly twelve seconds of English.
    MAX_CHUNK_CHARACTERS = 200
    MAX_GENERATED_SECONDS = 3600
    # 20 ms blends at the edges of a generated region; 5 ms only hid clicks.
    CROSSFADE_FRAMES = 960
    # Reference acceptance: mostly-silent, clipped or text-mismatched prompts give
    # Fish the wrong idea of the voice, so they fail instead of degrading quietly.
    MAX_REFERENCE_SILENCE_RATIO = 0.6
    MAX_REFERENCE_CLIPPED_RATIO = 0.005
    REFERENCE_CHARS_PER_SECOND = (2.0, 30.0)
    WARN_REFERENCE_SILENCE_RATIO = 0.4
    WARN_REFERENCE_SECONDS = 8.0

    # A take this far above the floor is kept without rendering the remaining ones.
    EARLY_STOP_MARGIN = 0.1
    TAKE_SEED_STRIDE = 16
    # Below this take length speaker similarity only ranks takes; it cannot fail them.
    MIN_SIMILARITY_SECONDS = 1.5
    # A take may decode into a word or two more than asked (CTC splits words); a
    # take that says a whole extra phrase is not the requested edit.
    EXTRA_WORD_TOLERANCE = 1
    # A number or symbol is one token in the text but several spoken words
    # ("2026" is "twenty twenty-six", "1,500" is "one thousand five hundred").
    SPOKEN_WORDS_PER_UNSPELLABLE_TOKEN = 5

    def __init__(
        self,
        fish,
        native: NativeExecutor,
        *,
        verifier=None,
        takes: int = 1,
        min_similarity: float = 0.75,
    ):
        self.fish = fish
        self.native = native
        self.verifier = verifier
        self.takes = max(1, min(4, takes)) if verifier is not None else 1
        self.min_similarity = min_similarity
        self.runner = CancellableProcessRunner()
        self.dsp = ProfileDspService(self.runner)
        self.mastering = AudioMasteringService(self.runner)

    @classmethod
    def text_chunks(cls, text: str) -> list[str]:
        # Preserve tokens and punctuation: never paraphrase or add style tags.
        limit = cls.MAX_CHUNK_CHARACTERS
        words = text.strip().split()
        chunks = []
        current: list[str] = []
        length = 0
        for word in words:
            if len(word) > limit:
                raise ValueError("reconstruction_text_token_too_long")
            if current and length + len(word) + 1 > limit:
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

    @staticmethod
    def edit_seed(base: int | None, job_identity: str, index: int) -> int:
        """One seed per edit so every chunk of it is the same take of the voice.

        Takes of one edit use consecutive seeds, so edits are spaced by the stride.
        """
        stride = FishReconstructionRenderer.TAKE_SEED_STRIDE
        if base is not None:
            return (base + index * stride) & 0x7FFFFFFF
        digest = hashlib.sha256(f"{job_identity}:{index}".encode()).digest()[:4]
        return (int.from_bytes(digest, "big") * stride) & 0x7FFFFFFF

    @staticmethod
    def crossfade(item: RenderedSpeechEdit, start: int, end: int) -> int:
        if item.audio is None:
            return 0
        return min(
            FishReconstructionRenderer.CROSSFADE_FRAMES, item.frames // 4, (end - start) // 2
        )

    @classmethod
    def reference_quality(
        cls, voice: np.ndarray, reference: VoiceReference, channel: int
    ) -> ReferenceQuality:
        duration = len(voice) / cls.RATE
        rms = AcousticMatcher.frame_rms(voice)
        peak_rms = float(np.max(rms)) if len(rms) else 0.0
        silent = rms < max(1e-4, peak_rms * 0.01)
        silence_ratio = float(np.mean(silent)) if len(rms) else 1.0
        clipped_ratio = float(np.mean(np.abs(voice) >= 0.99)) if len(voice) else 0.0
        chars = len(" ".join(reference.text.split()))
        chars_per_second = chars / duration if duration else 0.0
        if silence_ratio > cls.MAX_REFERENCE_SILENCE_RATIO:
            raise ValueError("voice_reference_mostly_silent")
        if clipped_ratio > cls.MAX_REFERENCE_CLIPPED_RATIO:
            raise ValueError("voice_reference_clipped")
        low, high = cls.REFERENCE_CHARS_PER_SECOND
        if not low <= chars_per_second <= high:
            raise ValueError("voice_reference_text_length_mismatch")
        warnings = []
        if silence_ratio > cls.WARN_REFERENCE_SILENCE_RATIO:
            warnings.append("reference_has_long_pauses")
        if duration < cls.WARN_REFERENCE_SECONDS:
            warnings.append("reference_shorter_than_recommended")
        return ReferenceQuality(
            duration_seconds=round(duration, 3),
            channel=channel,
            silence_ratio=round(silence_ratio, 3),
            clipped_ratio=round(clipped_ratio, 4),
            chars_per_second=round(chars_per_second, 2),
            peak=round(float(np.max(np.abs(voice))) if len(voice) else 0.0, 4),
            warnings=tuple(warnings),
        )

    @classmethod
    def reference_bytes(
        cls, source: Path, reference: VoiceReference, guard: ResourceGuard
    ) -> tuple[bytes, ReferenceQuality]:
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
        voice = values[:, channel]
        quality = cls.reference_quality(voice, reference, channel)
        stream = io.BytesIO()
        peak = float(np.max(np.abs(voice)))
        if peak > 0.95:
            voice = voice * (0.95 / peak)
        sf.write(stream, voice, cls.RATE, format="WAV", subtype="PCM_16")
        return stream.getvalue(), quality

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

    @staticmethod
    def mono(path: Path) -> np.ndarray:
        with sf.SoundFile(path) as audio:
            return audio.read(dtype="float32", always_2d=True)[:, 0]

    @staticmethod
    def decode_reference(payload: bytes) -> np.ndarray:
        values, _ = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
        return values[:, 0]

    @classmethod
    def judge_words(
        cls, edit: SpeechEdit, verification: VerificationResult
    ) -> tuple[bool | None, float | None]:
        """Whether the decoded transcript contains the new words; None when unverifiable."""
        from_text = cls.normalized_words(edit.generation_text)
        before, new, _ = edit.context_word_counts
        decoded = list(verification.decoded)
        matched = WordMatcher.match(from_text, decoded)
        checked = [matched[i] for i in range(before, before + new) if from_text[i]]
        if not checked:
            return None, None
        ratio = round(sum(checked) / len(checked), 3)
        spelled = [word for word in from_text if word]
        unspellable = len(from_text) - len(spelled)
        extra = len(decoded) - len(spelled)
        allowed = max(cls.EXTRA_WORD_TOLERANCE, len(spelled) // 10) + (
            unspellable * cls.SPOKEN_WORDS_PER_UNSPELLABLE_TOKEN
        )
        if extra > allowed:
            return False, ratio
        return WordMatcher.passes(checked), ratio

    @staticmethod
    def normalized_words(text: str) -> list[str]:
        return [cleaned for _, cleaned in CtcAligner.normalize_words(text)]

    @classmethod
    def cut(cls, source: Path, target: Path, plan: SplicePlan) -> int:
        with sf.SoundFile(source) as audio:
            audio.seek(plan.cut_start_frame)
            values = audio.read(
                plan.cut_end_frame - plan.cut_start_frame, dtype="float32", always_2d=True
            )
            rate = audio.samplerate
        with target.open("xb") as stream:
            sf.write(stream, values, rate, format="WAV", subtype="FLOAT")
        return len(values)

    @classmethod
    def preview_timeline(cls, edits: list[RenderedSpeechEdit]) -> list[dict]:
        """The output map a full assembly would produce, without writing it."""
        timeline = []
        cursor = 0
        output_cursor = 0
        for item in edits:
            start = round(item.edit.segment_start * cls.RATE)
            end = round(item.edit.segment_end * cls.RATE)
            output_cursor += start - cursor
            output_start = output_cursor
            output_cursor += item.frames
            timeline.append(item.row(start, end, output_start, output_cursor))
            cursor = end
        return timeline

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
                    if item.audio is not None:
                        edge = cls.crossfade(item, start, end)
                        with sf.SoundFile(item.audio) as replacement:
                            position = 0
                            while True:
                                guard.check_scratch()
                                block = replacement.read(cls.BLOCK, dtype="float32", always_2d=True)
                                if not len(block):
                                    break
                                # Blend within the generated segment, never consume
                                # neighbouring source frames or alter its timestamp map.
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
                    timeline.append(item.row(start, end, output_start, output_cursor))
                    cursor = end
                original.seek(cursor)
                while True:
                    guard.check_scratch()
                    data = original.read(cls.BLOCK, dtype="float32", always_2d=True)
                    if not len(data):
                        break
                    out.write(data)
        return timeline

    async def generate_take(
        self, raw: Path, text: str, ref: dict | None, options, seed: int, channels: int, guard
    ) -> int:
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
            for chunk in self.text_chunks(text):
                guard.check()
                payload = await self.fish.generate_speech(
                    text=chunk,
                    references=[{"audio": ref["audio"], "text": ref["text"]}] if ref else None,
                    language=options.language,
                    seed=seed,
                    max_new_tokens=1024,
                )
                count += await self.native.run(
                    self.append_generated, payload, output, channels, guard
                )
        return count

    def choose_take(self, candidates: list[GeneratedTake]) -> GeneratedTake:
        """Best verified take by similarity; fail loudly rather than deliver a bad one."""
        acceptable = [c for c in candidates if c.acceptable]
        if self.verifier is not None and not acceptable:
            raise ValueError("generated_words_unverified")
        pool = acceptable or candidates
        best = max(pool, key=lambda c: (-1.0 if c.similarity is None else c.similarity, -c.take))
        if (
            best.similarity is not None
            and best.similarity_reliable
            and best.similarity < self.min_similarity
        ):
            raise ValueError("voice_match_below_threshold")
        return best

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
        # May queue for host scratch disk, so keep it off the event loop (heartbeats).
        await self.native.run(
            guard.preflight_pcm, frames, channels, copies=3, output_bytes=frames * channels * 4
        )
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
        # One explicit reference serves every edit; only an edit long enough to be
        # a reference itself may fall back to its own interval.
        references: dict[int, dict] = {}
        chunks = {}
        shared = None
        if options.same_speaker and options.reference is not None:
            payload, quality = await self.native.run(
                self.reference_bytes, prepared, options.reference, guard
            )
            shared = {"audio": payload, "text": options.reference.text, "quality": quality}
        for index, edit in enumerate(edits):
            if edit.is_deletion:
                continue
            chunks[index] = self.text_chunks(edit.new_text)
            if not options.same_speaker:
                continue
            if shared is not None:
                references[index] = shared
                continue
            if not edit.can_self_reference:
                raise ValueError("same_speaker_edit_requires_explicit_reference")
            reference = VoiceReference(
                start_seconds=edit.segment_start,
                end_seconds=edit.segment_end,
                text=edit.original_text,
            )
            payload, quality = await self.native.run(
                self.reference_bytes, prepared, reference, guard
            )
            references[index] = {"audio": payload, "text": reference.text, "quality": quality}
        rendered = []
        total_generated = 0
        for index, edit in enumerate(edits):
            guard.check()
            if edit.is_deletion:
                rendered.append(RenderedSpeechEdit(edit, None, None, 0, None))
                continue
            ref = references.get(index)
            reference_audio = self.decode_reference(ref["audio"]) if ref else None
            base_seed = self.edit_seed(options.seed, job_identity, index)
            takes = options.takes or self.takes
            # Without the verifier the context cannot be aligned, so only the new
            # words are generated and the whole take replaces the interval.
            text = edit.generation_text if self.verifier is not None else edit.new_text
            before, new, after = (
                edit.context_word_counts
                if self.verifier is not None
                else (0, len(edit.new_text.split()), 0)
            )
            candidates: list[GeneratedTake] = []
            for take in range(takes):
                guard.check()
                seed = (base_seed + take) & 0x7FFFFFFF
                raw = guard.workspace / f"fish-segment-{index:03d}-take{take}.wav"
                count = await self.generate_take(raw, text, ref, options, seed, channels, guard)
                if count + total_generated > self.RATE * self.MAX_GENERATED_SECONDS:
                    raise ValueError("generated_speech_budget_exceeded")
                verification = None
                words_verified = ratio = None
                if self.verifier is not None:
                    verification = await self.native.run(
                        self.verifier.verify, self.mono(raw), text, reference_audio
                    )
                    words_verified, ratio = self.judge_words(edit, verification)
                candidate = GeneratedTake(
                    take, seed, raw, count, verification, words_verified, ratio
                )
                candidates.append(candidate)
                if candidate.acceptable and (
                    candidate.similarity is None
                    or (
                        candidate.similarity_reliable
                        and candidate.similarity >= self.min_similarity + self.EARLY_STOP_MARGIN
                    )
                ):
                    break
            best = self.choose_take(candidates)
            for candidate in candidates:
                if candidate is not best:
                    candidate.audio.unlink()
            plan = SplicePlan("whole_segment", 0, best.frames, "verifier_unavailable")
            if self.verifier is not None:
                plan = SplicePlanner.plan(
                    self.mono(best.audio),
                    best.verification.words if best.verification else None,
                    before,
                    new,
                    after,
                )
            if plan.mode != "words" and (before or after):
                # The sentence could not be aligned, so its context words must not
                # be inserted: regenerate the new words alone with the chosen seed.
                best.audio.unlink()
                raw = guard.workspace / f"fish-segment-{index:03d}-plain.wav"
                count = await self.generate_take(
                    raw, edit.new_text, ref, options, best.seed, channels, guard
                )
                best = GeneratedTake(
                    best.take,
                    best.seed,
                    raw,
                    count,
                    None,
                    best.words_verified,
                    best.word_match_ratio,
                )
                plan = SplicePlan("whole_segment", 0, count, "context_alignment_unavailable")
            total_generated += best.frames
            raw = best.audio
            if plan.mode == "words":
                trimmed = guard.workspace / f"fish-segment-{index:03d}-words.wav"
                frames_kept = await self.native.run(self.cut, raw, trimmed, plan)
                raw.unlink()
                raw = trimmed
            else:
                frames_kept = best.frames
            segment_audio = guard.workspace / f"segment-{index:03d}.wav"
            acoustic = await self.native.run(
                AcousticMatcher.match_segment,
                prepared,
                raw,
                segment_audio,
                round(edit.segment_start * self.RATE),
                round(edit.segment_end * self.RATE),
                guard,
            )
            raw.unlink()
            mastered = await self.native.run(
                self.mastering.master, segment_audio, MasteringSettings(), guard
            )
            delivery = guard.workspace / f"segment-{index:03d}.mp3"
            mastered.delivery.rename(delivery)
            rendered.append(
                RenderedSpeechEdit(
                    edit,
                    segment_audio,
                    delivery,
                    frames_kept,
                    hashlib.sha256(ref["audio"]).hexdigest() if ref else None,
                    mastered.gain_db,
                    seed=best.seed,
                    voice_reference=asdict(ref["quality"]) if ref else None,
                    acoustic=acoustic,
                    chunks=len(self.text_chunks(text)),
                    takes=[candidate.summary() for candidate in candidates],
                    selected_take=best.take,
                    similarity=None if best.similarity is None else round(best.similarity, 4),
                    words_verified=best.words_verified,
                    splice=plan.as_dict(self.RATE),
                )
            )
            if progress:
                progress(20 + 55 * (index + 1) / len(edits))
        expected = frames + sum(
            item.frames
            - (
                round(item.edit.segment_end * self.RATE)
                - round(item.edit.segment_start * self.RATE)
            )
            for item in rendered
        )
        result = {
            "segments": rendered,
            "engine": "fish_speech_s2_pro",
            "sample_rate": self.RATE,
            "source_frames": frames,
            "channels": channels,
            "timeline_policy": "natural_tts_duration_with_explicit_source_output_map",
            "generated_stereo_policy": "dual_mono_inside_edited_regions"
            if channels == 2
            else "mono",
            "word_accuracy_verified": self.verifier is not None
            and all(item.words_verified for item in rendered if item.audio is not None),
            "voice_verification": "wav2vec2_ctc_words_wavlm_sv_similarity"
            if self.verifier is not None
            else None,
            "requires_approval": True,
        }
        if operation == "preview":
            # A preview is the generated segments only: the caller auditions them in
            # place and confirms with the same seed, so the full track is rendered once.
            result.update(
                {
                    "delivery": None,
                    "timeline": self.preview_timeline(rendered),
                    "output_frames": expected,
                    "duration": expected / self.RATE,
                    "duration_delta_seconds": (expected - frames) / self.RATE,
                    "delivery_measurement": None,
                    "final_gain_db": 0.0,
                    "preview": True,
                }
            )
            return result
        target = guard.workspace / "reconstructed_float.wav"
        timeline = await self.native.run(self.assemble, prepared, target, rendered, guard)
        await self.native.run(
            AudioMasteringService.scan,
            target,
            guard,
            rate=self.RATE,
            channels=channels,
            frames=expected,
        )
        mastered = await self.native.run(self.mastering.master, target, MasteringSettings(), guard)
        result.update(
            {
                "delivery": mastered.delivery,
                "timeline": timeline,
                "output_frames": mastered.frames,
                "duration": mastered.frames / self.RATE,
                "duration_delta_seconds": (mastered.frames - frames) / self.RATE,
                "delivery_measurement": asdict(mastered.delivery_measurement),
                "final_gain_db": mastered.gain_db,
                "preview": False,
            }
        )
        return result
