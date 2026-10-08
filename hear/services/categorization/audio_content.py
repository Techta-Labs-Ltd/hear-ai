"""What a recording sounds like (music, speech, silence), judged from the audio itself.

The transcript cannot tell an instrumental from a silent file: both have no words.
Clips spread across the recording are scored by an AudioSet tagger; a recording
that is mostly music is a song whether or not anyone sings words the ASR keeps.
"""

from __future__ import annotations

import logging
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import numpy as np

from hear.execution.native import NativeExecutor

logger = logging.getLogger(__name__)


class AudioTagger(Protocol):
    def tag_sync(self, clips: list[np.ndarray], labels: tuple[str, ...]) -> list[dict[str, float]]: ...


class AudioContentService:
    LABELS = ("Music", "Singing", "Speech", "Silence")
    SAMPLE_RATE = 16000
    CLIP_SECONDS = 10.0
    # Short recordings are tiled end to end; longer ones get MAX_CLIPS spread evenly.
    MAX_CLIPS = 12
    # Clips quieter than this carry no evidence either way (the tagger calls
    # near-silence faintly musical).
    AUDIBLE_DBFS = -50.0
    # Measured on AST AudioSet: songs and instrumentals score Music 0.37-0.95 with
    # Speech <= 0.02; speech scores Speech 0.72-0.96 with Music <= 0.01; speech over
    # a music bed scores Speech 0.77 and Music 0.56 and stays speech.
    MUSIC_SCORE = 0.3
    SINGING_SCORE = 0.3
    SPEECH_SCORE = 0.4
    # Most of the recording, so a talk with a jingle at each end is not a song.
    MUSIC_SHARE = 0.7
    CLIP_TIMEOUT_SECONDS = 120

    def __init__(self, tagger: AudioTagger | None, native: NativeExecutor) -> None:
        self._tagger = tagger
        self._native = native

    async def analyse(self, path: str, duration_seconds: float, *, has_speech: bool) -> dict:
        if self._tagger is None or duration_seconds <= 0:
            return self.result_for("unknown")
        try:
            clips = await self._native.run(self._clips, path, duration_seconds)
            audible = [clip for clip in clips if self._dbfs(clip) > self.AUDIBLE_DBFS]
            if not audible:
                return self.result_for("silence", clips=len(clips))
            scores = await self._native.run(self._tagger.tag_sync, audible, self.LABELS)
        except Exception as exc:
            # Music detection enriches the result; it never fails the job.
            logger.warning("[AUDIO_CONTENT] analysis failed (%s)", exc)
            return self.result_for("unknown")
        music = sum(1 for score in scores if self._is_music(score))
        speech = sum(1 for score in scores if self._is_speech(score))
        music_share = music / len(scores)
        is_music = music_share >= self.MUSIC_SHARE or (not has_speech and music > 0 and speech == 0)
        if is_music:
            kind = "music"
        elif music and speech:
            kind = "speech_with_music"
        elif speech or has_speech:
            kind = "speech"
        else:
            kind = "other"
        return self.result_for(
            kind,
            clips=len(clips),
            audible_clips=len(scores),
            music_share=music_share,
            speech_share=speech / len(scores),
            singing=any(score.get("Singing", 0.0) >= self.SINGING_SCORE for score in scores),
        )

    @staticmethod
    def result_for(
        kind: str,
        *,
        clips: int = 0,
        audible_clips: int = 0,
        music_share: float = 0.0,
        speech_share: float = 0.0,
        singing: bool = False,
    ) -> dict:
        return {
            "kind": kind,
            "music": kind == "music",
            "music_share": round(music_share, 3),
            "speech_share": round(speech_share, 3),
            "singing": singing,
            "clips": clips,
            "audible_clips": audible_clips,
        }

    def _is_music(self, score: dict[str, float]) -> bool:
        # A clip with clear speech is talk, whatever plays under it; sung words are music.
        return (
            score.get("Music", 0.0) >= self.MUSIC_SCORE
            and score.get("Speech", 0.0) < self.SPEECH_SCORE
        ) or score.get("Singing", 0.0) >= self.SINGING_SCORE

    def _is_speech(self, score: dict[str, float]) -> bool:
        return not self._is_music(score) and score.get("Speech", 0.0) >= self.SPEECH_SCORE

    @staticmethod
    def _dbfs(clip: np.ndarray) -> float:
        if clip.size == 0:
            return -120.0
        rms = float(np.sqrt(np.mean(np.square(clip, dtype=np.float64))))
        return 20.0 * math.log10(max(rms, 1e-10))

    @classmethod
    def _offsets(cls, duration_seconds: float) -> tuple[list[float], float]:
        length = min(cls.CLIP_SECONDS, duration_seconds)
        count = min(cls.MAX_CLIPS, max(1, math.ceil(duration_seconds / cls.CLIP_SECONDS)))
        return [(duration_seconds - length) * (i + 0.5) / count for i in range(count)], length

    @classmethod
    def _clips(cls, path: str, duration_seconds: float) -> list[np.ndarray]:
        offsets, length = cls._offsets(duration_seconds)
        with ThreadPoolExecutor(max_workers=min(len(offsets), 6)) as pool:
            clips = list(pool.map(lambda offset: cls._clip(path, offset, length), offsets))
        return [clip for clip in clips if clip.size >= cls.SAMPLE_RATE]

    @classmethod
    def _clip(cls, path: str, offset: float, length: float) -> np.ndarray:
        """One mono 16 kHz clip via ffmpeg, which reads every format and skips bad frames."""
        try:
            completed = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error",
                 *(["-ss", f"{offset:.3f}"] if offset > 0 else []), "-i", path,
                 "-t", f"{length:.3f}", "-map", "0:a:0", "-vn", "-ac", "1",
                 "-ar", str(cls.SAMPLE_RATE), "-f", "f32le", "-"],
                capture_output=True, check=True, timeout=cls.CLIP_TIMEOUT_SECONDS,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return np.zeros(0, dtype=np.float32)
        return np.frombuffer(completed.stdout, dtype=np.float32)
