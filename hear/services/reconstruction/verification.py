"""Torch-free verification logic: CTC alignment, word checking and splice planning.

The models live in `hear.inference.speech_verifier`; everything here works on
plain arrays so the renderer, its tests and the planner never import torch.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher

import numpy as np


@dataclass(frozen=True)
class WordSpan:
    word: str
    start_seconds: float
    end_seconds: float
    score: float


@dataclass(frozen=True)
class VerificationResult:
    """What the verifier measured for one generated take."""

    words: tuple[WordSpan, ...] | None  # forced alignment of the full generated text
    decoded: tuple[str, ...]  # greedy CTC transcript of the generated audio
    similarity: float | None  # cosine speaker similarity to the reference
    aligned: bool

    def as_dict(self) -> dict:
        return {
            "aligned": self.aligned,
            "decoded": list(self.decoded),
            "similarity": None if self.similarity is None else round(self.similarity, 4),
            "words": None if self.words is None else [asdict(w) for w in self.words],
        }


class CtcAligner:
    """Viterbi forced alignment and greedy decoding over CTC log-probabilities."""

    WORD_DELIMITER = "|"
    ALLOWED = re.compile(r"[^A-Z']+")

    @staticmethod
    def normalize_words(text: str) -> list[tuple[str, str]]:
        """(original, normalised) per word; a normalised word may be empty (e.g. digits)."""
        words = []
        for original in text.split():
            cleaned = CtcAligner.ALLOWED.sub("", original.upper()).strip("'")
            words.append((original, cleaned))
        return words

    @staticmethod
    def tokens_for(words: list[str], vocab: dict[str, int]) -> list[int] | None:
        """Character ids with word delimiters, or None when a character is unknown."""
        tokens: list[int] = []
        delimiter = vocab.get(CtcAligner.WORD_DELIMITER)
        if delimiter is None:
            return None
        for index, word in enumerate(words):
            if index:
                tokens.append(delimiter)
            for char in word:
                if char not in vocab:
                    return None
                tokens.append(vocab[char])
        return tokens

    @staticmethod
    def viterbi(log_probs: np.ndarray, tokens: list[int], blank: int) -> np.ndarray | None:
        """Best path state per frame over the blank-extended token sequence, or None."""
        frames = log_probs.shape[0]
        extended = [blank]
        for token in tokens:
            extended += [token, blank]
        states = len(extended)
        ext = np.array(extended)
        # Transitions: stay, advance one, or skip a blank between distinct tokens.
        skip_ok = np.zeros(states, dtype=bool)
        skip_ok[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
        repeats = sum(1 for a, b in zip(tokens, tokens[1:], strict=False) if a == b)
        if not tokens or frames < len(tokens) + repeats:
            return None
        neg = -np.inf
        alpha = np.full(states, neg)
        alpha[0] = log_probs[0, ext[0]]
        if states > 1:
            alpha[1] = log_probs[0, ext[1]]
        back = np.zeros((frames, states), dtype=np.int8)
        for t in range(1, frames):
            stay = alpha
            one = np.concatenate(([neg], alpha[:-1]))
            two = np.concatenate(([neg, neg], alpha[:-2]))
            two = np.where(skip_ok, two, neg)
            stacked = np.stack([stay, one, two])
            choice = np.argmax(stacked, axis=0)
            back[t] = choice
            alpha = stacked[choice, np.arange(states)] + log_probs[t, ext]
        last = states - 1
        if states > 1 and alpha[states - 2] > alpha[last]:
            last = states - 2
        if not np.isfinite(alpha[last]):
            return None
        path = np.zeros(frames, dtype=np.int64)
        state = last
        for t in range(frames - 1, -1, -1):
            path[t] = state
            state -= int(back[t, state])
        return path

    @staticmethod
    def word_spans(
        path: np.ndarray,
        log_probs: np.ndarray,
        tokens: list[int],
        words: list[str],
        blank: int,
        delimiter: int,
        frame_seconds: float,
    ) -> list[WordSpan]:
        """Collapse the frame path into one timed, scored span per word."""
        extended = [blank]
        for token in tokens:
            extended += [token, blank]
        ext = np.array(extended)
        spans: list[WordSpan] = []
        word_index = 0
        start = None
        end = None
        scores: list[float] = []

        def flush():
            nonlocal start, end, scores, word_index
            if start is not None:
                spans.append(
                    WordSpan(
                        words[word_index],
                        round(start * frame_seconds, 4),
                        round((end + 1) * frame_seconds, 4),
                        round(float(np.mean(scores)), 4),
                    )
                )
            word_index += 1
            start, end, scores = None, None, []

        previous = -1
        for t, state in enumerate(path):
            token = ext[state]
            if token == blank:
                continue
            if token == delimiter:
                if state != previous:
                    flush()
                previous = state
                continue
            if start is None:
                start = t
            end = t
            scores.append(float(log_probs[t, token]))
            previous = state
        flush()
        return spans

    @staticmethod
    def greedy_decode(log_probs: np.ndarray, id_to_char: dict[int, str], blank: int) -> list[str]:
        best = np.argmax(log_probs, axis=1)
        chars = []
        previous = -1
        for token in best:
            if token != previous and token != blank:
                chars.append(id_to_char.get(int(token), ""))
            previous = token
        text = "".join(chars).replace(CtcAligner.WORD_DELIMITER, " ")
        return text.split()


class WordMatcher:
    """Does the generated audio say the requested words?"""

    MIN_CHAR_RATIO = 0.75

    @staticmethod
    def similar(a: str, b: str) -> bool:
        return a == b or SequenceMatcher(None, a, b).ratio() >= WordMatcher.MIN_CHAR_RATIO

    @staticmethod
    def match(expected: list[str], decoded: list[str]) -> list[bool]:
        """Per expected word, whether the decoded transcript contains it in order."""
        matched = [False] * len(expected)
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, expected, decoded).get_opcodes():
            if tag == "equal":
                for i in range(i1, i2):
                    matched[i] = True
            elif tag == "replace":
                for offset, i in enumerate(range(i1, i2)):
                    j = j1 + offset
                    if j < j2 and WordMatcher.similar(expected[i], decoded[j]):
                        matched[i] = True
        return matched

    @staticmethod
    def passes(matched: list[bool]) -> bool:
        if not matched:
            return False
        ratio = sum(matched) / len(matched)
        return ratio >= (1.0 if len(matched) <= 5 else 0.9)


@dataclass(frozen=True)
class SplicePlan:
    mode: str  # "words" or "whole_segment"
    cut_start_frame: int
    cut_end_frame: int
    reason: str | None = None
    new_word_spans: tuple[WordSpan, ...] = field(default_factory=tuple)

    def as_dict(self, rate: int) -> dict:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "cut_start_seconds": round(self.cut_start_frame / rate, 4),
            "cut_end_seconds": round(self.cut_end_frame / rate, 4),
            "new_words": [asdict(w) for w in self.new_word_spans],
        }


class SplicePlanner:
    """Cut only the new words out of a generated sentence, at the quietest gap frames."""

    RATE = 48000
    PROBE = 240  # 5 ms energy probe

    @staticmethod
    def quietest_frame(audio: np.ndarray, start: int, end: int) -> int:
        start = max(0, start)
        end = min(len(audio), end)
        if end - start < 2 * SplicePlanner.PROBE:
            return (start + end) // 2
        count = (end - start) // SplicePlanner.PROBE
        block = audio[start : start + count * SplicePlanner.PROBE].astype("float64")
        energy = np.mean(block.reshape(count, SplicePlanner.PROBE) ** 2, axis=1)
        return start + int(np.argmin(energy)) * SplicePlanner.PROBE + SplicePlanner.PROBE // 2

    @staticmethod
    def plan(
        audio: np.ndarray,
        words: tuple[WordSpan, ...] | None,
        before: int,
        new: int,
        after: int,
    ) -> SplicePlan:
        """`before`/`new`/`after` are word counts of the generated text in order."""
        total = len(audio)
        if before == 0 and after == 0:
            return SplicePlan("whole_segment", 0, total, "no_context")
        if words is None or len(words) != before + new + after or new == 0:
            return SplicePlan("whole_segment", 0, total, "alignment_unavailable")
        rate = SplicePlanner.RATE
        first = words[before]
        last = words[before + new - 1]
        new_start = round(first.start_seconds * rate)
        new_end = round(last.end_seconds * rate)
        if before:
            previous_end = round(words[before - 1].end_seconds * rate)
            cut_start = SplicePlanner.quietest_frame(audio, previous_end, new_start)
        else:
            cut_start = 0
        if after:
            next_start = round(words[before + new].start_seconds * rate)
            cut_end = SplicePlanner.quietest_frame(audio, new_end, next_start)
        else:
            cut_end = total
        cut_start = max(0, min(cut_start, total))
        cut_end = max(cut_start + 1, min(cut_end, total))
        return SplicePlan("words", cut_start, cut_end, None, tuple(words[before : before + new]))
