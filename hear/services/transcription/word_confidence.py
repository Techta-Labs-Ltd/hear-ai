"""Per-word confidence from CTC log-probabilities (torch-free).

The Qwen forced aligner places words in time but measures nothing about them;
its placeholder score of 1.0 is dropped by the transcription service, which is
why `word_confidence_available` was false on every transcript. This module
turns an independent CTC model's frame posteriors into a real number per word:
the geometric-mean probability of the word's own letters along the best path
through the frames the aligner assigned to it. A word the model cannot spell
(digits, symbols) gets None, never a guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hear.services.reconstruction.verification import CtcAligner

# Padding around the aligner's word span; its stamps are tight and 20 ms coarse.
PAD_SECONDS = 0.12
MIN_SPAN_FRAMES = 4
# Never equal the aligner's placeholder; the service treats 1.0 as "not measured".
MAX_PROB = 0.999


@dataclass(frozen=True)
class Piece:
    """CTC log-probabilities for one stretch of a window."""

    start_seconds: float  # within the window
    log_probs: np.ndarray  # [frames, vocab]
    frame_seconds: float

    @property
    def end_seconds(self) -> float:
        return self.start_seconds + self.log_probs.shape[0] * self.frame_seconds


class WordConfidence:
    @staticmethod
    def pieces_for(
        window_seconds: float, piece_seconds: float, overlap_seconds: float
    ) -> list[tuple[float, float]]:
        """(start, end) stretches covering the window, overlapping so no word straddles a cut."""
        if window_seconds <= 0:
            return []
        if window_seconds <= piece_seconds:
            return [(0.0, window_seconds)]
        step = piece_seconds - overlap_seconds
        starts = np.arange(0.0, max(window_seconds - overlap_seconds, 0.0), step)
        return [(float(s), float(min(s + piece_seconds, window_seconds))) for s in starts]

    @staticmethod
    def score_word(
        piece: Piece, word: str, start: float, end: float, vocab: dict[str, int], blank: int
    ) -> float | None:
        """Probability of `word` across the frames the aligner gave it, or None."""
        letters = CtcAligner.normalize_words(word)
        cleaned = letters[0][1] if letters else ""
        if not cleaned:
            return None
        tokens = CtcAligner.tokens_for([cleaned], vocab)
        if tokens is None:
            return None
        first = int(math.floor((start - PAD_SECONDS - piece.start_seconds) / piece.frame_seconds))
        last = int(math.ceil((end + PAD_SECONDS - piece.start_seconds) / piece.frame_seconds))
        first = max(0, first)
        last = min(piece.log_probs.shape[0], max(last, first + MIN_SPAN_FRAMES))
        if last - first < MIN_SPAN_FRAMES:
            return None
        span = piece.log_probs[first:last]
        path = CtcAligner.viterbi(span, tokens, blank)
        if path is None:
            return None
        extended = [blank]
        for token in tokens:
            extended += [token, blank]
        ext = np.array(extended)
        emitted = [float(span[t, ext[s]]) for t, s in enumerate(path) if ext[s] != blank]
        if not emitted:
            return None
        return float(min(MAX_PROB, math.exp(sum(emitted) / len(emitted))))

    @staticmethod
    def score_words(
        pieces: list[Piece], words: list[dict], vocab: dict[str, int], blank: int
    ) -> list[float | None]:
        """One probability per word (times relative to the window), None when unmeasurable."""
        scores: list[float | None] = []
        for item in words:
            start = float(item.get("start", 0.0))
            end = float(item.get("end", start))
            piece = next(
                (
                    p
                    for p in pieces
                    if p.start_seconds <= start - PAD_SECONDS and end + PAD_SECONDS <= p.end_seconds
                ),
                None,
            )
            if piece is None:
                piece = next((p for p in pieces if p.start_seconds <= start < p.end_seconds), None)
            if piece is None:
                scores.append(None)
                continue
            scores.append(
                WordConfidence.score_word(
                    piece, str(item.get("word", "")), start, end, vocab, blank
                )
            )
        return scores
