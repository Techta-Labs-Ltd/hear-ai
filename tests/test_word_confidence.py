"""Real per-word confidence replaces the aligner's placeholder."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from hear.services.transcription.service import TranscriptionService
from hear.services.transcription.word_confidence import MAX_PROB, Piece, WordConfidence

VOCAB = {"<pad>": 0, "A": 1, "B": 2, "|": 3, "C": 4}
BLANK = 0
FRAME = 0.02


def log_probs_for(sequence, confident=0.95):
    size = len(VOCAB)
    rows = []
    for token in sequence:
        row = np.full(size, (1 - confident) / (size - 1))
        row[token] = confident
        rows.append(np.log(row))
    return np.array(rows)


def test_pieces_overlap_so_no_word_straddles_a_cut():
    assert WordConfidence.pieces_for(10.0, 30.0, 2.0) == [(0.0, 10.0)]
    pieces = WordConfidence.pieces_for(70.0, 30.0, 2.0)
    assert pieces[0] == (0.0, 30.0) and pieces[1] == (28.0, 58.0) and pieces[-1][1] == 70.0
    assert WordConfidence.pieces_for(0.0, 30.0, 2.0) == []


def test_word_score_is_the_geometric_mean_of_its_letters():
    # 20 frames: silence, then "AB" clearly, then silence.
    frames = [0] * 5 + [1, 1, 0, 2, 2] + [0] * 10
    piece = Piece(0.0, log_probs_for(frames, confident=0.9), FRAME)
    score = WordConfidence.score_word(piece, "ab,", 0.10, 0.20, VOCAB, BLANK)
    assert score is not None and 0.85 < score <= MAX_PROB
    # The same span asked for the wrong word scores far lower.
    wrong = WordConfidence.score_word(piece, "cc", 0.10, 0.20, VOCAB, BLANK)
    assert wrong is not None and wrong < 0.1


def test_unspellable_or_unplaceable_words_get_none():
    piece = Piece(0.0, log_probs_for([0] * 20), FRAME)
    assert WordConfidence.score_word(piece, "2026", 0.1, 0.2, VOCAB, BLANK) is None
    assert WordConfidence.score_word(piece, "abz", 0.1, 0.2, VOCAB, BLANK) is None  # Z not in vocab
    assert WordConfidence.score_words(
        [piece], [{"word": "ab", "start": 5.0, "end": 5.2}], VOCAB, BLANK
    ) == [None]


def test_scores_never_reach_the_placeholder_value():
    piece = Piece(0.0, log_probs_for([1, 1, 1, 1], confident=0.9999), FRAME)
    score = WordConfidence.score_word(piece, "a", 0.0, 0.08, VOCAB, BLANK)
    assert score == MAX_PROB


class FakeScorer:
    source = "fake_ctc"

    def __init__(self):
        self.calls = []

    def score_window(self, samples, words):
        self.calls.append((len(samples), [w["word"] for w in words]))
        return [0.95 if w["word"] != "2026" else None for w in words]


@pytest.mark.anyio
async def test_service_replaces_placeholder_scores_and_reports_availability(tmp_path):
    path = tmp_path / "source.wav"
    sf.write(path, np.ones((32000, 1), dtype=np.float32) * 0.01, 16000)

    async def transcribe_window(audio, batch_size, language):
        return {
            "segments": [
                {
                    "text": "hello 2026",
                    "start": 0,
                    "end": 1.0,
                    "words": [
                        {"word": "hello", "start": 0.1, "end": 0.5, "score": 1.0},
                        {"word": "2026", "start": 0.6, "end": 0.9, "score": 1.0},
                    ],
                }
            ]
        }

    scorer = FakeScorer()
    service = TranscriptionService(
        SimpleNamespace(transcribe_window=transcribe_window), chunk_seconds=1, scorer=scorer
    )
    result = await service.transcribe_file(str(path))
    assert len(scorer.calls) == 2 and scorer.calls[0] == (16000, ["hello", "2026"])
    words = result["segments"][0]["words"]
    assert words[0]["prob"] == 0.95 and words[1]["prob"] is None
    assert result["word_confidence_available"] is True and result["confidence"] == 0.95
    assert result["word_confidence_source"] == "fake_ctc"
    assert result["performance"]["word_confidence_seconds"] >= 0


@pytest.mark.anyio
async def test_without_a_scorer_nothing_is_faked(tmp_path):
    path = tmp_path / "source.wav"
    sf.write(path, np.ones((16000, 1), dtype=np.float32) * 0.01, 16000)

    async def transcribe_window(audio, batch_size, language):
        return {
            "segments": [
                {
                    "text": "hello",
                    "start": 0,
                    "end": 1.0,
                    "words": [{"word": "hello", "start": 0.1, "end": 0.5, "score": 1.0}],
                }
            ]
        }

    result = await TranscriptionService(
        SimpleNamespace(transcribe_window=transcribe_window), chunk_seconds=1
    ).transcribe_file(str(path))
    assert result["word_confidence_available"] is False and result["word_confidence_source"] is None
    assert result["segments"][0]["words"][0]["prob"] is None


def test_settings_default_on():
    from hear.config import RuntimeSettings

    base = {"HEAR_IMAGE_REVISION": "x", "HEAR_WORKER_ROLE": "pipeline"}
    assert RuntimeSettings.from_environment(base).word_confidence_enabled is True
    assert (
        RuntimeSettings.from_environment(
            {**base, "HEAR_WORD_CONFIDENCE_ENABLED": "false"}
        ).word_confidence_enabled
        is False
    )
    assert asyncio  # keep import for readers of the async tests above
