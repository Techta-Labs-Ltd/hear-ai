"""Word verification, best-of-N take selection and word-level splicing."""

import asyncio
import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from hear.execution.native import NativeExecutor
from hear.services.reconstruction.fish_renderer import FishReconstructionRenderer
from hear.services.reconstruction.verification import (
    CtcAligner,
    SplicePlanner,
    VerificationResult,
    WordMatcher,
    WordSpan,
)
from hear.workflows.fish_reconstruction import FishReconstructionWorkflow
from tests.test_fish_tts_contract import (
    REFERENCE,
    FileSource,
    MemoryStorage,
    SyntheticSpeechDouble,
    request,
    source_file,
)

RATE = 48000
VOCAB = {"<pad>": 0, "A": 1, "B": 2, "|": 3, "C": 4}
BLANK = 0


def log_probs_for(sequence, confident=0.95):
    """One frame per listed token id with the given confidence."""
    size = len(VOCAB)
    rows = []
    for token in sequence:
        row = np.full(size, (1 - confident) / (size - 1))
        row[token] = confident
        rows.append(np.log(row))
    return np.array(rows)


def test_viterbi_aligns_and_times_words():
    frames = [1, 1, 0, 2, 3, 3, 4, 0]  # "AB" | "C"
    log_probs = log_probs_for(frames)
    tokens = CtcAligner.tokens_for(["AB", "C"], VOCAB)
    path = CtcAligner.viterbi(log_probs, tokens, BLANK)
    spans = CtcAligner.word_spans(path, log_probs, tokens, ["AB", "C"], BLANK, VOCAB["|"], 0.02)
    assert [s.word for s in spans] == ["AB", "C"]
    assert spans[0].start_seconds == 0 and spans[0].end_seconds == pytest.approx(0.08)
    assert spans[1].start_seconds == pytest.approx(0.12) and spans[1].end_seconds == pytest.approx(
        0.14
    )
    assert all(s.score > np.log(0.9) for s in spans)


def test_viterbi_handles_repeated_letters_and_impossible_lengths():
    assert CtcAligner.viterbi(log_probs_for([1, 1]), [1, 1], BLANK) is None  # needs a blank between
    assert CtcAligner.viterbi(log_probs_for([1, 0, 1]), [1, 1], BLANK) is not None
    assert CtcAligner.tokens_for(["A1"], VOCAB) is None


def test_greedy_decode_collapses_repeats_and_blanks():
    words = CtcAligner.greedy_decode(
        log_probs_for([1, 1, 0, 2, 3, 3, 4, 0]), {v: k for k, v in VOCAB.items()}, BLANK
    )
    assert words == ["AB", "C"]


def test_normalize_words_keeps_apostrophes_and_marks_digits_unalignable():
    assert CtcAligner.normalize_words("Don't stop, 2026!") == [
        ("Don't", "DON'T"),
        ("stop,", "STOP"),
        ("2026!", ""),
    ]


def test_word_matcher_tolerates_ctc_spelling_but_not_missing_words():
    expected = ["WE", "SHIP", "TOMORROW"]
    assert WordMatcher.match(expected, ["WE", "SHIP", "TOMOROW"]) == [True, True, True]
    assert WordMatcher.match(expected, ["WE", "TOMORROW"]) == [True, False, True]
    assert WordMatcher.passes([True, True, True]) and not WordMatcher.passes([True, False, True])
    assert WordMatcher.passes([True] * 9 + [False]) and not WordMatcher.passes(
        [True] * 7 + [False] * 3
    )


def test_splice_planner_cuts_in_the_quietest_gap_around_the_new_words():
    audio = np.ones(RATE, dtype="float32") * 0.2
    audio[RATE // 4 - 2400 : RATE // 4] = 0.001  # 50 ms dip before the new word
    words = (
        WordSpan("BEFORE", 0.0, 0.20, -0.1),
        WordSpan("NEW", 0.26, 0.50, -0.1),
        WordSpan("AFTER", 0.52, 1.0, -0.1),
    )
    plan = SplicePlanner.plan(audio, words, 1, 1, 1)
    assert plan.mode == "words"
    assert RATE // 4 - 2400 <= plan.cut_start_frame <= RATE // 4
    assert round(0.50 * RATE) <= plan.cut_end_frame <= round(0.52 * RATE)
    assert SplicePlanner.plan(audio, words, 0, 3, 0).mode == "whole_segment"
    assert SplicePlanner.plan(audio, None, 1, 1, 1).reason == "alignment_unavailable"


class FakeVerifier:
    """Scripted verifier: equal word spans over the take, similarity per call."""

    def __init__(self, similarities=(0.9,), decoded=None, aligned=True):
        self.similarities = list(similarities)
        self.decoded = decoded
        self.aligned = aligned
        self.calls = []

    def verify(self, audio, text, reference):
        self.calls.append((len(audio), text, None if reference is None else len(reference)))
        words = [cleaned for _, cleaned in CtcAligner.normalize_words(text)]
        duration = len(audio) / RATE
        spans = None
        if self.aligned:
            step = duration / len(words)
            spans = tuple(
                WordSpan(word, round(i * step, 4), round((i + 1) * step, 4), -0.05)
                for i, word in enumerate(words)
            )
        similarity = self.similarities[min(len(self.calls) - 1, len(self.similarities) - 1)]
        decoded = tuple(words if self.decoded is None else self.decoded)
        return VerificationResult(
            spans, decoded, None if reference is None else similarity, spans is not None
        )


def edit(**extra):
    return {
        "same_speaker": True,
        "reference": REFERENCE,
        "changes": [
            {
                "segment_start": 6,
                "segment_end": 6.5,
                "new_text": "tomorrow",
                "original_text": "today",
                "context_before": "we ship it",
                "context_after": "at noon",
            }
        ],
        **extra,
    }


class LongSpeechDouble(SyntheticSpeechDouble):
    """Two seconds per chunk: long enough for speaker similarity to be enforced."""

    async def generate_speech(self, **kwargs):
        self.calls.append(kwargs)
        t = np.arange(RATE * 2) / RATE
        wave = (0.06 * np.sin(2 * np.pi * 330 * t)).astype("float32")
        buffer = io.BytesIO()
        sf.write(buffer, wave, RATE, format="WAV", subtype="FLOAT")
        return buffer.getvalue()


def run(
    tmp_path,
    options,
    verifier,
    operation="replace_segments",
    takes=3,
    min_similarity=0.75,
    fish=None,
):
    path, _ = source_file(tmp_path, seconds=8)
    fish = fish or LongSpeechDouble()
    storage = MemoryStorage()

    async def go():
        native = NativeExecutor("verify-test")
        try:
            workflow = FishReconstructionWorkflow(
                FishReconstructionRenderer(
                    fish, native, verifier=verifier, takes=takes, min_similarity=min_similarity
                ),
                FileSource(path),
                storage,
                native,
                workspace_root=tmp_path / "jobs",
            )
            events = [e async for e in workflow.stream(request(path, operation, options))]
            return events[-1].data["outcome"]
        finally:
            await native.close()

    return asyncio.run(go()), fish, verifier


def test_only_the_new_words_are_spliced_into_the_interval(tmp_path):
    outcome, fish, verifier = run(tmp_path, edit(seed=100), FakeVerifier([0.9]))
    result = outcome["result"]["reconstructed_audio"]
    row = result["segments"][0]
    assert fish.calls[0]["text"] == "we ship it tomorrow at noon" and len(fish.calls) == 1
    assert (
        verifier.calls[0][1] == "we ship it tomorrow at noon" and verifier.calls[0][2] == 5.5 * RATE
    )
    # Six words over the 2 s synthetic take: the new word is the fourth 1/6th.
    assert row["splice"]["mode"] == "words" and row["duration"] == pytest.approx(2 / 6, abs=0.002)
    assert row["splice"]["new_words"][0]["word"] == "TOMORROW"
    assert row["words_verified"] is True and row["voice_similarity"] == 0.9
    assert row["selected_take"] == 0 and row["seed"] == 100 and len(row["takes"]) == 1
    assert result["word_accuracy_verified"] is True
    assert abs(result["output_frames"] - (8 * RATE - RATE // 2 + 16000)) <= 4
    assert result["voice_verification"] == "wav2vec2_ctc_words_wavlm_sv_similarity"


def test_best_take_is_kept_and_rendering_stops_once_a_take_is_clearly_good(tmp_path):
    outcome, fish, _ = run(tmp_path, edit(seed=100), FakeVerifier([0.70, 0.78, 0.93]), takes=4)
    row = outcome["result"]["reconstructed_audio"]["segments"][0]
    assert [c["seed"] for c in fish.calls] == [100, 101, 102]
    assert row["selected_take"] == 2 and row["seed"] == 102 and row["voice_similarity"] == 0.93
    assert [t["similarity"] for t in row["takes"]] == [0.70, 0.78, 0.93]


def test_all_takes_below_the_similarity_floor_fail_loudly(tmp_path):
    with pytest.raises(ValueError, match="voice_match_below_threshold"):
        run(tmp_path, edit(), FakeVerifier([0.5, 0.6, 0.7]))


def test_takes_that_do_not_say_the_new_words_are_rejected(tmp_path):
    wrong = FakeVerifier([0.95], decoded=["WE", "SHIP", "IT", "TODAY", "AT", "NOON"])
    with pytest.raises(ValueError, match="generated_words_unverified"):
        run(tmp_path, edit(), wrong)
    assert len(wrong.calls) == 3  # every take was tried before giving up


def test_unalignable_sentence_falls_back_to_generating_the_new_words_alone(tmp_path):
    outcome, fish, _ = run(tmp_path, edit(seed=7), FakeVerifier([0.9], aligned=False))
    row = outcome["result"]["reconstructed_audio"]["segments"][0]
    assert [c["text"] for c in fish.calls] == ["we ship it tomorrow at noon", "tomorrow"]
    assert fish.calls[1]["seed"] == 7
    assert row["splice"] == {
        "mode": "whole_segment",
        "reason": "context_alignment_unavailable",
        "cut_start_seconds": 0.0,
        "cut_end_seconds": 2.0,
        "new_words": [],
    }
    assert row["duration"] == 2.0


def test_without_a_verifier_only_the_new_words_are_generated(tmp_path):
    outcome, fish, _ = run(tmp_path, edit(), None, takes=3)
    result = outcome["result"]["reconstructed_audio"]
    row = result["segments"][0]
    assert [c["text"] for c in fish.calls] == ["tomorrow"]
    assert row["splice"]["reason"] == "verifier_unavailable" and row["words_verified"] is None
    assert result["word_accuracy_verified"] is False and result["voice_verification"] is None


def test_caller_can_limit_takes(tmp_path):
    outcome, fish, _ = run(tmp_path, edit(takes=1), FakeVerifier([0.76]), takes=4)
    assert len(fish.calls) == 1
    assert outcome["result"]["reconstructed_audio"]["segments"][0]["voice_similarity"] == 0.76


def test_preview_carries_verification_too(tmp_path):
    outcome, _, _ = run(tmp_path, edit(seed=3), FakeVerifier([0.88]), operation="preview")
    result = outcome["result"]["reconstructed_audio"]
    assert result["preview"] is True and result["word_accuracy_verified"] is True
    assert result["segments"][0]["splice"]["mode"] == "words"


def test_deletions_cannot_carry_context():
    from hear.contracts.reconstruction import SpeechEdit

    with pytest.raises(ValueError, match="deletion_takes_no_context"):
        SpeechEdit(segment_start=1, segment_end=2, is_deletion=True, context_before="x")
    assert (
        SpeechEdit(
            segment_start=1, segment_end=2, new_text="b", context_before="a", context_after="c"
        ).generation_text
        == "a b c"
    )
    assert Path  # keep the import used for readers of the helper signatures


def hi_alone(**extra):
    """A one-word utterance with no sentence around it."""
    return {
        "same_speaker": True,
        "reference": REFERENCE,
        "changes": [
            {"segment_start": 6, "segment_end": 6.3, "new_text": "Hi", "original_text": "Hey"}
        ],
        **extra,
    }


def test_a_lone_short_word_is_not_failed_on_unreliable_similarity(tmp_path):
    # Half-second takes: similarity ranks them but cannot reject them.
    outcome, fish, _ = run(
        tmp_path, hi_alone(), FakeVerifier([0.3, 0.55, 0.4]), fish=SyntheticSpeechDouble()
    )
    row = outcome["result"]["reconstructed_audio"]["segments"][0]
    assert len(fish.calls) == 3 and row["selected_take"] == 1 and row["voice_similarity"] == 0.55
    assert row["takes"][1]["similarity_reliable"] is False
    assert row["words_verified"] is True and row["splice"]["reason"] == "no_context"
    assert row["duration"] == 0.5


def test_a_take_that_says_more_than_asked_is_rejected(tmp_path):
    chatty = FakeVerifier([0.9], decoded=["HI", "THERE", "EVERYONE", "WELCOME"])
    with pytest.raises(ValueError, match="generated_words_unverified"):
        run(tmp_path, hi_alone(), chatty)
    # One extra decoded word is tolerated: CTC sometimes splits a word in two.
    outcome, _, _ = run(tmp_path, hi_alone(), FakeVerifier([0.9], decoded=["HI", "THERE"]))
    assert outcome["result"]["reconstructed_audio"]["segments"][0]["words_verified"] is True
