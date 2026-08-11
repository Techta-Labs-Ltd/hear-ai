"""Unit tests for the deterministic transcript diff engine.

Tests the core logic of compute_edit_segments, phrase expansion, segment
merging, and the edit_segments_to_changes conversion helper.  No GPU or
network access required.
"""

import pytest

from hear.services.reconstruction.diff import (
    EditSegment,
    _WordToken,
    _compute_edited_text,
    _edit_distance,
    _expand_ranges,
    _flatten_words,
    _merge_overlapping_ranges,
    _normalize_text,
    compute_edit_segments,
    correct_whisper_mishearings,
    edit_segments_to_changes,
    restore_punctuation_from_edit,
)


def _make_word_segments(pairs: list[tuple[str, float, float]]) -> list[dict]:
    """Helper to build word_segments from (word, start, end) tuples."""
    words = [{"word": w, "start": s, "end": e, "prob": 0.95} for w, s, e in pairs]
    return [{"id": 0, "start": words[0]["start"], "end": words[-1]["end"], "text": " ".join(w for w, _, _ in pairs), "words": words}]


def _make_multi_segment(segments: list[list[tuple[str, float, float]]]) -> list[dict]:
    """Helper to build multi-segment word_segments."""
    result = []
    for idx, pairs in enumerate(segments):
        words = [{"word": w, "start": s, "end": e, "prob": 0.95} for w, s, e in pairs]
        result.append({
            "id": idx,
            "start": words[0]["start"],
            "end": words[-1]["end"],
            "text": " ".join(w for w, _, _ in pairs),
            "words": words,
        })
    return result


class TestFlattenWords:
    def test_single_segment(self):
        segs = _make_word_segments([("hello", 0.0, 0.5), ("world", 0.5, 1.0)])
        flat = _flatten_words(segs)
        assert len(flat) == 2
        assert flat[0].word == "hello"
        assert flat[1].word == "world"

    def test_multi_segment(self):
        segs = _make_multi_segment([
            [("hello", 0.0, 0.5)],
            [("world", 1.0, 1.5)],
        ])
        flat = _flatten_words(segs)
        assert len(flat) == 2
        assert flat[0].start == 0.0
        assert flat[1].start == 1.0

    def test_empty_segments(self):
        assert _flatten_words([]) == []

    def test_empty_words(self):
        assert _flatten_words([{"words": []}]) == []


class TestNormalizeText:
    def test_basic(self):
        assert _normalize_text("Hello World") == ["hello", "world"]

    def test_extra_whitespace(self):
        assert _normalize_text("  hello   world  ") == ["hello", "world"]


class TestExpandRanges:
    def test_no_overlap(self):
        ranges = [(5, 10), (20, 25)]
        expanded = _expand_ranges(ranges, 30, 3)
        assert expanded[0] == (2, 13)
        assert expanded[1] == (17, 28)

    def test_boundary_clamp(self):
        ranges = [(0, 2)]
        expanded = _expand_ranges(ranges, 10, 5)
        assert expanded[0] == (0, 7)

    def test_total_clamp(self):
        ranges = [(8, 10)]
        expanded = _expand_ranges(ranges, 10, 5)
        assert expanded[0] == (3, 10)


class TestMergeOverlappingRanges:
    def _make_words(self, n: int, gap: float = 0.5):
        return [_WordToken(word=f"w{i}", start=i * gap, end=i * gap + 0.3) for i in range(n)]

    def test_merge_adjacent(self):
        words = self._make_words(10)
        ranges = [(1, 3), (3, 5)]
        merged = _merge_overlapping_ranges(ranges, words, 1.5)
        assert merged == [(1, 5)]

    def test_no_merge_far_apart(self):
        words = self._make_words(20)
        ranges = [(1, 3), (15, 17)]
        merged = _merge_overlapping_ranges(ranges, words, 1.0)
        assert len(merged) == 2

    def test_merge_within_gap(self):
        words = self._make_words(10)
        ranges = [(1, 3), (4, 6)]
        merged = _merge_overlapping_ranges(ranges, words, 1.5)
        assert merged == [(1, 6)]


class TestComputeEditedText:
    def test_simple_replace(self):
        orig = ["today", "we", "sold", "fifteen", "products"]
        edit = ["today", "we", "sold", "twenty", "five", "products"]
        opcodes = [("equal", 0, 3, 0, 3), ("replace", 3, 4, 3, 5), ("equal", 4, 5, 5, 6)]
        result = _compute_edited_text(orig, edit, opcodes, 2, 5)
        assert "twenty five" in result
        assert "products" in result


class TestComputeEditSegments:
    def test_no_changes(self):
        segs = _make_word_segments([("hello", 0.0, 0.5), ("world", 0.5, 1.0)])
        result = compute_edit_segments("hello world", "hello world", segs)
        assert result == []

    def test_single_word_change(self):
        segs = _make_word_segments([
            ("today", 0.0, 0.4),
            ("we", 0.4, 0.6),
            ("sold", 0.6, 0.9),
            ("fifteen", 0.9, 1.4),
            ("products", 1.4, 1.9),
        ])
        result = compute_edit_segments(
            "today we sold fifteen products",
            "today we sold twenty products",
            segs,
            expansion_words=1,
            merge_gap_seconds=999.0,
        )
        assert len(result) >= 1
        assert any("twenty" in seg.edited_text for seg in result)

    def test_insertion(self):
        segs = _make_word_segments([
            ("hello", 0.0, 0.5),
            ("world", 0.5, 1.0),
        ])
        result = compute_edit_segments(
            "hello world",
            "hello beautiful world",
            segs,
            expansion_words=1,
        )
        assert len(result) >= 1
        assert any("beautiful" in seg.edited_text for seg in result)

    def test_deletion(self):
        segs = _make_word_segments([
            ("hello", 0.0, 0.5),
            ("beautiful", 0.5, 1.0),
            ("world", 1.0, 1.5),
        ])
        result = compute_edit_segments(
            "hello beautiful world",
            "hello world",
            segs,
            expansion_words=1,
        )
        assert len(result) >= 1

    def test_fallback_no_word_segments(self):
        result = compute_edit_segments(
            "original text",
            "edited text",
            [],
        )
        assert len(result) == 1
        assert result[0].start_time == 0.0
        assert result[0].end_time == 0.0
        assert result[0].edited_text == "edited text"

    def test_expansion_context(self):
        segs = _make_word_segments([
            ("the", 0.0, 0.2),
            ("quick", 0.2, 0.5),
            ("brown", 0.5, 0.8),
            ("fox", 0.8, 1.1),
            ("jumps", 1.1, 1.4),
        ])
        result = compute_edit_segments(
            "the quick brown fox jumps",
            "the quick red fox jumps",
            segs,
            expansion_words=1,
            merge_gap_seconds=999.0,
        )
        assert len(result) >= 1
        for seg in result:
            assert seg.start_time >= 0.0
            assert seg.end_time > seg.start_time

    def test_multi_segment_input(self):
        segs = _make_multi_segment([
            [("hello", 0.0, 0.5), ("there", 0.5, 1.0)],
            [("good", 2.0, 2.5), ("morning", 2.5, 3.0)],
        ])
        result = compute_edit_segments(
            "hello there good morning",
            "hello there great morning",
            segs,
            expansion_words=0,
        )
        assert len(result) >= 1
        assert any("great" in seg.edited_text for seg in result)


class TestEditSegmentsToChanges:
    def test_filters_fallback_segments(self):
        segments = [
            EditSegment(
                start_time=0.0, end_time=0.0,
                original_text="old", edited_text="new",
                left_context="", right_context="",
            )
        ]
        assert edit_segments_to_changes(segments) == []

    def test_valid_segments(self):
        segments = [
            EditSegment(
                start_time=1.0, end_time=2.0,
                original_text="old text", edited_text="new text",
                left_context="before", right_context="after",
            )
        ]
        changes = edit_segments_to_changes(segments)
        assert len(changes) == 1
        assert changes[0]["segment_start"] == 1.0
        assert changes[0]["segment_end"] == 2.0
        assert changes[0]["new_text"] == "new text"

    def test_filters_zero_duration(self):
        segments = [
            EditSegment(
                start_time=1.5, end_time=1.5,
                original_text="same", edited_text="same",
                left_context="", right_context="",
            )
        ]
        assert edit_segments_to_changes(segments) == []

    def test_mixed(self):
        segments = [
            EditSegment(0.0, 0.0, "a", "b", "", ""),
            EditSegment(1.0, 2.0, "c", "d", "", ""),
            EditSegment(3.0, 3.0, "e", "f", "", ""),
        ]
        changes = edit_segments_to_changes(segments)
        assert len(changes) == 1
        assert changes[0]["segment_start"] == 1.0


class TestRestorePunctuationFromEdit:
    def test_basic_punctuation_restoration(self):
        whisper = "hello world this is great"
        edited = "Hello, world. This is great."
        result = restore_punctuation_from_edit(whisper, edited)
        assert "Hello," in result
        assert "world." in result
        assert "great." in result

    def test_word_change_keeps_whisper_words(self):
        whisper = "hello wonderful world today"
        edited = "Hello, beautiful world. Today is amazing."
        result = restore_punctuation_from_edit(whisper, edited)
        assert "hello" in result.lower()
        assert "world" in result.lower()
        assert "today" in result.lower()
        assert "wonderful" in result.lower()

    def test_insertion_in_edit(self):
        whisper = "hello world"
        edited = "Hello, beautiful world."
        result = restore_punctuation_from_edit(whisper, edited)
        assert "hello" in result.lower()
        assert "world" in result.lower()

    def test_deletion_in_whisper(self):
        whisper = "hello extra words world"
        edited = "Hello world."
        result = restore_punctuation_from_edit(whisper, edited)
        assert "hello" in result.lower()
        assert "world" in result.lower()
        assert "extra" in result.lower()
        assert "words" in result.lower()

    def test_no_change_needed(self):
        whisper = "Hello, world. This is great."
        edited = "Hello, world. This is great."
        result = restore_punctuation_from_edit(whisper, edited)
        assert result == "Hello, world. This is great."

    def test_empty_whisper_returns_edit(self):
        result = restore_punctuation_from_edit("", "Hello, world.")
        assert result == "Hello, world."

    def test_empty_edit_returns_whisper(self):
        result = restore_punctuation_from_edit("hello world", "")
        assert result == "hello world"

    def test_casing_restored(self):
        whisper = "today we sold twenty five products"
        edited = "Today we sold twenty-five products."
        result = restore_punctuation_from_edit(whisper, edited)
        assert "Today" in result
        assert "sold" in result
        assert "products." in result
        assert "twenty five" in result.lower() or "twenty" in result.lower()


class TestEditedTextCasing:
    def test_preserves_original_casing(self):
        segs = _make_word_segments([
            ("hello", 0.0, 0.5),
            ("world", 0.5, 1.0),
        ])
        result = compute_edit_segments(
            "hello world",
            "Hello, World.",
            segs,
            expansion_words=1,
        )
        assert len(result) >= 1
        seg = result[0]
        assert "Hello," in seg.edited_text
        assert "World." in seg.edited_text


class TestEditDistance:
    def test_identical(self):
        assert _edit_distance("hello", "hello") == 0

    def test_one_char_different(self):
        assert _edit_distance("hello", "hallo") == 1

    def test_two_chars_different(self):
        assert _edit_distance("helo", "hello") <= 2

    def test_three_chars_different(self):
        assert _edit_distance("favour", "favre") == 3

    def test_completely_different(self):
        assert _edit_distance("hello", "world") > 2

    def test_empty_string(self):
        assert _edit_distance("", "abc") == 3

    def test_case_sensitive(self):
        assert _edit_distance("Hello", "hello") == 1


class TestCorrectWhisperMishearings:
    def test_fixes_near_miss(self):
        whisper = "hello world this is helo"
        edited = "hello world this is hello"
        result = correct_whisper_mishearings(whisper, edited)
        assert "hello" in result
        assert "helo" not in result

    def test_preserves_correct_words(self):
        whisper = "hello world this is great"
        edited = "hello world this is great"
        result = correct_whisper_mishearings(whisper, edited)
        assert result == "hello world this is great"

    def test_keeps_truly_different_words(self):
        whisper = "hello world this is banana"
        edited = "hello world this is orange"
        result = correct_whisper_mishearings(whisper, edited)
        assert "banana" in result

    def test_multiple_mishearings(self):
        whisper = "helo wrld how are you"
        edited = "hello world how are you"
        result = correct_whisper_mishearings(whisper, edited)
        assert "hello" in result
        assert "world" in result
        assert "helo" not in result
        assert "wrld" not in result

    def test_max_distance_boundary(self):
        whisper = "elephant is big"
        edited = "elegant is big"
        result = correct_whisper_mishearings(whisper, edited, max_distance=2)
        assert "elegant" in result

    def test_empty_inputs(self):
        assert correct_whisper_mishearings("", "hello") == ""
        assert correct_whisper_mishearings("hello", "") == "hello"

    def test_preserves_punctuation(self):
        whisper = "Hello, world. This is wrld today."
        edited = "Hello, world. This is world today."
        result = correct_whisper_mishearings(whisper, edited)
        assert "world" in result
        assert "," in result
        assert "." in result
