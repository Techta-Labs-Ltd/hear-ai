"""Deterministic transcript diff engine for edited-transcript reconstruction.

Takes an original transcript (with word-level timestamps from Whisper) and an
edited transcript, computes the minimal set of changed segments, expands each
change into a natural phrase with surrounding context, and merges nearby edits.

No LLM is used -- pure difflib-based, deterministic and fast.
"""

import difflib
import re
from dataclasses import dataclass, field
from typing import Optional

from hear.config import settings


@dataclass
class EditSegment:
    start_time: float
    end_time: float
    original_text: str
    edited_text: str
    left_context: str
    right_context: str


@dataclass
class _WordToken:
    word: str
    start: float
    end: float


def _flatten_words(segments: list[dict]) -> list[_WordToken]:
    """Flatten segments[].words[] into a single ordered list of _WordToken."""
    tokens: list[_WordToken] = []
    for seg in segments:
        for w in seg.get("words", []):
            word_text = w.get("word", "").strip()
            if not word_text:
                continue
            tokens.append(_WordToken(
                word=word_text,
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", 0.0)),
            ))
    return tokens


def _normalize_text(text: str) -> list[str]:
    """Split text into lowercase word tokens for diff matching."""
    return text.strip().lower().split()


def _strip_word(w: str) -> str:
    """Normalize a word for comparison by removing surrounding punctuation."""
    return re.sub(r'^[^\w]+|[^\w]+$', '', w).lower()


def _sentence_boundaries(words: list[_WordToken]) -> list[int]:
    """Find sentence boundary positions (indices after ``. ! ?`` tokens).

    Returns a sorted list of word indices that mark the END of a sentence.
    Index ``i`` means ``words[i]`` is the last word of a sentence.
    """
    boundaries: list[int] = []
    for i, w in enumerate(words):
        stripped = w.word.strip()
        if stripped.endswith((".", "!", "?")) and not stripped.lower() in (
            "mr.", "mrs.", "dr.", "ms.", "prof.", "rev.", "st.", "ave.",
        ):
            if re.match(r'^\d+\.$|^[A-Z]{2,5}\.$|^[A-Z][a-z]\.$', stripped):
                continue
            boundaries.append(i)
    return boundaries


def _snap_to_sentences(
    ranges: list[tuple[int, int]],
    words: list[_WordToken],
) -> list[tuple[int, int]]:
    """Expand each range to the nearest sentence boundaries.

    Start expands LEFT to the preceding sentence boundary (or stays at 0).
    End expands RIGHT to the next sentence boundary (or end of word list).
    Returns unique sentence-level ranges (adjacent same-sentence ranges merged).
    """
    if not ranges or not words:
        return ranges

    boundaries = _sentence_boundaries(words)
    if not boundaries:
        return ranges

    seen: set[tuple[int, int]] = set()
    for start_idx, end_idx in ranges:
        # Find the sentence this range falls into
        new_start = 0
        for b in boundaries:
            if b < start_idx:
                new_start = b + 1  # start after this boundary
        new_end = len(words)
        for b in boundaries:
            if b >= end_idx - 1:
                new_end = b + 1  # include through this boundary
                break
        if new_end <= new_start:
            new_end = min(new_start + 1, len(words))
        seen.add((new_start, new_end))

    return sorted(seen)


def compute_edit_segments(
    original_transcript: str,
    edited_transcript: str,
    word_segments: list[dict],
    *,
    expansion_words: Optional[int] = None,
    merge_gap_seconds: Optional[float] = None,
) -> list[EditSegment]:
    """Compare original and edited transcripts and return minimal edit segments.

    Parameters
    ----------
    original_transcript : str
        The original transcript text.
    edited_transcript : str
        The user-edited transcript text.
    word_segments : list[dict]
        Word-level segments from Faster-Whisper, each with a ``words`` list of
        ``{word, start, end, prob}`` dicts.
    expansion_words : int, optional
        Number of context words to expand around each change. Defaults to
        ``settings.EDIT_PHRASE_EXPANSION_WORDS``.
    merge_gap_seconds : float, optional
        Maximum gap (in seconds) between edits that should be merged into a
        single segment. Defaults to ``settings.EDIT_MERGE_GAP_SECONDS``.

    Returns
    -------
    list[EditSegment]
        Ordered list of segments that need regeneration.
    """
    expand = expansion_words if expansion_words is not None else settings.EDIT_PHRASE_EXPANSION_WORDS
    merge_gap = merge_gap_seconds if merge_gap_seconds is not None else settings.EDIT_MERGE_GAP_SECONDS

    orig_words = _flatten_words(word_segments)
    if not orig_words:
        return _fallback_full_diff(original_transcript, edited_transcript)

    orig_tokens = [w.word.strip().lower() for w in orig_words]
    edit_tokens_lower = _normalize_text(edited_transcript)
    edit_tokens_case = edited_transcript.strip().split()

    sm = difflib.SequenceMatcher(None, orig_tokens, edit_tokens_lower, autojunk=False)
    opcodes = sm.get_opcodes()

    changed_ranges: list[tuple[int, int]] = []
    has_trailing_cut = False
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        if tag in ("delete", "replace") and len(edit_tokens_lower) >= 5 and j1 >= len(edit_tokens_lower) - 2 and (i2 - i1) > (j2 - j1) * 3:
            # Trailing: original has much more content than edit at the end.
            # Only include the edit-side portion for TTS; the rest is a deletion.
            if tag == "replace":
                changed_ranges.append((i1, min(i1 + (j2 - j1) + 5, i2)))
                has_trailing_cut = True
            elif tag == "delete":
                has_trailing_cut = True
            continue
        changed_ranges.append((i1, i2))

    trailing_delete_start = None
    if has_trailing_cut:
        for tag, i1, i2, j1, j2 in reversed(opcodes):
            if tag in ("delete", "replace") and (i2 - i1) > (j2 - j1) * 3:
                trailing_delete_start = i1
                break
            elif tag == "equal":
                break

    if not changed_ranges:
        return []

    snapped = _snap_to_sentences(changed_ranges, orig_words)
    if len(snapped) > 1 or _sentence_boundaries(orig_words):
        # Sentence boundaries were found and multiple sentences affected
        merged = sorted(snapped)
    else:
        # Fall back to word-level expansion + merging
        expanded = _expand_ranges(changed_ranges, len(orig_tokens), expand)
        merged = _merge_overlapping_ranges(expanded, orig_words, merge_gap)

    segments: list[EditSegment] = []
    for rng in merged:
        start_idx, end_idx = rng
        end_idx = min(end_idx, len(orig_words))

        if start_idx >= end_idx:
            continue

        start_time = orig_words[start_idx].start
        end_time = orig_words[end_idx - 1].end
        original_text = " ".join(w.word for w in orig_words[start_idx:end_idx])

        edited_text = _compute_edited_text(orig_tokens, edit_tokens_case, opcodes, start_idx, end_idx)

        if end_idx < len(orig_words) and edited_text.strip():
            trailing_orig = [_strip_word(w.word) for w in orig_words[end_idx:end_idx + 10]]
            edit_words_list = edited_text.strip().split()
            strip_count = 0
            for i in range(1, min(len(edit_words_list), len(trailing_orig)) + 1):
                e_w = _strip_word(edit_words_list[-i])
                t_w = trailing_orig[i - 1]
                if e_w == t_w or _edit_distance(e_w, t_w) <= 1:
                    strip_count = i
                else:
                    break
            if strip_count > 0:
                end_idx -= strip_count
                if end_idx <= start_idx:
                    end_idx = start_idx + 1
                edited_text = " ".join(edit_words_list[:-strip_count])
                original_text = " ".join(w.word for w in orig_words[start_idx:end_idx])
                end_time = orig_words[end_idx - 1].end

        left_ctx_start = max(0, start_idx - expand)
        left_context = " ".join(w.word for w in orig_words[left_ctx_start:start_idx])

        right_ctx_end = min(len(orig_words), end_idx + expand)
        right_context = " ".join(w.word for w in orig_words[end_idx:right_ctx_end])

        segments.append(EditSegment(
            start_time=round(start_time, 3),
            end_time=round(end_time, 3),
            original_text=original_text,
            edited_text=edited_text,
            left_context=left_context,
            right_context=right_context,
        ))

    # Deduplicate: remove superset segments (larger time range containing a smaller one)
    # when the larger segment's edited text is empty or identical to the smaller's.
    deduped: list[EditSegment] = []
    for seg in sorted(segments, key=lambda s: (s.end_time - s.start_time)):
        is_redundant = False
        for existing in deduped:
            if (seg.start_time <= existing.start_time and seg.end_time >= existing.end_time
                    and (not seg.edited_text.strip() or seg.edited_text == existing.edited_text)):
                is_redundant = True
                break
        if not is_redundant:
            deduped.append(seg)

    # Add trailing deletion: original has extra text beyond the edit
    if trailing_delete_start is not None and trailing_delete_start < len(orig_words):
        td_start = trailing_delete_start
        td_end = len(orig_words)
        td_start_time = round(orig_words[td_start].start, 3)
        td_end_time = round(orig_words[td_end - 1].end, 3)
        if not any(d.start_time <= td_start_time and d.end_time >= td_end_time
                   for d in deduped):
            deduped.append(EditSegment(
                start_time=td_start_time,
                end_time=td_end_time,
                original_text=" ".join(w.word for w in orig_words[td_start:td_end]),
                edited_text="",
                left_context="",
                right_context="",
            ))

    return deduped


def _expand_ranges(
    ranges: list[tuple[int, int]],
    total: int,
    expansion: int,
) -> list[tuple[int, int]]:
    """Expand each range by *expansion* words on each side."""
    expanded: list[tuple[int, int]] = []
    for start, end in ranges:
        expanded.append((
            max(0, start - expansion),
            min(total, end + expansion),
        ))
    return expanded


def _merge_overlapping_ranges(
    ranges: list[tuple[int, int]],
    words: list[_WordToken],
    gap_seconds: float,
) -> list[tuple[int, int]]:
    """Merge overlapping/adjacent ranges and ranges within *gap_seconds*."""
    if not ranges:
        return []

    sorted_ranges = sorted(ranges, key=lambda r: r[0])
    merged: list[tuple[int, int]] = [sorted_ranges[0]]

    for start, end in sorted_ranges[1:]:
        prev_start, prev_end = merged[-1]

        time_gap = 0.0
        if prev_end < len(words) and start < len(words):
            time_gap = words[start].start - words[prev_end - 1].end

        if start <= prev_end or time_gap <= gap_seconds:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def _compute_edited_text(
    orig_tokens: list[str],
    edit_tokens: list[str],
    opcodes: list[tuple],
    range_start: int,
    range_end: int,
) -> str:
    """Determine the replacement text for a given range of original tokens.

    Collects the edited-side tokens from all opcodes whose original-side range
    overlaps with [range_start, range_end).
    """
    parts: list[str] = []

    for tag, i1, i2, j1, j2 in opcodes:
        if i2 <= range_start:
            continue
        if i1 >= range_end:
            continue

        if tag == "equal":
            overlap_start = max(i1, range_start)
            overlap_end = min(i2, range_end)
            offset_in_op = overlap_start - i1
            length = overlap_end - overlap_start
            for k in range(length):
                parts.append(edit_tokens[j1 + offset_in_op + k])
        elif tag == "replace":
            if j1 < j2:
                parts.extend(edit_tokens[j1:j2])
        elif tag == "insert":
            if i1 >= range_start and i1 < range_end:
                parts.extend(edit_tokens[j1:j2])
        elif tag == "delete":
            pass

    return " ".join(parts) if parts else ""


def _fallback_full_diff(
    original_transcript: str,
    edited_transcript: str,
) -> list[EditSegment]:
    """Fallback when no word-level timestamps are available.

    Returns a single EditSegment spanning the entire transcript with a rough
    estimated duration.  This is used when word_segments is empty (e.g. the
    track was never transcribed with word_timestamps=True).
    """
    orig = original_transcript.strip()
    edited = edited_transcript.strip()
    if orig == edited:
        return []

    return [EditSegment(
        start_time=0.0,
        end_time=0.0,
        original_text=orig,
        edited_text=edited,
        left_context="",
        right_context="",
    )]


def edit_segments_to_changes(segments: list[EditSegment]) -> list[dict]:
    """Convert EditSegment list to SegmentChange-compatible dicts.

    Filters out segments with zero-duration that came from the fallback path
    (those require full rebuild, not segment reconstruction).  Marks segments
    where the edited text is empty as deletions.
    """
    changes: list[dict] = []
    for seg in segments:
        if seg.start_time <= 0.0 and seg.end_time <= 0.0:
            continue
        if seg.end_time <= seg.start_time:
            continue
        text = seg.edited_text.strip()
        is_deletion = not text
        changes.append({
            "segment_start": seg.start_time,
            "segment_end": seg.end_time,
            "new_text": seg.edited_text,
            "original_text": seg.original_text,
            "is_deletion": is_deletion,
        })
    return changes


def restore_punctuation_from_edit(
    whisper_transcript: str,
    edited_transcript: str,
) -> str:
    """Align Whisper output with the edited transcript and restore punctuation.

    Whisper often fails to produce punctuation when transcribing TTS-generated
    audio because TTS lacks natural prosodic cues.  This function uses sequence
    alignment to match Whisper's words against the edited transcript and
    substitutes the edited transcript's word forms (which include punctuation
    and original casing) for matching words.

    Parameters
    ----------
    whisper_transcript : str
        The raw Whisper transcription (may lack punctuation).
    edited_transcript : str
        The user-edited transcript with correct punctuation and casing.

    Returns
    -------
    str
        The restored transcript with punctuation from the edited transcript
        applied to matching Whisper words.
    """
    whisper_words = whisper_transcript.strip().split()
    edit_words = edited_transcript.strip().split()

    if not whisper_words:
        return edited_transcript.strip()
    if not edit_words:
        return whisper_transcript.strip()

    def _strip_punct(w: str) -> str:
        return re.sub(r'^[^\w]+|[^\w]+$', '', w).lower()

    whisper_stripped = [_strip_punct(w) for w in whisper_words]
    edit_stripped = [_strip_punct(w) for w in edit_words]

    sm = difflib.SequenceMatcher(None, whisper_stripped, edit_stripped, autojunk=False)

    restored: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            restored.extend(edit_words[j1:j2])
        elif tag == "replace":
            edit_sub = edit_words[j1:j2]
            edit_stripped_sub = edit_stripped[j1:j2]
            whisper_sub = whisper_words[i1:i2]
            whisper_stripped_sub = whisper_stripped[i1:i2]

            used_indices: set[int] = set()
            for idx, w in enumerate(whisper_sub):
                w_s = whisper_stripped_sub[idx]
                found = False
                for e_idx, e_s in enumerate(edit_stripped_sub):
                    if e_idx in used_indices:
                        continue
                    if e_s == w_s:
                        restored.append(edit_sub[e_idx])
                        used_indices.add(e_idx)
                        found = True
                        break
                if not found:
                    restored.append(w)
        elif tag == "delete":
            restored.extend(whisper_words[i1:i2])
        elif tag == "insert":
            restored.extend(edit_words[j1:j2])

    return " ".join(restored) if restored else edited_transcript.strip()


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(
                prev[j + 1] + 1,
                curr[j] + 1,
                prev[j] + (0 if ca == cb else 1),
            ))
        prev = curr
    return prev[-1]


def correct_whisper_mishearings(
    whisper_text: str,
    edited_text: str,
    *,
    max_distance: int = 2,
) -> str:
    """Fix Whisper word-level errors by fuzzy-matching against the edited transcript.

    When Whisper transcribes TTS audio, it may mishear individual words
    (e.g. ``favour`` → ``favre``).  This function uses sequence alignment
    plus Levenshtein distance to detect near-matches and replace them with
    the correct word from the edited transcript.

    Parameters
    ----------
    whisper_text : str
        Raw Whisper output, already punctuation-restored if applicable.
    edited_text : str
        The user-edited transcript (ground truth).
    max_distance : int
        Maximum Levenshtein distance for a word to be considered a mishearing.

    Returns
    -------
    str
        Corrected transcript with misheard words fixed.
    """
    whisper_words = whisper_text.strip().split()
    edit_words = edited_text.strip().split()

    if not whisper_words or not edit_words:
        return whisper_text.strip()

    def _strip_punct(w: str) -> str:
        return re.sub(r'^[^\w]+|[^\w]+$', '', w).lower()

    whisper_stripped = [_strip_punct(w) for w in whisper_words]
    edit_stripped = [_strip_punct(w) for w in edit_words]

    sm = difflib.SequenceMatcher(None, whisper_stripped, edit_stripped, autojunk=False)
    opcodes = sm.get_opcodes()

    corrected: list[str] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            corrected.extend(whisper_words[i1:i2])
        elif tag == "replace":
            w_sub = whisper_words[i1:i2]
            e_sub = edit_words[j1:j2]
            w_stripped_sub = whisper_stripped[i1:i2]
            e_stripped_sub = edit_stripped[j1:j2]

            used_edit: set[int] = set()
            for idx, w in enumerate(w_sub):
                w_s = w_stripped_sub[idx]
                best_match = None
                best_dist = max_distance + 1
                for e_idx, e_s in enumerate(e_stripped_sub):
                    if e_idx in used_edit:
                        continue
                    dist = _edit_distance(w_s, e_s)
                    if dist <= max_distance and dist < best_dist:
                        best_dist = dist
                        best_match = e_idx
                if best_match is not None:
                    corrected.append(e_sub[best_match])
                    used_edit.add(best_match)
                else:
                    corrected.append(w)
        elif tag == "delete":
            corrected.extend(whisper_words[i1:i2])
        elif tag == "insert":
            corrected.extend(edit_words[j1:j2])

    return " ".join(corrected) if corrected else whisper_text.strip()
