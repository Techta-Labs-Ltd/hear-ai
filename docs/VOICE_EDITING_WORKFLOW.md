# Voice editing workflow (frontend and backend)

How the editor should drive `reconstruction` jobs so Fish clones the speaker
correctly and the user cannot submit an edit the worker will do badly. The
worker contract is in [AUDIO_JOBS.md](AUDIO_JOBS.md#reconstruction); this page
is the product-side counterpart.

## 1. Select words, not time ranges

The pipeline's transcription returns `segments[].words[{word, start, end, prob}]`.
`prob` is a measured per-word confidence from an independent CTC model
(`word_confidence_source: "wav2vec2_ctc"`), 0 to 0.999. It is `null` for a word
the model cannot spell (digits, symbols) and for every word of a transcript made
before scoring existed, in which case `word_confidence_available` is false.
**A missing score is "unknown", never "bad": it must not block anything.** The
editor works on that word list:

1. The user highlights one or more **contiguous words** and types the
   replacement. They never edit times and never edit the original words.
2. `segment_start` = `start` of the first selected word, `segment_end` = `end`
   of the last. If a neighbouring gap is under 40 ms, move the boundary to the
   middle of the gap so the cut lands between words rather than inside one.
3. `original_text` = the selected words joined with spaces, verbatim from the
   aligner, including fillers and punctuation as transcribed.
4. `context_before` / `context_after` = the remaining words of the same
   sentence (bounded by sentence punctuation, or by a pause over 300 ms, up to
   about 12 words on each side). These let the worker generate the whole
   sentence and splice back only the changed words.
5. `new_text` = the user's replacement. Keep it short: warn when it is more than
   twice the length of `original_text`, because the interval grows by the
   difference. Spell out digits, currency and abbreviations ("twenty twenty
   six", "pounds", "for example"): the worker does not normalise text and the
   verifier cannot check digits.

An insertion is a selection of zero words at a word boundary:
`segment_start == segment_end` at that boundary, `original_text` empty, with
context on both sides.

## 2. Always send a voice reference

Never rely on the worker cloning from the edited interval. The backend picks the
reference automatically from the same transcript and the frontend shows it as
the track's **voice sample**:

- 8 to 15 seconds of continuous speech by the same speaker (same channel for
  independent stereo voices; set `reference.channel`).
- No gap between words exceeds 500 ms and no word has a zero-length stamp.
- When scores exist, every word has `prob` ≥ 0.9; when the transcript has no
  scores, use the timing rules alone and label the sample "unverified".
- Not inside a segment the pipeline marked silent or music, and not overlapping
  the edit.
- Prefer the candidate closest in time to the edit (mood and pace match).
- `reference.text` = those words verbatim from the aligner.

Cache the choice per track revision. Let the user audition it and pick another
stretch, but always from the transcript, never free text. If no stretch
qualifies, say so and offer re-transcription or a different clip before any
edit is allowed: a job without a reference under 5 s fails on purpose.

## 3. Block edits the model cannot do well

Disable the edit affordance, with a reason shown, when:

- any selected word has a `prob` under 0.6 (a `null` score is allowed; show
  "unverified" and offer Re-transcribe as an optional improvement, never a gate);
- the selection lies in a silent or music segment, or spans two sentences;
- the track is stereo with independent voices and no channel was chosen;
- the language is not English (`en`/`en-GB` only);
- the selection is longer than 20 seconds (split it into two edits);
- more than 64 changes are pending, or the pending text exceeds 20,000
  characters.

Hide `rebuild` for anything over about 30 seconds. It regenerates the entire
track as synthetic speech and is the worst case for voice mismatch.

## 4. Preview in place, then confirm

1. Submit `operation: "preview"` with the changes, the reference, `seed` (any
   integer; store it), and optionally `takes` (1–4; 3 is the default).
2. The outcome has `delivery: null` and one MP3 per segment with its timeline
   row. Play source up to `output_start_seconds`, the segment MP3, then the
   source from `segment_end`. Show `voice_similarity`, `words_verified` and the
   `takes` list; let the user **Regenerate** (new `seed`) or **Accept**.
3. Accept submits the identical body with `operation: "replace_segments"` (or
   `edit_transcript`) and the same `seed`. The worker renders the full track
   once; the backend's existing confirm flow swaps the media and queues a new
   transcription.
4. Previews expire with the job; do not let a stale preview be confirmed after
   the track's `audio_revision` changed.

## 5. Handle the worker's refusals as product states

| Error | Meaning | UI action |
| --- | --- | --- |
| `same_speaker_requires_explicit_reference_or_5_to_20_second_aligned_interval` | No usable reference was sent | Pick a voice sample (section 2) |
| `voice_reference_mostly_silent`, `voice_reference_clipped`, `voice_reference_text_length_mismatch` | The chosen sample is unusable | Pick another stretch automatically, then let the user choose |
| `generated_words_unverified` | No take said the new words | Suggest spelling out numbers or simplifying the phrase; offer regenerate |
| `voice_match_below_threshold` | Takes did not sound like the reference | Regenerate with a new seed; if it repeats, change the voice sample |
| `reference_shorter_than_recommended`, `reference_has_long_pauses` (warnings) | Clone quality may suffer | Show a hint next to the voice sample |

Never auto-approve. `requires_approval` is always true and the backend keeps the
original until the user confirms.

## 6. What the backend adds to the envelope

- The chosen `reference` (section 2), the per-track cached seed, and the
  user's `changes` with `context_before`/`context_after` filled from the
  transcript; the backend, not the browser, derives all text from the aligner.
- On confirm, apply the `segments[]` timeline map to the transcript, then queue
  transcription so word timings are regenerated for later edits.
- Read `/capabilities` → `reconstruction.voice_verification`; if it is null the
  worker runs unverified (takes = 1) and the UI should show previews as
  unverified.
