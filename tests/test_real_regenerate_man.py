"""
Real-world end-to-end test: Edit Transcript → Regenerate → Retranscribe → Correct

Uses actual audio from test_audio.wav with known Whisper output.
Shows the full pipeline comparing before/after correction.
"""

import asyncio
import io
import os
import sys
import tempfile

import torch
import torchaudio

sys.path.insert(0, "/workspace/hear-ai")

from app.services.diff_engine import (
    compute_edit_segments,
    edit_segments_to_changes,
    restore_punctuation_from_edit,
    correct_whisper_mishearings,
)
from app.services.fishspeech_client import FishSpeechClient
from app.services.synthesizer import SpeechSynthesizer
from app.services.transcriber import TranscriptionService
from app.services.enhancer_utils.tts_post_processor import TTSPostProcessor


def _save_wav(wf, sr):
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    torchaudio.save(path, wf, sr)
    return path


def _sep(title):
    print(f"\n{'='*75}")
    print(f"  {title}")
    print(f"{'='*75}")


def _line(label, text, marker=""):
    print(f"  {marker:4s} {label:12s}: {text[:90]}")


def count_punct(t):
    return sum(1 for c in t if c in ",.!?;")


async def main():
    _sep("STEP 1: Transcribe original audio")
    wf, sr = torchaudio.load("test_audio.wav")
    if sr != 44100:
        wf = torchaudio.functional.resample(wf, sr, 44100)
        sr = 44100

    clip_start = int(sr * 15)
    clip_end = clip_start + int(sr * 12)
    original_clip = wf[:, clip_start:clip_end]
    clip_path = _save_wav(original_clip, sr)

    transcriber = TranscriptionService()
    transcriber.load()

    transcript_data = await transcriber.transcribe(
        open(clip_path, "rb").read(), language="en"
    )
    original_text = (transcript_data.get("transcript") or "").strip()
    word_segments = (transcript_data.get("segments") or [])
    conf = transcript_data.get("confidence", 0)

    _line("Original", original_text)
    _line("Confidence", f"{conf:.4f}")
    _line("Segments", str(len(word_segments)))
    os.unlink(clip_path)

    # ────────────────────────────────────────────────────────────────
    _sep("STEP 2: Simulate user editing the transcript")
    edited = (
        "Political party went to the zoo on a Thursday, because it was a great idea. "
        "They went with Councillor Gillian Ford as its leader, "
        "and Councillor Barry Mugglestone as her deputy."
    )

    _line("Original", original_text)
    _line("Edited", edited)
    print(f"\n  User changes: punctuation added (commas, periods), 'good' → 'great'")

    # ────────────────────────────────────────────────────────────────
    _sep("STEP 3: Diff engine finds changed segments")
    edit_segments = compute_edit_segments(
        original_transcript=original_text,
        edited_transcript=edited,
        word_segments=word_segments,
    )
    changes = edit_segments_to_changes(edit_segments)

    print(f"  Changes detected: {len(changes)}")
    for i, ch in enumerate(changes):
        _line(f"Change {i+1}", f"[{ch['segment_start']:.1f}s-{ch['segment_end']:.1f}s]")
        _line("  Original", ch["original_text"])
        _line("  New", ch["new_text"])

    if not changes:
        print("  No changes detected! Exiting.")
        return

    # ────────────────────────────────────────────────────────────────
    _sep("STEP 4: Generate TTS for each changed segment")

    synthesizer = SpeechSynthesizer()
    synthesizer.load()
    fish = FishSpeechClient()

    # Full audio for splicing
    full_path = _save_wav(wf, sr)
    original_wf = wf

    merged = original_wf
    temp_files = [full_path]

    for i, ch in enumerate(changes):
        ss = int(ch["segment_start"] * sr)
        se = int(ch["segment_end"] * sr)

        ref_clip = merged[:, ss:se]
        ref_path = _save_wav(ref_clip, sr)
        temp_files.append(ref_path)

        print(f"\n  Segment {i+1}: '{ch['new_text'][:70]}...'")
        print(f"    Time: {ch['segment_start']:.1f}s → {ch['segment_end']:.1f}s")

        emotion = SpeechSynthesizer._analyze_prosody(ref_clip, sr, 0, ref_clip.shape[1])
        if emotion:
            print(f"    Prosody → temp={emotion.get('temperature')}, "
                  f"top_p={emotion.get('top_p')}, chunk={emotion.get('chunk_length')}")

        tts_bytes = fish.generate_speech(
            text=ch["new_text"],
            reference_audio_path=ref_path,
            temperature=emotion.get("temperature") if emotion else None,
            top_p=emotion.get("top_p") if emotion else None,
            chunk_length=emotion.get("chunk_length") if emotion else None,
        )

        fd, tts_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        temp_files.append(tts_path)
        with open(tts_path, "wb") as f:
            f.write(tts_bytes)

        tts_wf, tts_sr = torchaudio.load(tts_path)
        if tts_sr != sr:
            tts_wf = torchaudio.functional.resample(tts_wf, tts_sr, sr)

        ref_seg = merged[:, ss:se]
        tts_wf = synthesizer._time_stretch_to_match(tts_wf, ref_seg)
        tts_wf = TTSPostProcessor.process(tts_wf, ref_seg, sr)
        merged = synthesizer._splice_segment(merged, tts_wf, ss, se)

        print(f"    TTS: {tts_wf.shape[1]/sr:.1f}s, spliced OK")

    # Save regenerated
    regen_path = _save_wav(merged, sr)
    temp_files.append(regen_path)
    print(f"\n  Regenerated total: {merged.shape[1]/sr:.1f}s")

    # ────────────────────────────────────────────────────────────────
    _sep("STEP 5: Retranscribe regenerated audio (pipeline callback flow)")

    regen_data = await transcriber.transcribe(
        open(regen_path, "rb").read(), language="en"
    )
    regen_text = (regen_data.get("transcript") or "").strip()
    regen_conf = regen_data.get("confidence", 0)

    _line("Regenerated", regen_text)
    _line("Confidence", f"{regen_conf:.4f}")

    edit_p = count_punct(edited)
    regen_p = count_punct(regen_text)
    print(f"  Punctuation:   Edited={edit_p}, Whisper output={regen_p}"
          f" ({(regen_p/edit_p*100) if edit_p else 0:.0f}%)")

    # ────────────────────────────────────────────────────────────────
    _sep("STEP 6: Correction pipeline (punctuation + fuzzy matching)")

    restored = restore_punctuation_from_edit(regen_text, edited)
    corrected = correct_whisper_mishearings(restored, edited, max_distance=3)

    _line("Edited", edited)
    _line("Raw Whisper", regen_text)
    _line("→ Restored", restored, ">>>")
    _line("→ Corrected", corrected, ">>>")

    rest_p = count_punct(restored)
    corr_p = count_punct(corrected)

    print(f"\n  Punctuation journey:")
    print(f"    Edited has:           {edit_p}")
    print(f"    Whisper raw:          {regen_p}  ({regen_p/edit_p*100:.0f}%)" if edit_p else f"    Whisper raw:          {regen_p}")
    print(f"    After restore:        {rest_p}  ({rest_p/edit_p*100:.0f}%)" if edit_p else f"    After restore:        {rest_p}")
    print(f"    After correct:        {corr_p}  ({corr_p/edit_p*100:.0f}%)" if edit_p else f"    After correct:        {corr_p}")

    import re
    def _strip(s):
        return set(re.sub(r"[^\w\s]", "", s).lower().split())

    e_set = _strip(edited)
    r_set = _strip(regen_text)
    c_set = _strip(corrected)

    print(f"\n  Word-level accuracy vs edited:")
    print(f"    Raw Whisper:    {len(r_set & e_set)}/{len(e_set)} words match ({len(r_set & e_set)/len(e_set)*100:.0f}%)")
    print(f"    After pipeline: {len(c_set & e_set)}/{len(e_set)} words match ({len(c_set & e_set)/len(e_set)*100:.0f}%)")

    # ────────────────────────────────────────────────────────────────
    _sep("VERDICT")

    if corrected == edited:
        print("  PERFECT RESTORATION! Pipeline output matches edited transcript exactly.")
    elif len(c_set & e_set) == len(e_set):
        print("  ALL WORDS MATCH. Only minor punctuation/casing differences remain.")
    else:
        missing = e_set - c_set
        extra = c_set - e_set
        if missing:
            print(f"  Missing words: {', '.join(sorted(missing)[:5])}")
        if extra:
            print(f"  Extra words: {', '.join(sorted(extra)[:5])}")

    # Cleanup
    for p in temp_files:
        if os.path.exists(p):
            os.unlink(p)


if __name__ == "__main__":
    asyncio.run(main())
