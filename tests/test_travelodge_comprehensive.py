"""
Comprehensive real-world test: Travelodge audio → Edit → Regenerate → Analyze

Tests:
  1. Original transcription quality
  2. Edit + diff accuracy  
  3. TTS generation with emotion params
  4. Time-stretch behavior (check for speed artifacts)
  5. Splice quality (chopping, pops)
  6. Retranscription accuracy (before/after correction)
  7. Word-level error detection

Outputs audio samples for listening at each stage.
"""

import asyncio
import io
import os
import re
import sys
import tempfile
import wave

import numpy as np
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

AUDIO_FILE = "/workspace/hear-ai/Travelodge Hotel Construction Update.wav"
OUT_DIR = "/workspace/hear-ai/travelodge_test"

def save_wav(wf, sr, name):
    path = os.path.join(OUT_DIR, name)
    torchaudio.save(path, wf, sr)
    print(f"  → {name}: {wf.shape[1]/sr:.1f}s  {os.path.getsize(path)//1024}KB")
    return path

def sep(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def line(label, text, marker=""):
    print(f"  {marker:4s} {label:14s}: {str(text)[:85]}")

def count_punct(t):
    return sum(1 for c in t if c in ",.!?;")

async def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── LOAD ────────────────────────────────────────────────────────
    sep("LOAD AUDIO")
    wf, sr = torchaudio.load(AUDIO_FILE)
    if sr != 44100:
        wf = torchaudio.functional.resample(wf, sr, 44100)
        sr = 44100
    dur_total = wf.shape[1] / sr
    line("File", AUDIO_FILE)
    line("Duration", f"{dur_total:.1f}s")
    line("Sample rate", sr)
    save_wav(wf, sr, "00_full_original.wav")

    # ── TRANSCRIBE ORIGINAL ─────────────────────────────────────────
    sep("TRANSCRIBE ORIGINAL")
    transcriber = TranscriptionService()
    transcriber.load()

    original_data = await transcriber.transcribe(
        open(AUDIO_FILE, "rb").read(), language="en"
    )
    original_text = (original_data.get("transcript") or "").strip()
    word_segments = (original_data.get("segments") or [])
    orig_conf = original_data.get("confidence", 0)

    line("Transcript", original_text)
    line("Confidence", f"{orig_conf:.4f}")
    line("Segments", len(word_segments))
    line("Punctuation", f"{count_punct(original_text)} marks")
    print(f"\n  Full original text:")
    print(f"  {original_text}")

    # ── CREATE EDITED TRANSCRIPT ────────────────────────────────────
    sep("EDIT TRANSCRIPT")
    # Realistic edit: preserve structure, add punctuation, fix a word
    edited = original_text
    # Add punctuation where Whisper missed it
    edits_made = []
    if "update" in edited.lower() and "Update." not in edited:
        edited = edited.replace("update", "Update.")
        edits_made.append("'update' → 'Update.'")
    if "hotel" in edited.lower() and "hotel " in edited.lower() and "Hotel," not in edited:
        edited = re.sub(r'(?i)\bhotel\b', 'Hotel,', edited, count=1)
        edits_made.append("capitalized first 'Hotel' + comma")
    if re.search(r'\b(?:is|will be|was)\b', edited, re.I) and "." not in edited[-5:]:
        # Add period at end if missing
        if not edited.rstrip().endswith((".", "!", "?")):
            edited = edited.rstrip() + "."
            edits_made.append("added final period")

    line("Edits made", ", ".join(edits_made) if edits_made else "minor punctuation")
    line("Original", original_text)
    line("Edited", edited)

    # ── DIFF ────────────────────────────────────────────────────────
    sep("DIFF: FIND CHANGED SEGMENTS")
    edit_segs = compute_edit_segments(
        original_transcript=original_text,
        edited_transcript=edited,
        word_segments=word_segments,
    )
    changes = edit_segments_to_changes(edit_segs)

    print(f"  Edit segments found: {len(edit_segs)}")
    print(f"  Changes (after filter): {len(changes)}")
    for i, s in enumerate(edit_segs):
        print(f"\n  Segment {i+1}: {s.start_time:.1f}s → {s.end_time:.1f}s  ({s.end_time-s.start_time:.1f}s)")
        line("  Original text", s.original_text)
        line("  Edited text", s.edited_text)
        if s.left_context:
            line("  Left context", s.left_context)
        if s.right_context:
            line("  Right context", s.right_context)

    if not changes:
        print("\n  NO CHANGES DETECTED. Exiting.")
        return

    # ── TTS SETUP ───────────────────────────────────────────────────
    sep("TTS SETUP")
    synth = SpeechSynthesizer()
    synth.load()
    fish = FishSpeechClient()

    # ── ANALYZE PROSODY per segment ─────────────────────────────────
    sep("PROSODY ANALYSIS per segment")
    for i, ch in enumerate(changes):
        ss = int(ch["segment_start"] * sr)
        se = int(ch["segment_end"] * sr)
        ref = wf[:, ss:se]
        emotion = SpeechSynthesizer._analyze_prosody(ref, sr, 0, ref.shape[1])
        print(f"  Segment {i+1} [{ch['segment_start']:.1f}s-{ch['segment_end']:.1f}s]:")
        print(f"    Ref dur: {ref.shape[1]/sr:.1f}s")
        print(f"    Emotion: {emotion}")
        ch["_emotion"] = emotion

    # ── REGENERATE ──────────────────────────────────────────────────
    sep("REGENERATE: TTS generation per segment")

    merged = wf
    all_samples = []

    for i, ch in enumerate(changes):
        ss = int(ch["segment_start"] * sr)
        se = int(ch["segment_end"] * sr)
        text = ch["new_text"]

        print(f"\n  ── Change {i+1} ──")
        line("Time range", f"{ch['segment_start']:.1f}s → {ch['segment_end']:.1f}s ({ch['segment_end']-ch['segment_start']:.1f}s)")
        line("New text", text)
        print(f"    Word count: {len(text.split())}")

        # Reference clip for TTS
        ref_clip = merged[:, ss:se]
        ref_path = os.path.join(OUT_DIR, f"change_{i+1}_ref.wav")
        torchaudio.save(ref_path, ref_clip, sr)

        # Generate TTS
        emotion = ch.get("_emotion", {})
        print(f"    TTS params: temp={emotion.get('temperature', 0.8)} top_p={emotion.get('top_p', 0.8)} chunk={emotion.get('chunk_length', 200)}")

        tts_bytes = fish.generate_speech(
            text=text,
            reference_audio_path=ref_path,
            temperature=emotion.get("temperature"),
            top_p=emotion.get("top_p"),
            chunk_length=emotion.get("chunk_length"),
            seed=i + 42,
        )
        tts_raw_path = os.path.join(OUT_DIR, f"change_{i+1}_tts_raw.wav")
        with open(tts_raw_path, "wb") as f:
            f.write(tts_bytes)

        tts_wf, tts_sr = torchaudio.load(tts_raw_path)
        if tts_sr != sr:
            tts_wf = torchaudio.functional.resample(tts_wf, tts_sr, sr)
        tts_dur_before = tts_wf.shape[1] / sr
        line("TTS raw dur", f"{tts_dur_before:.1f}s")

        # Time-stretch analysis
        ref_dur = ref_clip.shape[1] / sr
        ratio = ref_dur / max(tts_dur_before, 0.01)
        ratio_clamped = max(0.7, min(1.5, ratio))
        print(f"    TTS→Ref ratio: {ratio:.2f} (clamped: {ratio_clamped:.2f})")
        if abs(ratio - 1.0) > 0.5:
            print(f"    ⚠ LARGE TIME STRETCH — may cause speed artifacts!")
        if ratio_clamped != ratio:
            print(f"    ⚠ Ratio clamped from {ratio:.2f} → {ratio_clamped:.2f}")

        # Apply time-stretch
        tts_stretched = synth._time_stretch_to_match(tts_wf, ref_clip)
        tts_dur_after_stretch = tts_stretched.shape[1] / sr
        line("TTS stretched", f"{tts_dur_after_stretch:.1f}s")

        # Post-process
        tts_processed = TTSPostProcessor.process(tts_stretched, ref_clip, sr)
        tts_dur_after_pp = tts_processed.shape[1] / sr
        line("TTS processed", f"{tts_dur_after_pp:.1f}s")
        save_wav(tts_processed, sr, f"change_{i+1}_tts_final.wav")

        # Splice
        merged = synth._splice_segment(merged, tts_processed, ss, se)
        print(f"    ✓ Spliced")

    # Save regenerated full audio
    save_wav(merged, sr, "regenerated_full.wav")

    # Save the changed segments in context
    for i, ch in enumerate(changes):
        ss = int(ch["segment_start"] * sr)
        se = int(ch["segment_end"] * sr)
        ctx_ss = max(0, ss - int(2 * sr))
        ctx_se = min(merged.shape[1], se + int(2 * sr))
        save_wav(merged[:, ctx_ss:ctx_se], sr, f"change_{i+1}_context.wav")
        # Also original for comparison
        save_wav(wf[:, ctx_ss:ctx_se], sr, f"change_{i+1}_original_context.wav")

    # ── RETRANSCRIBE ────────────────────────────────────────────────
    sep("RETRANSCRIBE REGENERATED AUDIO")

    # Retranscribe full audio
    regen_full_path = os.path.join(OUT_DIR, "regenerated_full.wav")
    regen_data = await transcriber.transcribe(
        open(regen_full_path, "rb").read(), language="en"
    )
    regen_raw = (regen_data.get("transcript") or "").strip()
    regen_conf = regen_data.get("confidence", 0)

    line("Raw retranscript", regen_raw)
    line("Confidence", f"{regen_conf:.4f}")

    # Also retranscribe each changed segment individually
    print(f"\n  Per-segment retranscription:")
    for i, ch in enumerate(changes):
        ss = int(ch["segment_start"] * sr)
        se = int(ch["segment_end"] * sr)
        seg_wf = merged[:, ss:se]
        seg_path = os.path.join(OUT_DIR, f"change_{i+1}_regenerated_seg.wav")
        torchaudio.save(seg_path, seg_wf, sr)
        seg_data = await transcriber.transcribe(open(seg_path, "rb").read(), language="en")
        seg_text = (seg_data.get("transcript") or "").strip()
        line(f"  Seg {i+1} raw", seg_text)
        line(f"  Seg {i+1} edit", ch["new_text"])

    # ── CORRECT ─────────────────────────────────────────────────────
    sep("CORRECTION PIPELINE")

    for i, ch in enumerate(changes):
        ss = int(ch["segment_start"] * sr)
        se = int(ch["segment_end"] * sr)
        seg_path = os.path.join(OUT_DIR, f"change_{i+1}_regenerated_seg.wav")
        seg_data = await transcriber.transcribe(open(seg_path, "rb").read(), language="en")
        seg_raw = (seg_data.get("transcript") or "").strip()
        edit_text = ch["new_text"]

        print(f"\n  ── Segment {i+1} Correction ──")
        line("Edit text", edit_text)
        line("Raw Whisper", seg_raw)
        line("Punct edit", count_punct(edit_text))
        line("Punct raw", count_punct(seg_raw))

        restored = restore_punctuation_from_edit(seg_raw, edit_text)
        corrected = correct_whisper_mishearings(restored, edit_text, max_distance=3)

        line("→ Restored", restored, ">>>")
        line("→ Corrected", corrected, ">>>")

        # Word accuracy
        def strip_punct(s):
            return set(re.sub(r"[^\w\s]", "", s).lower().split())

        e_words = strip_punct(edit_text)
        r_words = strip_punct(seg_raw)
        c_words = strip_punct(corrected)

        print(f"    Word accuracy: raw={len(r_words & e_words)}/{len(e_words)}, "
              f"corrected={len(c_words & e_words)}/{len(e_words)}")

        punct_before = count_punct(seg_raw)
        punct_after = count_punct(corrected)
        punct_edit = count_punct(edit_text)
        if punct_edit > 0:
            print(f"    Punctuation: raw={punct_before} → corrected={punct_after} (edit={punct_edit})")

        # Show differences
        if corrected != edit_text:
            cw = corrected.split()
            ew = edit_text.split()
            diffs = []
            for j, (a, b) in enumerate(zip(cw, ew)):
                if a.lower().strip(",.!?;") != b.lower().strip(",.!?;"):
                    diffs.append(f"pos{j}: '{b}' vs '{a}'")
            if diffs:
                print(f"    ⚠ Word differences ({len(diffs)}):")
                for d in diffs[:5]:
                    print(f"      {d}")

    # ── DIAGNOSE TIME-STRETCH ───────────────────────────────────────
    sep("TIME-STRETCH DIAGNOSIS")
    print("  Checking all changes for problematic time-stretch ratios...")
    for i, ch in enumerate(changes):
        ss = int(ch["segment_start"] * sr)
        se = int(ch["segment_end"] * sr)
        ref_dur = ch["segment_end"] - ch["segment_start"]

        tts_path = os.path.join(OUT_DIR, f"change_{i+1}_tts_raw.wav")
        tts_wf, _ = torchaudio.load(tts_path)
        tts_dur = tts_wf.shape[1] / sr
        ratio = ref_dur / max(tts_dur, 0.01)

        status = "OK" if 0.7 <= ratio <= 1.5 else "⚠ CLAMPED"
        if abs(ratio - 1.0) > 0.3:
            status += " (large stretch)"
        print(f"    Change {i+1}: ref={ref_dur:.1f}s  tts_raw={tts_dur:.1f}s  ratio={ratio:.2f}  {status}")

    # Suggest fix if needed
    too_large = any(
        abs(((ch["segment_end"] - ch["segment_start"]) /
             max((torchaudio.load(os.path.join(OUT_DIR, f"change_{i+1}_tts_raw.wav"))[0].shape[1] / sr), 0.01)) - 1.0)
        > 0.5
        for i, ch in enumerate(changes)
    )
    if too_large:
        print("\n  ⚠ Some segments have large time-stretch ratios.")
        print("  Consider: reduce expansion_words so TTS text is closer to original length,")
        print("  or accept the raw TTS duration without stretching.")

    # ── CHOPPING DIAGNOSIS ──────────────────────────────────────────
    sep("CHOPPING DIAGNOSIS")
    print("  Checking for edge artifacts at splice points...")
    for i, ch in enumerate(changes):
        ss = int(ch["segment_start"] * sr)
        se = int(ch["segment_end"] * sr)

        # Check crossfade region energy
        ctx = merged[:, max(0, ss-50):min(merged.shape[1], ss+200)]
        ctx_energy = ctx.abs().mean().item()
        seg_energy = merged[:, ss:min(merged.shape[1], ss+1000)].abs().mean().item()

        line(f"  Splice {i+1}", f"transition energy ratio: {ctx_energy/max(seg_energy,1e-8):.3f}")
        if ctx_energy < seg_energy * 0.1:
            print(f"    ⚠ Possible silence/chopping at splice point")

    # ── SUMMARY ─────────────────────────────────────────────────────
    sep("TEST SUMMARY")
    print(f"  Audio file:        Travelodge Hotel Construction Update.wav")
    print(f"  Duration:          {dur_total:.1f}s")
    print(f"  Original conf:     {orig_conf:.4f}")
    print(f"  Changes detected:  {len(changes)}")
    print(f"  Retranscript conf: {regen_conf:.4f}")
    print(f"  Samples in:        {OUT_DIR}/")
    print(f"\n  Files:")
    for f in sorted(os.listdir(OUT_DIR)):
        fp = os.path.join(OUT_DIR, f)
        print(f"    {f}  ({os.path.getsize(fp)//1024}KB)")

if __name__ == "__main__":
    asyncio.run(main())
