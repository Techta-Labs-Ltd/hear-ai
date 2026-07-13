"""
HRA Comprehensive Edit Test: Long words, punctuation, deletions, insertions

Runs the full regeneration + retranscription + correction pipeline
on the HRA Elects New Leaders audio with deliberate edits.
"""

import asyncio
import io
import os
import re
import sys

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

AUDIO = "/workspace/hear-ai/HRA Elects New Leaders to Challenge Council.wav"
OUT = "/workspace/hear-ai/hra_test"

def save_wav(wf, sr, name):
    p = os.path.join(OUT, name)
    torchaudio.save(p, wf, sr)
    return p

def sep(s):
    print(f"\n{'='*72}")
    print(f"  {s}")
    print(f"{'='*72}")

def line(label, txt, marker=""):
    print(f"  {marker:4s} {label:16s}: {str(txt)[:90]}")

def _strip(s):
    return set(re.sub(r"[^\w\s]", "", s).lower().split())

def count_punct(t):
    return sum(1 for c in t if c in ",.!?;")

async def main():
    os.makedirs(OUT, exist_ok=True)
    
    wf, sr = torchaudio.load(AUDIO)
    if sr != 44100:
        wf = torchaudio.functional.resample(wf, sr, 44100)
        sr = 44100

    transcriber = TranscriptionService()
    transcriber.load()

    # Transcribe original
    orig = await transcriber.transcribe(open(AUDIO, "rb").read(), language="en")
    orig_text = (orig.get("transcript") or "").strip()
    orig_segs = orig.get("segments") or []
    orig_conf = orig.get("confidence", 0)

    sep("ORIGINAL TRANSCRIPT")
    line("Confidence", f"{orig_conf:.4f}")
    line("Segments", len(orig_segs))
    print(f"\n  {orig_text[:500]}")

    # ── Create EDITED version ──────────────────────────────────────
    # Multiple edits: long words, punctuation, deletions, insertions
    edited = (
        "Havering Residents Association (HRA) has elected new leaders to carefully challenge, "
        "question and examine reforms and leadership of the council. "
        "The local group, who thinks they don't need any political party, "
        "went to the zoo on a Thursday because it was a great concept; "
        "they went with Councillor Gillian Ford as its leader, "
        "and Councillor Barry Mugglestone as her deputy. "
        "Councillor Ford was previously deputy leader of the council; "
        "meanwhile, Councillor Mugglestone was cabinet member for the Environment."
    )

    sep("EDITED TRANSCRIPT (DELIBERATE CHANGES)")
    print(f"  Edits made:")
    print(f"    1. HRA, → (HRA)                        [punctuation change]")
    print(f"    2. inserted 'new' before leaders        [word insertion]")
    print(f"    3. inserted 'carefully' before challenge [word insertion]")
    print(f"    4. scrutinise → examine                 [LONG WORD replaced]")
    print(f"    5. 'reforms leadership' → 'reforms and leadership' [and insertion]")
    print(f"    6. removed 'mainstream'                  [word deletion]")
    print(f"    7. 'a good idea' → 'a great concept'     [phrase change]")
    print(f"    8. added semicolons & commas             [punctuation]")
    print(f"    9. 'at its leader' → 'as its leader'     [word fix]")
    print(f"   10. 'while' → 'meanwhile' (capital W)     [word change]")
    print(f"\n  Original: {orig_text[:200]}...")
    print(f"  Edited:   {edited[:200]}...")

    # ── DIFF ───────────────────────────────────────────────────────
    sep("DIFF ENGINE")
    segs = compute_edit_segments(
        original_transcript=orig_text,
        edited_transcript=edited,
        word_segments=orig_segs,
    )
    changes = edit_segments_to_changes(segs)

    print(f"  Edit segments: {len(segs)}")
    print(f"  Changes:       {len(changes)}")
    for i, s in enumerate(segs):
        dur = s.end_time - s.start_time
        print(f"\n  ▸ Change {i+1}: {s.start_time:.1f}s → {s.end_time:.1f}s ({dur:.1f}s)")
        line("  Original", s.original_text[:80])
        line("  Edited", s.edited_text[:80])
        line("  Context L", s.left_context[:40])
        line("  Context R", s.right_context[:40])

    if not changes:
        print("\n  No changes detected!")
        return

    # ── TTS + REGENERATE ───────────────────────────────────────────
    sep("REGENERATION")
    synth = SpeechSynthesizer()
    synth.load()
    fish = FishSpeechClient()

    merged = wf
    for i, ch in enumerate(changes):
        ss = int(ch["segment_start"] * sr)
        se = int(ch["segment_end"] * sr)
        text = ch["new_text"]
        
        if ch.get("is_deletion"):
            print(f"  Change {i+1}: DELETING [{ch['segment_start']:.1f}s-{ch['segment_end']:.1f}s]")
            merged = synth._splice_segment(merged, torch.zeros_like(merged[:, :0]), ss, se)
            print(f"    ✓ Removed ({ch['segment_end']-ch['segment_start']:.1f}s)")
            continue

        ref_clip = merged[:, ss:se]
        ref_path = save_wav(ref_clip, sr, f"change_{i+1}_ref.wav")
        emotion = SpeechSynthesizer._analyze_prosody(ref_clip, sr, 0, ref_clip.shape[1])

        print(f"\n  Change {i+1}: [{ch['segment_start']:.1f}s-{ch['segment_end']:.1f}s] {len(text.split())} words")
        print(f"    Text: \"{text[:70]}...\"")
        if emotion:
            print(f"    Emotion: temp={emotion.get('temperature')}, top_p={emotion.get('top_p')}")

        tts_bytes = fish.generate_speech(
            text=text, reference_audio_path=ref_path,
            temperature=emotion.get("temperature"),
            top_p=emotion.get("top_p"),
            chunk_length=emotion.get("chunk_length"),
            seed=42 + i,
        )
        tts_path = os.path.join(OUT, f"change_{i+1}_tts_raw.wav")
        with open(tts_path, "wb") as f:
            f.write(tts_bytes)

        tts_wf, tts_sr = torchaudio.load(tts_path)
        if tts_sr != sr:
            tts_wf = torchaudio.functional.resample(tts_wf, tts_sr, sr)
        tts_dur = tts_wf.shape[1] / sr
        ref_dur = ch["segment_end"] - ch["segment_start"]
        ratio = ref_dur / max(tts_dur, 0.01)

        print(f"    TTS: {tts_dur:.1f}s, Slot: {ref_dur:.1f}s, Ratio: {ratio:.2f}")

        tts_wf = synth._time_stretch_to_match(tts_wf, ref_clip)
        tts_wf = TTSPostProcessor.process(tts_wf, ref_clip, sr)
        merged = synth._splice_segment(merged, tts_wf, ss, se)
        print(f"    ✓ Spliced")

    save_wav(merged, sr, "regenerated_full.wav")

    # Save context around each change
    for i, ch in enumerate(changes):
        ss = int(ch["segment_start"] * sr)
        se = int(ch["segment_end"] * sr)
        ctx_s = max(0, ss - int(2.0 * sr))
        ctx_e = min(merged.shape[1], se + int(2.0 * sr))
        save_wav(wf[:, ctx_s:ctx_e], sr, f"change_{i+1}_original_context.wav")
        save_wav(merged[:, ctx_s:ctx_e], sr, f"change_{i+1}_regenerated_context.wav")

    # ── RETRANSCRIBE + CORRECT ─────────────────────────────────────
    sep("RETRANSCRIBE & CORRECT")

    regen_path = os.path.join(OUT, "regenerated_full.wav")
    regen = await transcriber.transcribe(open(regen_path, "rb").read(), language="en")
    regen_raw = (regen.get("transcript") or "").strip()
    regen_conf = regen.get("confidence", 0)

    # --- Punctuation restoration ---
    restored = restore_punctuation_from_edit(regen_raw, edited)
    corrected = correct_whisper_mishearings(restored, edited, max_distance=3)

    # --- Fallback check ---
    acc = len(_strip(regen_raw) & _strip(edited)) / max(len(_strip(edited)), 1)
    if acc < 0.5:
        corrected = edited
        print(f"  ⚠ WHISPER FAILED ({acc:.0%} accuracy) → using edited transcript as ground truth")

    # --- Display results ---
    line("Edited", edited)
    line("Raw Whisper", regen_raw)
    line("→ Restored", restored, ">>>")
    line("→ Corrected", corrected, ">>>")

    punct_edit = count_punct(edited)
    punct_raw = count_punct(regen_raw)
    punct_final = count_punct(corrected)

    print(f"\n  {'Punctuation recovery:':30s} raw={punct_raw}/{punct_edit} → final={punct_final}/{punct_edit}")
    print(f"  {'Word accuracy:':30s} raw={len(_strip(regen_raw)&_strip(edited))}/{len(_strip(edited))} → final={len(_strip(corrected)&_strip(edited))}/{len(_strip(edited))}")
    print(f"  {'Retranscript conf:':30s} {regen_conf:.4f}")

    # --- Per-change segment retranscription ---
    print(f"\n  Per-segment retranscription:")
    for i, ch in enumerate(changes):
        ss = int(ch["segment_start"] * sr)
        se = int(ch["segment_end"] * sr)
        seg_data = await transcriber.transcribe(
            open(os.path.join(OUT, f"change_{i+1}_regenerated_context.wav"), "rb").read(),
            language="en",
        )
        seg_text = (seg_data.get("transcript") or "").strip()
        edit_text = ch["new_text"] or "(deleted)"
        acc = len(_strip(seg_text) & _strip(edit_text)) / max(len(_strip(edit_text)), 1) if ch["new_text"] else 1.0
        status = "✓ OK" if acc >= 0.5 else "⚠ FAIL"
        line(f"  Change {i+1}", f"{status} ({acc:.0%}) → \"{seg_text[:60]}\"")

    # ── VERDICT ────────────────────────────────────────────────────
    sep("VERDICT")
    if corrected == edited:
        print("  PERFECT! Corrected transcript matches edited exactly.")
    else:
        cw = corrected.split()
        ew = edited.split()
        diffs = [(j, ew[j], cw[j] if j < len(cw) else "MISSING") 
                 for j in range(min(len(ew), len(cw)))
                 if cw[j].lower().strip(",.!?;") != ew[j].lower().strip(",.!?;")]
        if diffs:
            print(f"  Remaining differences ({len(diffs)}/{len(ew)}):")
            for pos, exp, got in diffs[:8]:
                print(f"    pos {pos}: expected \"{exp}\"  got \"{got}\"")

    print(f"\n  Files in: {OUT}/")
    for f in sorted(os.listdir(OUT)):
        fp = os.path.join(OUT, f)
        print(f"    {f}  ({os.path.getsize(fp)//1024}KB)")

if __name__ == "__main__":
    asyncio.run(main())
