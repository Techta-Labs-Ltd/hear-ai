"""
Targeted test: HRA audio with the user's specific edits.

Original: "...scrutinise reforms leadership of the council..."
Edit:     "...scrutinise the leadership of the council throughout 2026."

Tests: word replacement, deletion, insertion of new words not in original.
"""
import asyncio, io, os, re, sys, torch, torchaudio
sys.path.insert(0, "/workspace/hear-ai")

from app.services.diff_engine import (compute_edit_segments, edit_segments_to_changes,
    restore_punctuation_from_edit, correct_whisper_mishearings)
from app.services.fishspeech_client import FishSpeechClient
from app.services.synthesizer import SpeechSynthesizer
from app.services.transcriber import TranscriptionService
from app.services.enhancer_utils.tts_post_processor import TTSPostProcessor

AUDIO = "/workspace/hear-ai/HRA Elects New Leaders to Challenge Council.wav"
OUT = "/workspace/hear-ai/hra_user_test"

def sep(s): print(f"\n{'='*70}\n  {s}\n{'='*70}")
def ln(l, t, m=""): print(f"  {m:4s} {l:16s}: {str(t)[:90]}")
def _s(s): return set(re.sub(r"[^\w\s]", "", s).lower().split())
def cp(t): return sum(1 for c in t if c in ",.!?;")

async def main():
    os.makedirs(OUT, exist_ok=True)
    wf, sr = torchaudio.load(AUDIO)
    if sr != 44100: wf = torchaudio.functional.resample(wf, sr, 44100); sr = 44100
    
    t = TranscriptionService(); t.load()

    # ── Original transcript ────────────────────────────────────────
    orig_data = await t.transcribe(open(AUDIO,"rb").read(), language="en")
    orig_text = (orig_data.get("transcript") or "").strip()
    word_segs = orig_data.get("segments") or []
    
    sep("ORIGINAL WHISPER TRANSCRIPT")
    print(f"  {orig_text}")

    # ── User's edited version ──────────────────────────────────────
    edited = "Havering Residents Association, HRA, has elected leaders to challenge, question and scrutinise the leadership of the council throughout 2026."
    
    sep("USER'S EDITED VERSION")
    print(f"  {edited}")
    
    # Show changes
    print(f"\n  CHANGES FROM ORIGINAL:")
    print(f"    'reforms leadership'  →  'the leadership'     [word replacement]")
    print(f"    No 'throughout 2026'  →  ADDED 'throughout 2026.' [new text insertion]")
    print(f"    Original has extra:    'Former leader Ray Morgan retired...'  [trailing deletion]")

    # ── Diff ───────────────────────────────────────────────────────
    sep("DIFF: WHAT THE ENGINE FOUND")
    segs = compute_edit_segments(
        original_transcript=orig_text,
        edited_transcript=edited,
        word_segments=word_segs,
    )
    changes = edit_segments_to_changes(segs)
    
    print(f"  Edit segments: {len(segs)}")
    for i, s in enumerate(segs):
        print(f"\n  ▸ Segment {i+1}: {s.start_time:.1f}s → {s.end_time:.1f}s ({s.end_time-s.start_time:.1f}s)")
        ln("  Original", s.original_text[:80])
        ln("  Edited", s.edited_text[:80])
        ln("  Is deletion", s.edited_text.strip() == "")

    if not changes:
        print("  No changes!")
        return

    print(f"  Changes: {len(changes)}")

    # ── Regenerate ─────────────────────────────────────────────────
    sep("REGENERATION")
    synth = SpeechSynthesizer(); synth.load()
    fish = FishSpeechClient()
    merged = wf

    for i, ch in enumerate(changes):
        ss = int(ch["segment_start"] * sr); se = int(ch["segment_end"] * sr)
        
        if ch.get("is_deletion"):
            print(f"  Change {i+1}: DELETING [{ch['segment_start']:.1f}s-{ch['segment_end']:.1f}s]")
            merged = synth._splice_segment(merged, torch.zeros_like(merged[:,:0]), ss, se)
            continue

        text = ch["new_text"]; ref = merged[:, ss:se]
        emo = SpeechSynthesizer._analyze_prosody(ref, sr, 0, ref.shape[1])
        
        print(f"\n  Change {i+1}: {ch['segment_start']:.1f}s→{ch['segment_end']:.1f}s ({len(text.split())} words)")
        print(f"    Text: \"{text[:80]}...\"")
        print(f"    Ratio: ref={ch['segment_end']-ch['segment_start']:.1f}s")

        rp = os.path.join(OUT, f"ref_{i}.wav")
        torchaudio.save(rp, ref, sr)
        
        tts_bytes = fish.generate_speech(
            text=text, reference_audio_path=rp,
            temperature=emo.get("temperature"), top_p=emo.get("top_p"),
            chunk_length=emo.get("chunk_length"), seed=42+i,
        )
        tp = os.path.join(OUT, f"tts_raw_{i}.wav")
        with open(tp, "wb") as f: f.write(tts_bytes)
        tts_wf2, tsr2 = torchaudio.load(tp)
        if tsr2 != sr: tts_wf2 = torchaudio.functional.resample(tts_wf2, tsr2, sr)
        
        ratio = (ch["segment_end"]-ch["segment_start"]) / max(tts_wf2.shape[1]/sr, 0.01)
        print(f"    TTS raw: {tts_wf2.shape[1]/sr:.1f}s, ref={ch['segment_end']-ch['segment_start']:.1f}s, ratio={ratio:.2f}")
        
        tts_wf2 = synth._time_stretch_to_match(tts_wf2, ref)
        ref_seg = merged[:, ss:se]
        tts_wf2 = TTSPostProcessor.process(tts_wf2, ref_seg, sr)
        merged = synth._splice_segment(merged, tts_wf2, ss, se)
        print(f"    ✓ Spliced")

    # Save
    regen_path = os.path.join(OUT, "regenerated.wav")
    torchaudio.save(regen_path, merged, sr)
    
    # ── Retranscribe ──────────────────────────────────────────────
    sep("RETRANSCRIBE THE REGENERATED AUDIO")
    regen_data = await t.transcribe(open(regen_path,"rb").read(), language="en")
    regen_raw = (regen_data.get("transcript") or "").strip()
    
    # Also retranscribe just the changed segments
    seg_results = []
    for i, ch in enumerate(changes):
        ss = int(ch["segment_start"] * sr); se = int(ch["segment_end"] * sr)
        ctx_s = max(0, ss - int(2*sr)); ctx_e = min(merged.shape[1], se + int(2*sr))
        seg_wf = merged[:, ctx_s:ctx_e]
        sp = os.path.join(OUT, f"seg_{i}_context.wav")
        torchaudio.save(sp, seg_wf, sr)
        sd = await t.transcribe(open(sp,"rb").read(), language="en")
        seg_results.append((sd.get("transcript") or "").strip())

    # ── Correct ───────────────────────────────────────────────────
    sep("CORRECTION PIPELINE")
    
    restored = restore_punctuation_from_edit(regen_raw, edited)
    corrected = correct_whisper_mishearings(restored, edited, max_distance=3)
    
    acc = len(_s(regen_raw) & _s(edited)) / max(len(_s(edited)), 1)
    if acc < 0.5:
        corrected = edited
        print(f"  ⚠ WHISPER FAILED ({acc:.0%}) — using edited transcript")
    
    ln("Edited", edited)
    ln("Raw Whisper", regen_raw)
    ln("→ Restored", restored, ">>>")
    ln("→ Corrected", corrected, ">>>")
    
    print(f"\n  Punctuation: edit={cp(edited)}  raw={cp(regen_raw)}  restored={cp(restored)}  final={cp(corrected)}")
    print(f"  Word accuracy: raw={len(_s(regen_raw)&_s(edited))}/{len(_s(edited))}  final={len(_s(corrected)&_s(edited))}/{len(_s(edited))}")
    print(f"  Confidence: {regen_data.get('confidence', 0):.4f}")
    
    # Per-segment
    print(f"\n  Per-segment retranscription:")
    for i, (ch, seg_txt) in enumerate(zip(changes, seg_results)):
        edit_t = ch.get("new_text") or "(deleted)"
        acc_s = len(_s(seg_txt) & _s(edit_t)) / max(len(_s(edit_t)), 1) if ch.get("new_text") else 1.0
        s = "✓" if acc_s >= 0.5 else "⚠ FAIL"
        print(f"    Seg {i+1} {s} ({acc_s:.0%}): \"{seg_txt[:70]}\"")
    
    # ── Verdict ───────────────────────────────────────────────────
    sep("VERDICT")
    if corrected == edited:
        print("  ✓ PERFECT MATCH!")
    else:
        cw, ew = corrected.split(), edited.split()
        diffs = [(j, ew[j], cw[j] if j < len(cw) else "?") for j in range(min(len(ew), len(cw)))
                 if cw[j].lower().strip(",.!?;") != ew[j].lower().strip(",.!?;")]
        if diffs:
            print(f"  {len(diffs)} differences:")
            for p, e, g in diffs[:8]:
                print(f"    pos {p}: \"{e}\" → \"{g}\"")

if __name__ == "__main__":
    asyncio.run(main())
