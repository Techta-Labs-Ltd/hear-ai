"""
Large API edit test: regenerate HRA audio with massive edits, then test
the full correction pipeline via the transcription API.
"""
import asyncio, json, os, re, sys, time
import torch, torchaudio
import httpx

sys.path.insert(0, "/workspace/hear-ai")

from app.services.synthesizer import SpeechSynthesizer
from app.services.fishspeech_client import FishSpeechClient
from app.services.enhancer_utils.tts_post_processor import TTSPostProcessor
from app.services.diff_engine import compute_edit_segments, edit_segments_to_changes

AUDIO = "/workspace/hear-ai/HRA Elects New Leaders to Challenge Council.wav"
OUT = "/workspace/hear-ai/large_edit_test"
API = "http://localhost:8000"
SECRET = "8c361a305f40bfc53b165b41c274b00fdba004445e8473bc0e04da8d0093aca4"
H = {"X-Service-Key": SECRET, "Content-Type": "application/json"}

ORIGINAL = (
    "Havering Residents Association, HRA, has elected leaders to challenge, "
    "question and scrutinise reforms leadership of the council. "
    "The local group, who thinks they don't need any mainstream political party, "
    "went to the zoo on a Thursday because it was a good idea they went with "
    "councillor Gillian Ford at its leader and councillor Barry Mugglestone as her deputy. "
    "Councillor Ford was previously deputy leader of the council, "
    "while councillor Mugglestone was cabinet member for the Environment."
)

EDITED = (
    "Havering Residents Association (HRA) has elected new leaders to carefully challenge, "
    "question and examine the administration of the council throughout 2025. "
    "The local reform group, who thinks they don't need any political tactics, "
    "went to the zoo on a Thursday because it was a brilliant concept; "
    "they went with Councillor Gillian Ford as their leader, "
    "and Councillor Barry Mugglestone as her vice-chair. "
    "Councillor Ford was previously chief executive of the council; "
    "meanwhile, Councillor Mugglestone was cabinet secretary for the Environment."
)

CHANGES_MADE = [
    "HRA, → (HRA)",
    "inserted 'new' before leaders",
    "inserted 'carefully' before challenge",
    "scrutinise → examine",
    "reforms → administration",
    "council. → council throughout 2025.",
    "local group → local reform group",
    "mainstream political party → political tactics",
    "good idea → brilliant concept",
    "at its leader → as their leader",
    "deputy → vice-chair",
    "previously deputy leader → previously chief executive",
    "while → meanwhile",
    "cabinet member → cabinet secretary",
]


def sep(s):
    print(f"\n{'='*70}\n  {s}\n{'='*70}")


def ln(l, t):
    print(f"  {l:16s}: {str(t)[:90]}")


def _s(s):
    return set(re.sub(r"[^\w\s]", "", s).lower().split())


def cp(t):
    return sum(1 for c in t if c in ",.!?;")


async def api_transcribe(audio_path: str) -> dict:
    with open(audio_path, "rb") as f:
        data = f.read()
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(
            f"{API}/api/v1/transcribe",
            json={"audio": data.hex()},
            headers=H,
        )
        if r.status_code != 200:
            return {"error": r.text}
        return r.json()


async def main():
    os.makedirs(OUT, exist_ok=True)
    wf, sr = torchaudio.load(AUDIO)
    if sr != 44100:
        wf = torchaudio.functional.resample(wf, sr, 44100)
        sr = 44100

    sep("EDIT SUMMARY")
    print(f"  Changes ({len(CHANGES_MADE)} total):")
    for c in CHANGES_MADE:
        print(f"    - {c}")

    sep("STEP 1: Regenerate audio with all edits")

    synth = SpeechSynthesizer()
    synth.load()
    fish = FishSpeechClient()

    word_segs = [
        {
            "id": 0,
            "start": 1.0,
            "end": 50.0,
            "text": ORIGINAL,
            "words": [
                {"word": w, "start": 1.0 + i * 0.3, "end": 1.0 + i * 0.3 + 0.2, "prob": 0.9}
                for i, w in enumerate(ORIGINAL.split())
            ],
        }
    ]

    segs = compute_edit_segments(ORIGINAL, EDITED, word_segs, expansion_words=3)
    changes = edit_segments_to_changes(segs)

    print(f"\n  Edit segments found: {len(segs)}")
    for i, s in enumerate(segs):
        print(f"    [{i+1}] {s.start_time:.1f}s-{s.end_time:.1f}s ({s.end_time-s.start_time:.1f}s)")
        print(f"        orig: \"{s.original_text[:70]}...\"")
        print(f"        edit: \"{s.edited_text[:70]}...\"")

    merged = wf
    tts_count = 0
    for i, ch in enumerate(changes):
        ss = int(ch["segment_start"] * sr)
        se = int(ch["segment_end"] * sr)
        text = ch["new_text"]
        wc = len(text.split())

        if ch.get("is_deletion"):
            print(f"\n  Change {i+1}: DELETE [{ch['segment_start']:.1f}s-{ch['segment_end']:.1f}s]")
            merged = synth._splice_segment(merged, torch.zeros_like(merged[:, :0]), ss, se)
            continue

        print(f"\n  Change {i+1}: TTS {wc} words [{ch['segment_start']:.1f}s-{ch['segment_end']:.1f}s]")
        ref = merged[:, ss:se]
        ref_path = os.path.join(OUT, f"ref_{i}.wav")
        torchaudio.save(ref_path, ref, sr)

        tts_bytes = fish.generate_speech(
            text=text, reference_audio_path=ref_path,
            seed=42 + i,
        )
        tp = os.path.join(OUT, f"tts_raw_{i}.wav")
        with open(tp, "wb") as f:
            f.write(tts_bytes)
        tts_wf, tsr = torchaudio.load(tp)
        if tsr != sr:
            tts_wf = torchaudio.functional.resample(tts_wf, tsr, sr)

        tts_wf = synth._time_stretch_to_match(tts_wf, ref)
        tts_wf = TTSPostProcessor.process(tts_wf, ref, sr)
        merged = synth._splice_segment(merged, tts_wf, ss, se)
        tts_count += 1
        print(f"    ✓ Spliced ({tts_wf.shape[1]/sr:.1f}s)")

    regen_path = os.path.join(OUT, "regenerated.wav")
    torchaudio.save(regen_path, merged, sr)
    print(f"\n  ✓ Regenerated audio: {regen_path} ({merged.shape[1]/sr:.1f}s)")

    sep("STEP 2: Transcribe regenerated audio via API")
    tx_data = await api_transcribe(regen_path)
    if "error" in tx_data:
        print(f"  API error: {tx_data['error']}")
        return
    regen_raw = tx_data.get("transcript", "").strip()
    conf = tx_data.get("confidence", 0)
    print(f"  Whisper raw:  \"{regen_raw[:120]}...\"")
    print(f"  Confidence:   {conf:.4f}")

    sep("STEP 3: Transcribe ORIGINAL audio for comparison")
    orig_data = await api_transcribe(AUDIO)
    orig_raw = orig_data.get("transcript", "").strip()
    print(f"  Original:     \"{orig_raw[:120]}...\"")

    sep("STEP 4: Correction Pipeline")
    from app.services.diff_engine import restore_punctuation_from_edit, correct_whisper_mishearings

    acc = len(_s(regen_raw) & _s(EDITED)) / max(len(_s(EDITED)), 1)
    print(f"  Whisper accuracy vs edit: {len(_s(regen_raw) & _s(EDITED))}/{len(_s(EDITED))} ({acc:.0%})")

    if acc < 0.4 and len(_s(EDITED)) >= 3:
        print(f"  Accuracy {acc:.0%} < 40% → FALLBACK to edited transcript")
        final = EDITED
        method = "fallback"
    else:
        restored = restore_punctuation_from_edit(regen_raw, EDITED)
        final = correct_whisper_mishearings(restored, EDITED, max_distance=3)
        method = "restore+correct"

    print(f"\n  Method: {method}")
    ln("Edit", EDITED)
    ln("Raw Whisper", regen_raw)
    ln("Corrected", final)

    edit_p = cp(EDITED)
    raw_p = cp(regen_raw)
    final_p = cp(final)
    raw_w = len(_s(regen_raw) & _s(EDITED))
    final_w = len(_s(final) & _s(EDITED))
    total_w = len(_s(EDITED))

    print(f"\n  Punctuation: edit={edit_p} → raw={raw_p} → corrected={final_p}")
    print(f"  Word accuracy: raw={raw_w}/{total_w} ({raw_w/total_w:.0%}) → corrected={final_w}/{total_w} ({final_w/total_w:.0%})")

    sep("VERDICT")

    if final.strip() == EDITED.strip():
        print("  ✓ PERFECT MATCH! Corrected transcript equals edited exactly.")
    else:
        diffs = []
        fw = final.split()
        ew = EDITED.split()
        for i in range(max(len(fw), len(ew))):
            a = fw[i].lower().strip(",.!?;") if i < len(fw) else "MISSING"
            b = ew[i].lower().strip(",.!?;") if i < len(ew) else "EXTRA"
            if a != b:
                diffs.append((i, ew[i] if i < len(ew) else "?", fw[i] if i < len(fw) else "?"))
        print(f"  Differences: {len(diffs)}/{max(len(fw), len(ew))}")
        for pos, e, g in diffs[:8]:
            print(f"    pos {pos}: \"{e}\" → \"{g}\"")

    print(f"\n  Files: {OUT}/")
    for f in sorted(os.listdir(OUT)):
        print(f"    {f}")


if __name__ == "__main__":
    asyncio.run(main())
