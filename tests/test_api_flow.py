import asyncio, json, time, httpx

BASE = "http://localhost:8000"
SECRET = "8c361a305f40bfc53b165b41c274b00fdba004445e8473bc0e04da8d0093aca4"
HEADERS = {"X-Service-Key": SECRET, "Content-Type": "application/json"}

AUDIO_URL = "file:///workspace/hear-ai/HRA Elects New Leaders to Challenge Council.wav"

EDIT = (
    "Havering Residents Association (HRA) has elected new leaders to carefully challenge, "
    "question and examine the leadership of the council throughout 2026."
)

async def submit(job_id: str, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{BASE}/api/v1/process", json=payload, headers=HEADERS)
        r.raise_for_status()
        return r.json()

async def get_job(job_id: str) -> dict:
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{BASE}/api/v1/jobs/{job_id}", headers=HEADERS)
        return r.json()

async def main():
    print("=" * 70)
    print("  API TEST: edit_transcript → pipeline → correction")
    print("=" * 70)

    # ── STEP 1: Health check ──────────────────────────────────────
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(f"{BASE}/health")
    print(f"\n  Health: {r.json()['status']}")

    # ── STEP 2: Submit edit_transcript job ────────────────────────
    edit_job_id = f"test-edit-{int(time.time())}"
    print(f"\n  [1] Submit edit_transcript job: {edit_job_id}")

    edit_result = await submit(edit_job_id, {
        "job_id": edit_job_id,
        "track_id": "HRA-track",
        "job_type": "edit_transcript",
        "edited_transcript": EDIT,
        "audio_url": AUDIO_URL,
    })
    print(f"  Accepted: {edit_result['status']}")

    # ── STEP 3: Wait for edit_transcript to complete ──────────────
    print(f"\n  [2] Waiting for edit_transcript to complete...")
    for attempt in range(60):
        await asyncio.sleep(3)
        job = await get_job(edit_job_id)
        status = job.get("status", "")
        stage = job.get("current_stage", "")
        print(f"    attempt {attempt+1}: {status} ({stage})", end="\r")
        if status == "completed":
            print(f"\n    ✓ Completed!")
            break
        if status == "failed":
            print(f"\n    ✗ Failed: {job.get('error')}")
            return
    else:
        print(f"\n    Timeout waiting for completion")
        return

    result = job.get("result", {})
    print(f"    changes_detected:  {result.get('changes_detected')}")
    print(f"    is_regenerated:    {result.get('is_regenerated')}")
    print(f"    edited_transcript: {result.get('edited_transcript', '')[:80]}...")
    audio = result.get("reconstructed_audio", {})
    if audio:
        print(f"    audio_url:         {audio.get('audio_url', '')[:60]}...")
        print(f"    duration:          {audio.get('duration')}s")

    # ── STEP 4: Submit pipeline job for same track ────────────────
    pipeline_job_id = f"test-pipe-{int(time.time())}"
    print(f"\n  [3] Submit pipeline job: {pipeline_job_id}")

    pipe_result = await submit(pipeline_job_id, {
        "job_id": pipeline_job_id,
        "track_id": "HRA-track",
        "job_type": "pipeline",
        "edited_transcript": EDIT,
        "audio_url": audio.get("audio_url", AUDIO_URL) if audio else AUDIO_URL,
    })
    print(f"  Accepted: {pipe_result['status']}")

    # ── STEP 5: Wait for pipeline to complete ─────────────────────
    print(f"\n  [4] Waiting for pipeline to complete...")
    for attempt in range(60):
        await asyncio.sleep(3)
        job = await get_job(pipeline_job_id)
        status = job.get("status", "")
        stage = job.get("current_stage", "")
        print(f"    attempt {attempt+1}: {status} ({stage})", end="\r")
        if status == "completed":
            print(f"\n    ✓ Completed!")
            break
        if status == "failed":
            print(f"\n    ✗ Failed: {job.get('error')}")
            return
    else:
        print(f"\n    Timeout")
        return

    result = job.get("result", {})
    tx = result.get("transcription", {})
    whisp = tx.get("transcript", "")
    edited_flag = tx.get("edited", False)
    restored_flag = tx.get("restored", False)
    is_regen = tx.get("is_regenerated", False)
    failed = tx.get("whisper_failed", False)
    conf = tx.get("confidence", 0)
    re_generated = result.get("is_regenerated", False)

    print(f"\n  ── Pipeline Results ──")
    print(f"  Whisper transcript:   {whisp[:100]}...")
    print(f"  Confidence:           {conf:.4f}")
    print(f"  edited:               {edited_flag}")
    print(f"  restored:             {restored_flag}")
    print(f"  whisper_failed:       {failed}")
    print(f"  is_regenerated:       {is_regen}")
    print(f"  result.is_regenerated: {re_generated}")
    print(f"  edited_transcript:    {result.get('edited_transcript', '')[:80]}...")
    mod = result.get("moderation", {})
    if mod:
        print(f"  moderated:            {mod.get('flagged')}")
    cat = result.get("categorization", {})
    if cat:
        print(f"  categories:           {', '.join(str(x) for x in cat.get('categories', [])[:3])}")

    # ── STEP 6: Compare ──────────────────────────────────────────
    print(f"\n  ── Comparison ──")
    print(f"  Edit:     {EDIT[:100]}...")
    print(f"  Whisper:  {whisp[:100]}...")

    import re
    def strip(s):
        return set(re.sub(r"[^\w\s]", "", s).lower().split())
    e_set = strip(EDIT)
    w_set = strip(whisp)
    acc = len(w_set & e_set) / max(len(e_set), 1)
    print(f"  Word accuracy: {len(w_set & e_set)}/{len(e_set)} ({acc:.0%})")

    if whisp.strip() == EDIT.strip():
        print(f"  ✓ PERFECT MATCH!")
    elif acc >= 0.8:
        print(f"  ✓ Good match ({(acc*100):.0f}%)")

if __name__ == "__main__":
    asyncio.run(main())
