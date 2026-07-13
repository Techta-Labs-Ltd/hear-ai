"""Test that pipeline reconstruction matches direct reconstruction.

Sends the exact same changes through both endpoints and compares the output.
"""
import requests
import time
import json
import uuid
import os

BASE = "http://localhost:8000"
HEADERS = {
    "X-Service-Key": "8c361a305f40bfc53b165b41c274b00fdba004445e8473bc0e04da8d0093aca4",
    "Content-Type": "application/json",
}
OUTPUT_DIR = "/workspace/reconstruct_output"

AUDIO_URL = (
    "https://media.hear.surf/pipeline-source-mp3/"
    "1e75983e-8fa3-4859-ba99-281df885750c/"
    "18249299-2b8f-4a9b-b333-3a1ee3f478b4-6a2633be-d937-4215-9097-ca27896a0346.mp3"
)

changes = [
    {
        "segment_start": 0.0,
        "segment_end": 5.8,
        "new_text": "thought we have this working and that has started a new recording and dropped off from the point of",
        "original_text": "and that has started a new recording and dropped off",
    },
    {
        "segment_start": 12.28,
        "segment_end": 15.88,
        "new_text": "does have permission related issues and bottleneck so either Safari is not compatible, but chrome is",
        "original_text": "so either Safari is not",
    },
    {
        "segment_start": 25.8,
        "segment_end": 26.66,
        "new_text": "all works fine",
        "original_text": "",
    },
    {
        "segment_start": 34.24,
        "segment_end": 34.56,
        "new_text": "safari does have limitation for recording and dont allow some certain types of file",
        "original_text": "",
    },
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("PIPELINE vs DIRECT - SAME PAYLOAD COMPARISON")
print("=" * 70)

# ── 1. Direct endpoint ─────────────────────────────────────────────
print("\n[1] POST /api/v1/reconstruct (direct)")
print("-" * 50)
direct_payload = {
    "audio_url": AUDIO_URL,
    "track_id": f"compare-direct-{uuid.uuid4().hex[:8]}",
    "changes": changes,
    "same_speaker": True,
}

t0 = time.time()
resp1 = requests.post(f"{BASE}/api/v1/reconstruct", json=direct_payload, headers=HEADERS, timeout=600)
elapsed1 = time.time() - t0
print(f"Response ({resp1.status_code}) in {elapsed1:.1f}s")
r1 = resp1.json()
print(json.dumps(r1, indent=2))

if resp1.status_code == 200:
    url1 = r1.get("audio_url", "")
    if url1:
        out1 = os.path.join(OUTPUT_DIR, "pipeline_compare_direct.mp3")
        rd = requests.get(url1, timeout=60)
        if rd.status_code == 200:
            with open(out1, "wb") as f:
                f.write(rd.content)
            print(f"Downloaded: {out1} ({len(rd.content)/1024:.0f} KB)")

# ── 2. Pipeline endpoint ──────────────────────────────────────────
print("\n[2] POST /api/v1/process job_type=reconstruct (pipeline)")
print("-" * 50)
job_id = str(uuid.uuid4())
pipeline_payload = {
    "job_id": job_id,
    "track_id": f"compare-pipeline-{uuid.uuid4().hex[:8]}",
    "job_type": "reconstruct",
    "audio_url": AUDIO_URL,
    "changes": changes,
    "same_speaker": True,
}
print(f"Payload: {json.dumps(pipeline_payload, indent=2)}")

resp2 = requests.post(f"{BASE}/api/v1/process", json=pipeline_payload, headers=HEADERS)
print(f"Response ({resp2.status_code}): {json.dumps(resp2.json(), indent=2)}")

if resp2.status_code == 202:
    print(f"\nPolling job {job_id}...")
    result2 = {}
    for i in range(120):
        time.sleep(3)
        poll = requests.get(f"{BASE}/api/v1/jobs/{job_id}", headers=HEADERS)
        pdata = poll.json()
        status = pdata.get("status", "unknown")
        stage = pdata.get("current_stage", "")
        print(f"  [{i*3}s] status={status}  stage={stage}")
        if status in ("completed", "failed", "cancelled"):
            result2 = pdata.get("result", {})
            print(f"\nFinal result:")
            print(json.dumps(pdata, indent=2, default=str)[:2000])
            break

    if result2:
        audio_info = result2.get("reconstructed_audio", {})
        url2 = audio_info.get("audio_url", "")
        if url2:
            out2 = os.path.join(OUTPUT_DIR, "pipeline_compare_pipeline.mp3")
            rd2 = requests.get(url2, timeout=60)
            if rd2.status_code == 200:
                with open(out2, "wb") as f:
                    f.write(rd2.content)
                print(f"\nDownloaded: {out2} ({len(rd2.content)/1024:.0f} KB)")

# ── Compare ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

d1 = os.path.join(OUTPUT_DIR, "pipeline_compare_direct.mp3")
d2 = os.path.join(OUTPUT_DIR, "pipeline_compare_pipeline.mp3")

if os.path.exists(d1):
    s1 = os.path.getsize(d1)
    print(f"Direct:   {d1} ({s1/1024:.0f} KB)")
if os.path.exists(d2):
    s2 = os.path.getsize(d2)
    print(f"Pipeline: {d2} ({s2/1024:.0f} KB)")

if os.path.exists(d1) and os.path.exists(d2):
    import wave
    with wave.open(d1, "rb") as w1:
        dur1 = w1.getnframes() / w1.getframerate()
    with wave.open(d2, "rb") as w2:
        dur2 = w2.getnframes() / w2.getframerate()
    print(f"\nDirect duration:   {dur1:.1f}s")
    print(f"Pipeline duration: {dur2:.1f}s")
    diff = abs(dur1 - dur2)
    if diff < 5.0:
        print(f"Difference: {diff:.1f}s - GOOD (within 5s tolerance)")
    else:
        print(f"Difference: {diff:.1f}s - WARNING (large difference)")
