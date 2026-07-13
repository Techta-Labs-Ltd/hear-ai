import requests
import time
import json
import sys

BASE = "http://localhost:8000"
HEADERS = {"X-Service-Key": "8c361a305f40bfc53b165b41c274b00fdba004445e8473bc0e04da8d0093aca4", "Content-Type": "application/json"}

AUDIO_URL = "https://media.hear.surf/pipeline-source-mp3/1e75983e-8fa3-4859-ba99-281df885750c/18249299-2b8f-4a9b-b333-3a1ee3f478b4-6a2633be-d937-4215-9097-ca27896a0346.mp3"
TRACK_ID = "test-reconstruct-live"

print("=" * 70)
print("RECONSTRUCTION API TEST")
print("=" * 70)

# ── 1. Direct reconstruction via /api/v1/reconstruct ──────────────
print("\n[1] POST /api/v1/reconstruct (synchronous)")
print("-" * 50)

payload = {
    "audio_url": AUDIO_URL,
    "track_id": TRACK_ID,
    "changes": [
        {
            "segment_start": 0.0,
            "segment_end": 2.12,
            "new_text": "Praise the Lord, praise the Lord"
        }
    ],
    "same_speaker": True
}

print(f"\nREQUEST PAYLOAD:")
print(json.dumps(payload, indent=2))

print(f"\nSending request...")
t0 = time.time()
resp = requests.post(f"{BASE}/api/v1/reconstruct", json=payload, headers=HEADERS)
elapsed = time.time() - t0

print(f"\nRESPONSE ({resp.status_code}) in {elapsed:.1f}s:")
print(json.dumps(resp.json(), indent=2))

if resp.status_code != 200:
    print("\nDirect reconstruction failed, trying async pipeline...")

# ── 2. Async reconstruction via /api/v1/process ──────────────────
print("\n\n[2] POST /api/v1/process job_type=reconstruct (async)")
print("-" * 50)

import uuid
job_id = str(uuid.uuid4())

async_payload = {
    "job_id": job_id,
    "track_id": TRACK_ID,
    "job_type": "reconstruct",
    "audio_url": AUDIO_URL,
    "changes": [
        {
            "segment_start": 0.0,
            "segment_end": 2.12,
            "new_text": "Praise the Lord, praise the Lord"
        }
    ],
    "same_speaker": True
}

print(f"\nREQUEST PAYLOAD:")
print(json.dumps(async_payload, indent=2))

print(f"\nSending request...")
resp2 = requests.post(f"{BASE}/api/v1/process", json=async_payload, headers=HEADERS)
print(f"\nRESPONSE ({resp2.status_code}):")
print(json.dumps(resp2.json(), indent=2))

if resp2.status_code == 202:
    print(f"\nPolling job status for {job_id}...")
    for i in range(60):
        time.sleep(5)
        poll = requests.get(f"{BASE}/api/v1/jobs/{job_id}", headers=HEADERS)
        data = poll.json()
        status = data.get("status", "unknown")
        stage = data.get("current_stage", "")
        print(f"  [{i*5}s] status={status}  stage={stage}")

        if status in ("completed", "failed", "cancelled"):
            print(f"\nFINAL RESULT:")
            print(json.dumps(data, indent=2))
            break

# ── 3. Edit transcript via /api/v1/edit-transcript ───────────────
print("\n\n[3] POST /api/v1/edit-transcript (async, transcript diff)")
print("-" * 50)

edit_job_id = str(uuid.uuid4())
edit_payload = {
    "job_id": edit_job_id,
    "track_id": TRACK_ID,
    "edited_transcript": "Praise the Lord, praise the Lord, hallelujah, hallelujah, hallelujah, hallelujah, hallelujah, hallelujah, hallelujah, hallelujah, Hallelujah.",
    "same_speaker": True
}

print(f"\nREQUEST PAYLOAD:")
print(json.dumps(edit_payload, indent=2))

print(f"\nSending request...")
resp3 = requests.post(f"{BASE}/api/v1/edit-transcript", json=edit_payload, headers=HEADERS)
print(f"\nRESPONSE ({resp3.status_code}):")
print(json.dumps(resp3.json(), indent=2))

if resp3.status_code == 202:
    print(f"\nPolling job status for {edit_job_id}...")
    for i in range(60):
        time.sleep(5)
        poll = requests.get(f"{BASE}/api/v1/jobs/{edit_job_id}", headers=HEADERS)
        data = poll.json()
        status = data.get("status", "unknown")
        stage = data.get("current_stage", "")
        print(f"  [{i*5}s] status={status}  stage={stage}")

        if status in ("completed", "failed", "cancelled"):
            print(f"\nFINAL RESULT:")
            print(json.dumps(data, indent=2))
            break

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
