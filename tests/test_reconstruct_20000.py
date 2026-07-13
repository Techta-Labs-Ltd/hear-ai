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

changes = [
    {
        "segment_start": 127.92,
        "segment_end": 128.6,
        "new_text": "and enviroment micro bodies",
        "original_text": None
    },
    {
        "segment_start": 139.62,
        "segment_end": 139.96,
        "new_text": "season, we will be launching a give away in northrumbia",
        "original_text": None
    }
]

AUDIO_URL = "https://media.hear.surf/pipeline-source-mp3/8ad18866-e1c4-4055-9064-151958b0f8c3/1faaf4fe-717c-4180-a304-1b13743b7d0f-824a7377-e31e-4858-86bb-2505bf2c659e.mp3"

print("=" * 70)
print("PIPELINE RECONSTRUCT TEST (data from reconstruct.log)")
print("Endpoint: POST /api/v1/process (async)")
print("=" * 70)

job_id = str(uuid.uuid4())
payload = {
    "job_id": job_id,
    "track_id": f"pipeline-test-{uuid.uuid4().hex[:8]}",
    "job_type": "reconstruct",
    "audio_url": AUDIO_URL,
    "changes": changes,
    "same_speaker": True,
}

print(f"\n[1] Submitting job {job_id}...")
t0 = time.time()
resp = requests.post(f"{BASE}/api/v1/process", json=payload, headers=HEADERS)
print(f"    Response ({resp.status_code}): {json.dumps(resp.json())}")

if resp.status_code != 202:
    print(f"FAILED to submit: {json.dumps(resp.json(), indent=2)}")
    exit(1)

print(f"\n[2] Polling job status...")
for attempt in range(120):
    time.sleep(3)
    poll = requests.get(f"{BASE}/api/v1/jobs/{job_id}", headers=HEADERS)
    pdata = poll.json()
    status = pdata.get("status", "unknown")
    stage = pdata.get("current_stage", "")

    if attempt % 5 == 0 or status in ("completed", "failed", "cancelled"):
        elapsed = time.time() - t0
        print(f"    [{elapsed:.0f}s] status={status} stage={stage}")

    if status in ("completed", "failed", "cancelled"):
        print(f"\n    Final status: {status}")
        if status == "completed":
            result = pdata.get("result", {})
            aurl = result.get("reconstructed_audio", {}).get("audio_url") or result.get("audio_url", "")
            print(f"    Audio URL: {aurl}")
        else:
            print(f"    Error: {pdata.get('error', 'unknown')}")
        break
