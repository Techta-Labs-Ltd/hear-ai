import requests
import time
import json
import uuid

BASE = "http://localhost:8000"
HEADERS = {"X-Service-Key": "8c361a305f40bfc53b165b41c274b00fdba004445e8473bc0e04da8d0093aca4", "Content-Type": "application/json"}

AUDIO_URL = "https://media.hear.surf/pipeline-source-mp3/1e75983e-8fa3-4859-ba99-281df885750c/18249299-2b8f-4a9b-b333-3a1ee3f478b4-6a2633be-d937-4215-9097-ca27896a0346.mp3"

changes = [
    {
        "segment_start": 9.76,
        "segment_end": 10.08,
        "original_text": "women",
        "new_text": "stay long women"
    },
    {
        "segment_start": 25.32,
        "segment_end": 25.74,
        "original_text": "terracotta",
        "new_text": "sculpture"
    },
    {
        "segment_start": 113.98,
        "segment_end": 113.98,
        "original_text": "",
        "new_text": "found love in coding"
    }
]

print("=" * 70)
print("TEST: Full payload with original_text (replace + insert)")
print("=" * 70)

# 1. Direct endpoint
print("\n[1] POST /api/v1/reconstruct (direct)")
print("-" * 50)
direct_payload = {
    "audio_url": AUDIO_URL,
    "track_id": "test-original-text-direct",
    "changes": changes,
    "same_speaker": True
}
print("PAYLOAD:", json.dumps(direct_payload, indent=2))
t0 = time.time()
resp1 = requests.post(f"{BASE}/api/v1/reconstruct", json=direct_payload, headers=HEADERS)
elapsed1 = time.time() - t0
print(f"\nRESPONSE ({resp1.status_code}) in {elapsed1:.1f}s:")
r1 = resp1.json()
print(json.dumps(r1, indent=2))

# 2. Pipeline endpoint
print("\n\n[2] POST /api/v1/process job_type=reconstruct (pipeline)")
print("-" * 50)
job_id = str(uuid.uuid4())
pipeline_payload = {
    "job_id": job_id,
    "track_id": "test-original-text-pipeline",
    "job_type": "reconstruct",
    "audio_url": AUDIO_URL,
    "changes": changes,
    "same_speaker": True
}
print("PAYLOAD:", json.dumps(pipeline_payload, indent=2))
resp2 = requests.post(f"{BASE}/api/v1/process", json=pipeline_payload, headers=HEADERS)
print(f"\nRESPONSE ({resp2.status_code}):")
print(json.dumps(resp2.json(), indent=2))

if resp2.status_code == 202:
    print(f"\nPolling...")
    for i in range(60):
        time.sleep(3)
        poll = requests.get(f"{BASE}/api/v1/jobs/{job_id}", headers=HEADERS)
        data = poll.json()
        status = data.get("status", "unknown")
        stage = data.get("current_stage", "")
        print(f"  [{i*3}s] status={status}  stage={stage}")
        if status in ("completed", "failed", "cancelled"):
            print(f"\nFINAL RESULT:")
            r2 = data.get("result", {})
            print(json.dumps(r2, indent=2, default=str))
            if status == "failed":
                print(f"\nERROR: {data.get('error')}")
            break

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
