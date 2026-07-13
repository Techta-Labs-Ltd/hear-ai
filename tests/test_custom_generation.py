import requests
import time
import json
import uuid

BASE = "http://localhost:8000"
HEADERS = {
    "X-Service-Key": "8c361a305f40bfc53b165b41c274b00fdba004445e8473bc0e04da8d0093aca4",
    "Content-Type": "application/json",
}

# The recording we transcribed
AUDIO_URL = "https://media.hear.surf/pipeline-source-mp3/8ad18866-e1c4-4055-9064-151958b0f8c3/1faaf4fe-717c-4180-a304-1b13743b7d0f-824a7377-e31e-4858-86bb-2505bf2c659e.mp3"

changes = [
    {
        "segment_start": 49.22,
        "segment_end": 52.78,
        "new_text": "Students at Cambridge University who planted five hundred trees",
        "original_text": None
    },
    {
        "segment_start": 69.02,
        "segment_end": 73.78,
        "new_text": "including nearly five thousand saplings at West London Park to form a massive new forest.",
        "original_text": None
    },
    {
        "segment_start": 90.54,
        "segment_end": 96.56,
        "new_text": "The council's dedicated environmental team have planted over ten thousand majestic trees throughout the entire region of Southern England.",
        "original_text": None
    }
]

print("=" * 70)
print("CUSTOM RECONSTRUCT TEST (original_text=None)")
print("Endpoint: POST /api/v1/process (async)")
print("=" * 70)

job_id = str(uuid.uuid4())
pipeline_payload = {
    "job_id": job_id,
    "track_id": f"pipeline-test-{job_id[:8]}",
    "job_type": "reconstruct",
    "audio_url": AUDIO_URL,
    "changes": changes,
    "same_speaker": True
}

resp = requests.post(f"{BASE}/api/v1/process", json=pipeline_payload, headers=HEADERS)
if resp.status_code == 202:
    print(f"\n[1] Submitting job {job_id}...")
    print(f"    Response (202): {json.dumps(resp.json())}")
    print(f"\n[2] Polling job status...")
    
    for i in range(120):
        time.sleep(3)
        poll = requests.get(f"{BASE}/api/v1/jobs/{job_id}", headers=HEADERS)
        data = poll.json()
        status = data.get("status", "unknown")
        stage = data.get("current_stage", "")
        
        print(f"    [{i*3}s] status={status} stage={stage}")
        
        if status in ("completed", "failed", "cancelled"):
            print(f"\n    Final status: {status}")
            if status == "completed":
                r2 = data.get("result", {})
                audio_url = r2.get("audio_url") or data.get("audio_url")
                if not audio_url:
                    print(f"    Wait, full data: {json.dumps(data, indent=2)}")
                print(f"    Audio URL: {audio_url}")
            else:
                print(f"    ERROR: {data.get('error')}")
            break
else:
    print(f"FAILED TO SUBMIT: {resp.status_code}")
    print(resp.text)
