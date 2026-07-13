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
print("LONG TEXT RECONSTRUCTION TEST (4 segments, mixed long + short)")
print("=" * 70)
for i, c in enumerate(changes):
    wc = len(c["new_text"].split())
    print(f"  Change {i+1}: {c['segment_start']}-{c['segment_end']}s | {wc} words | \"{c['new_text'][:60]}...\"")

# ── Direct reconstruction ──────────────────────────────────────────
print("\n[1] POST /api/v1/reconstruct (direct, synchronous)")
print("-" * 50)
payload = {
    "audio_url": AUDIO_URL,
    "track_id": f"long-text-test-{uuid.uuid4().hex[:8]}",
    "changes": changes,
    "same_speaker": True,
}
print(f"Payload: {json.dumps(payload, indent=2)}")

t0 = time.time()
resp = requests.post(f"{BASE}/api/v1/reconstruct", json=payload, headers=HEADERS, timeout=600)
elapsed = time.time() - t0

print(f"\nResponse ({resp.status_code}) in {elapsed:.1f}s:")
data = resp.json()
print(json.dumps(data, indent=2))

if resp.status_code == 200:
    audio_url = data.get("audio_url", "")
    if audio_url:
        print(f"\n[2] Downloading result...")
        out_path = os.path.join(OUTPUT_DIR, "long_text_reconstruct.mp3")
        try:
            rd = requests.get(audio_url, timeout=60)
            if rd.status_code == 200:
                with open(out_path, "wb") as f:
                    f.write(rd.content)
                print(f"    Saved: {out_path} ({len(rd.content)/1024:.0f} KB)")
            else:
                print(f"    Download failed: {rd.status_code}")
                print(f"    URL: {audio_url}")
        except Exception as e:
            print(f"    Download error: {e}")
            print(f"    URL: {audio_url}")
else:
    print(f"\nDirect failed, trying pipeline...")
    job_id = str(uuid.uuid4())
    pipeline_payload = {
        "job_id": job_id,
        "track_id": f"long-text-pipeline-{uuid.uuid4().hex[:8]}",
        "job_type": "reconstruct",
        "audio_url": AUDIO_URL,
        "changes": changes,
        "same_speaker": True,
    }
    resp2 = requests.post(f"{BASE}/api/v1/process", json=pipeline_payload, headers=HEADERS)
    print(f"Pipeline response ({resp2.status_code}):")
    print(json.dumps(resp2.json(), indent=2))

    if resp2.status_code == 202:
        print(f"\nPolling job {job_id}...")
        for i in range(120):
            time.sleep(3)
            poll = requests.get(f"{BASE}/api/v1/jobs/{job_id}", headers=HEADERS)
            pdata = poll.json()
            status = pdata.get("status", "unknown")
            stage = pdata.get("current_stage", "")
            print(f"  [{i*3}s] status={status}  stage={stage}")
            if status in ("completed", "failed", "cancelled"):
                print(f"\nFinal result:")
                print(json.dumps(pdata, indent=2, default=str))
                if status == "completed":
                    result = pdata.get("result", {})
                    aurl = result.get("audio_url", "")
                    if aurl:
                        out_path = os.path.join(OUTPUT_DIR, "long_text_reconstruct.mp3")
                        try:
                            rd = requests.get(aurl, timeout=60)
                            if rd.status_code == 200:
                                with open(out_path, "wb") as f:
                                    f.write(rd.content)
                                print(f"\nDownloaded: {out_path} ({len(rd.content)/1024:.0f} KB)")
                        except Exception as e:
                            print(f"Download error: {e}")
                            print(f"URL: {aurl}")
                break

print("\n" + "=" * 70)
print(f"Output dir: {OUTPUT_DIR}/")
print("=" * 70)
