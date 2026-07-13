"""Run a single reconstruction edit and download the result from Backblaze.

Usage:
    AI_SERVICE_SECRET=your_key python tests/test_reconstruct_and_download.py
"""

import asyncio
import httpx
import os
import sys

BASE_URL = os.getenv("AI_SERVICE_URL", "http://localhost:8000")
SERVICE_KEY = os.getenv("AI_SERVICE_SECRET", "")

# Arsenal WSL commentary audio
REAL_AUDIO_URL = (
    "https://media.hear.surf/pipeline-source-mp3/"
    "24f1f062-74ed-45ec-a261-a33857d8c712/"
    "18a49f2e-ff13-4b0e-88f9-70ad22c10790-68a1cf9d-9a9c-47c2-87dd-a4809ae13a92.mp3"
)

OUTPUT_DIR = "/workspace/reconstruct_output"


async def main():
    if not SERVICE_KEY:
        print("ERROR: Set AI_SERVICE_SECRET env var")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    async with httpx.AsyncClient(timeout=600) as client:
        # Health check
        print("[1] Health check...")
        r = await client.get(f"{BASE_URL}/health")
        health = r.json()
        print(f"    Server: {health['status']}, GPU: {health.get('gpu_name', 'N/A')}")
        print(f"    Models: {health.get('models_loaded', [])}")
        if "higgs_audio" not in health.get("models_loaded", []):
            print("    WARNING: higgs_audio not loaded!")

        # Test 1: Single word edit "five" -> "six"
        print("\n[2] Reconstructing: 'five' -> 'six' (voice clone mode)...")
        print(f"    Audio source: Arsenal WSL commentary (235.5s)")
        print(f"    Edit segment: 6.84s - 7.18s (0.34s word)")

        body = {
            "audio_url": REAL_AUDIO_URL,
            "track_id": f"inspect-test-{os.urandom(4).hex()}",
            "segment_start": 6.84,
            "segment_end": 7.18,
            "new_text": "six",
            "same_speaker": True,
        }

        r = await client.post(
            f"{BASE_URL}/api/v1/reconstruct",
            headers={"X-Service-Key": SERVICE_KEY, "Content-Type": "application/json"},
            json=body,
            timeout=300.0,
        )

        if r.status_code != 200:
            print(f"    FAILED: Status {r.status_code}")
            print(f"    Response: {r.text[:500]}")
            sys.exit(1)

        data = r.json()
        audio_url = data.get("audio_url", "")
        b2_key = data.get("b2_key", "")
        duration = data.get("duration", 0)
        segments = data.get("segments_applied", 0)

        print(f"    Status: 200 OK")
        print(f"    Segments applied: {segments}")
        print(f"    Duration: {duration}s")
        print(f"    B2 key: {b2_key}")
        print(f"    B2 URL: {audio_url}")

        # Download the result
        print(f"\n[3] Downloading result from Backblaze...")
        out_path = os.path.join(OUTPUT_DIR, "reconstruct_word_edit.mp3")
        r = await client.get(audio_url, timeout=60.0)
        if r.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(r.content)
            size_kb = len(r.content) / 1024
            print(f"    Saved: {out_path} ({size_kb:.1f} KB)")
        else:
            print(f"    Download failed: {r.status_code}")

        # Test 2: Phrase edit "not doing very well" -> "absolutely terrible"
        print("\n[4] Reconstructing: 'not doing very well' -> 'absolutely terrible'...")
        body2 = {
            "audio_url": REAL_AUDIO_URL,
            "track_id": f"inspect-test-{os.urandom(4).hex()}",
            "segment_start": 1.88,
            "segment_end": 2.94,
            "new_text": "absolutely terrible",
            "same_speaker": True,
        }

        r2 = await client.post(
            f"{BASE_URL}/api/v1/reconstruct",
            headers={"X-Service-Key": SERVICE_KEY, "Content-Type": "application/json"},
            json=body2,
            timeout=300.0,
        )

        if r2.status_code == 200:
            data2 = r2.json()
            audio_url2 = data2.get("audio_url", "")
            print(f"    Status: 200 OK")
            print(f"    Duration: {data2.get('duration', 0)}s")
            print(f"    B2 URL: {audio_url2}")

            out_path2 = os.path.join(OUTPUT_DIR, "reconstruct_phrase_edit.mp3")
            r2d = await client.get(audio_url2, timeout=60.0)
            if r2d.status_code == 200:
                with open(out_path2, "wb") as f:
                    f.write(r2d.content)
                size_kb = len(r2d.content) / 1024
                print(f"    Saved: {out_path2} ({size_kb:.1f} KB)")
        else:
            print(f"    FAILED: {r2.status_code} - {r2.text[:300]}")

    print(f"\nDone. Output files in: {OUTPUT_DIR}/")
    for f in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath) / 1024
        print(f"  {f} ({size:.1f} KB)")


if __name__ == "__main__":
    asyncio.run(main())
