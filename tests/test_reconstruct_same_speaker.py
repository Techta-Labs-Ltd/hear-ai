"""Reconstruction test with real user audio to verify same-speaker voice matching.

Uses a personal narrative audio (Paul talking about his guide dog Rocco).
Edits a specific phrase and downloads the result for inspection.

Usage:
    AI_SERVICE_SECRET=your_key python tests/test_reconstruct_same_speaker.py
"""

import asyncio
import httpx
import os
import sys

BASE_URL = os.getenv("AI_SERVICE_URL", "http://localhost:8000")
SERVICE_KEY = os.getenv("AI_SERVICE_SECRET", "")
OUTPUT_DIR = "/workspace/reconstruct_output"

AUDIO_URL = (
    "https://media.hear.surf/pipeline-source-mp3/"
    "1e75983e-8fa3-4859-ba99-281df885750c/"
    "18249299-2b8f-4a9b-b333-3a1ee3f478b4-6a2633be-d937-4215-9097-ca27896a0346.mp3"
)


async def main():
    if not SERVICE_KEY:
        print("ERROR: Set AI_SERVICE_SECRET env var")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    async with httpx.AsyncClient(timeout=600) as client:

        # Health check
        print("=" * 70)
        print("  Same-Speaker Reconstruction Test")
        print("=" * 70)
        print()
        r = await client.get(f"{BASE_URL}/health")
        health = r.json()
        print(f"[1] Server: {health['status']}, GPU: {health.get('gpu_name')}")
        print(f"    Models: {health.get('models_loaded', [])}")
        assert "higgs_audio" in health.get("models_loaded", []), "Higgs Audio not loaded"

        # Edit 1: "completely hands-free" -> "totally wireless" (phrase edit)
        #
        # Word timestamps:
        #   37.32 - 37.68  completely
        #   37.68 - 38.06  hands
        #   38.06 - 38.36  free
        #
        # Context: "They're brilliant, completely hands-free with tiny speakers..."
        print()
        print("[2] Edit 1: 'completely hands-free' -> 'totally wireless'")
        print('    Original: "They\'re brilliant, completely hands-free with tiny speakers..."')
        print('    Edited:   "They\'re brilliant, totally wireless with tiny speakers..."')
        print("    Segment: 37.32s - 38.36s (1.04s)")

        body1 = {
            "audio_url": AUDIO_URL,
            "track_id": f"same-speaker-test-1-{os.urandom(4).hex()}",
            "segment_start": 37.32,
            "segment_end": 38.36,
            "new_text": "totally wireless",
            "same_speaker": True,
        }

        print("    Sending (voice clone mode)...")
        r1 = await client.post(
            f"{BASE_URL}/api/v1/reconstruct",
            headers={"X-Service-Key": SERVICE_KEY, "Content-Type": "application/json"},
            json=body1,
            timeout=300.0,
        )

        if r1.status_code != 200:
            print(f"    FAILED: {r1.status_code} - {r1.text[:500]}")
            sys.exit(1)

        data1 = r1.json()
        url1 = data1.get("audio_url", "")
        print(f"    Status: 200 OK")
        print(f"    Duration: {data1.get('duration', 0)}s")
        print(f"    B2 key: {data1.get('b2_key', '')}")
        print(f"    B2 URL: {url1}")

        # Download
        out1 = os.path.join(OUTPUT_DIR, "same_speaker_edit1_wireless.mp3")
        rd1 = await client.get(url1, timeout=60.0)
        if rd1.status_code == 200:
            with open(out1, "wb") as f:
                f.write(rd1.content)
            print(f"    Downloaded: {out1} ({len(rd1.content)/1024:.0f} KB)")
        else:
            print(f"    Download failed: {rd1.status_code}")

        # Edit 2: "I have complete confidence in him" -> "I trust him with my life"
        #
        # Word timestamps:
        #   17.04 - 17.20  I
        #   17.20 - 17.46  have
        #   17.46 - 17.90  complete
        #   17.90 - 18.24  confidence
        #   18.24 - 18.46  in
        #   18.46 - 18.46  him
        #
        # Context: "a path he knows well and I have complete confidence in him."
        print()
        print("[3] Edit 2: 'I have complete confidence in him' -> 'I trust him with my life'")
        print('    Original: "...a path he knows well and I have complete confidence in him."')
        print('    Edited:   "...a path he knows well and I trust him with my life."')
        print("    Segment: 17.04s - 18.46s (1.42s)")

        body2 = {
            "audio_url": AUDIO_URL,
            "track_id": f"same-speaker-test-2-{os.urandom(4).hex()}",
            "segment_start": 17.04,
            "segment_end": 18.46,
            "new_text": "I trust him with my life",
            "same_speaker": True,
        }

        print("    Sending (voice clone mode)...")
        r2 = await client.post(
            f"{BASE_URL}/api/v1/reconstruct",
            headers={"X-Service-Key": SERVICE_KEY, "Content-Type": "application/json"},
            json=body2,
            timeout=300.0,
        )

        if r2.status_code != 200:
            print(f"    FAILED: {r2.status_code} - {r2.text[:500]}")
        else:
            data2 = r2.json()
            url2 = data2.get("audio_url", "")
            print(f"    Status: 200 OK")
            print(f"    Duration: {data2.get('duration', 0)}s")
            print(f"    B2 key: {data2.get('b2_key', '')}")
            print(f"    B2 URL: {url2}")

            out2 = os.path.join(OUTPUT_DIR, "same_speaker_edit2_trust.mp3")
            rd2 = await client.get(url2, timeout=60.0)
            if rd2.status_code == 200:
                with open(out2, "wb") as f:
                    f.write(rd2.content)
                print(f"    Downloaded: {out2} ({len(rd2.content)/1024:.0f} KB)")

        # Edit 3: Multi-segment — both edits in one request
        print()
        print("[4] Edit 3: Multi-segment (both edits combined)")
        body3 = {
            "audio_url": AUDIO_URL,
            "track_id": f"same-speaker-test-3-{os.urandom(4).hex()}",
            "changes": [
                {"segment_start": 17.04, "segment_end": 18.46, "new_text": "I trust him with my life"},
                {"segment_start": 37.32, "segment_end": 38.36, "new_text": "totally wireless"},
            ],
            "same_speaker": True,
        }

        print("    Sending 2-segment edit (voice clone)...")
        r3 = await client.post(
            f"{BASE_URL}/api/v1/reconstruct",
            headers={"X-Service-Key": SERVICE_KEY, "Content-Type": "application/json"},
            json=body3,
            timeout=600.0,
        )

        if r3.status_code != 200:
            print(f"    FAILED: {r3.status_code} - {r3.text[:500]}")
        else:
            data3 = r3.json()
            url3 = data3.get("audio_url", "")
            print(f"    Status: 200 OK")
            print(f"    Segments applied: {data3.get('segments_applied', 0)}")
            print(f"    Duration: {data3.get('duration', 0)}s")
            print(f"    B2 key: {data3.get('b2_key', '')}")
            print(f"    B2 URL: {url3}")

            out3 = os.path.join(OUTPUT_DIR, "same_speaker_edit3_multi.mp3")
            rd3 = await client.get(url3, timeout=60.0)
            if rd3.status_code == 200:
                with open(out3, "wb") as f:
                    f.write(rd3.content)
                print(f"    Downloaded: {out3} ({len(rd3.content)/1024:.0f} KB)")

        print()
        print("=" * 70)
        print("  All outputs saved to: " + OUTPUT_DIR)
        print("  Listen and verify:")
        print("    - same_speaker_edit1_wireless.mp3  ('totally wireless' at ~37s)")
        print("    - same_speaker_edit2_trust.mp3     ('I trust him with my life' at ~17s)")
        print("    - same_speaker_edit3_multi.mp3     (both edits combined)")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
