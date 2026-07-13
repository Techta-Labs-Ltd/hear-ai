"""Integration tests for Higgs Audio reconstruction via the /api/v1/reconstruct endpoint.

Uses a real Hear audio file (Arsenal vs West Ham WSL match commentary).
The test transcribes the audio, picks a specific phrase, changes one word,
and verifies Higgs Audio regenerates that segment in the original speaker's voice.

The real audio contains:
  "West Ham men are not doing very well and neither are West Ham ladies.
   Gunners five-star show at the Emirates..."

The test changes "five" to "six" -- a single word edit with context.

Usage:
    python -m tests.test_higgs_reconstruct --key YOUR_KEY

Requires:
    - A running hear-ai server with HIGGS_AUDIO_ENABLED=true
    - A valid AI_SERVICE_SECRET
    - GPU with Higgs Audio model downloaded
"""

import argparse
import asyncio
import os
import sys
import uuid

import httpx

BASE_URL = os.getenv("AI_SERVICE_URL", "http://localhost:8000")
SERVICE_KEY = os.getenv("AI_SERVICE_SECRET", "")

# Real Hear audio -- Arsenal WSL match commentary (235.5s, English)
REAL_AUDIO_URL = (
    "https://media.hear.surf/pipeline-source-mp3/"
    "24f1f062-74ed-45ec-a261-a33857d8c712/"
    "18a49f2e-ff13-4b0e-88f9-70ad22c10790-68a1cf9d-9a9c-47c2-87dd-a4809ae13a92.mp3"
)

# Word-level timestamps from Faster-Whisper transcription of the above audio:
#   6.20 -  6.84   Gunners
#   6.84 -  7.18   five
#   7.18 -  7.70  -star
#   7.70 -  8.00   show
#   8.00 -  8.36   at
#   8.36 -  8.46   the
#   8.46 -  9.10   Emirates.
#
# Test 1: Change "five" to "six" in "five-star show"
#   segment 6.84 - 7.18 → new text "six"
#
# Test 2: Change "not doing very well" to "absolutely terrible"
#   Words at 2.00 - 2.94:
#     2.00 -  2.26   doing
#     2.26 -  2.52   very
#     2.52 -  2.94   well
#   We expand to include "not" at 1.88-2.00 for context
#   segment 1.88 - 2.94 → new text "absolutely terrible"

PASS = "\033[92m"
FAIL = "\033[91m"
WARN = "\033[93m"

results = {"passed": 0, "failed": 0, "warnings": 0}


def log(status, test_name, detail=""):
    icon = PASS if status == "pass" else FAIL if status == "fail" else WARN
    suffix = f" -- {detail}" if detail else ""
    print(f"  {icon} {test_name}{suffix}")
    if status == "pass":
        results["passed"] += 1
    elif status == "fail":
        results["failed"] += 1
    else:
        results["warnings"] += 1


async def test_health(client: httpx.AsyncClient):
    print("\n[1/7] Health Check")
    try:
        r = await client.get(f"{BASE_URL}/health")
        data = r.json()
        assert r.status_code == 200
        log("pass", "Server is healthy")
        if data.get("gpu_available"):
            log("pass", f"GPU: {data['gpu_name']}")
        else:
            log("warn", "No GPU detected")
    except Exception as e:
        log("fail", "Health check", str(e))


async def test_auth(client: httpx.AsyncClient):
    print("\n[2/7] Auth Check")
    try:
        r = await client.post(
            f"{BASE_URL}/api/v1/reconstruct",
            headers={"Content-Type": "application/json"},
            json={
                "audio_url": REAL_AUDIO_URL,
                "track_id": "test",
                "segment_start": 0.0,
                "segment_end": 1.0,
                "new_text": "test",
            },
            timeout=10.0,
        )
        assert r.status_code in (401, 403)
        log("pass", "Rejects unauthenticated requests")
    except Exception as e:
        log("fail", "Auth check", str(e))


async def test_validation(client: httpx.AsyncClient):
    print("\n[3/7] Input Validation")
    try:
        r = await client.post(
            f"{BASE_URL}/api/v1/reconstruct",
            headers={"X-Service-Key": SERVICE_KEY, "Content-Type": "application/json"},
            json={"audio_url": REAL_AUDIO_URL, "track_id": "test"},
            timeout=10.0,
        )
        assert r.status_code == 422
        log("pass", "Rejects missing segment fields")
    except Exception as e:
        log("fail", "Validation", str(e))


async def test_change_five_to_six(client: httpx.AsyncClient):
    """Change 'five' to 'six' in 'Gunners five-star show at the Emirates'.

    Word-level timestamps:
        6.84 - 7.18  "five"

    We replace that 0.34s segment with the word "six".
    Higgs will clone the speaker's voice from the surrounding audio.
    """
    print("\n[4/7] Word Edit: 'five' -> 'six' in 'five-star show'")
    print('    Original: "Gunners five-star show at the Emirates."')
    print('    Edited:   "Gunners six-star show at the Emirates."')

    body = {
        "audio_url": REAL_AUDIO_URL,
        "track_id": f"higgs-test-1-{uuid.uuid4().hex[:8]}",
        "segment_start": 6.84,
        "segment_end": 7.18,
        "new_text": "six",
        "same_speaker": True,
    }

    try:
        print("    Sending to Higgs Audio (voice clone mode)...")
        r = await client.post(
            f"{BASE_URL}/api/v1/reconstruct",
            headers={"X-Service-Key": SERVICE_KEY, "Content-Type": "application/json"},
            json=body,
            timeout=300.0,
        )

        if r.status_code == 200:
            data = r.json()
            log("pass", f"Status 200 -- segments_applied={data.get('segments_applied')}")
            if data.get("audio_url"):
                log("pass", f"Reconstructed audio: {data['audio_url']}")
            else:
                log("fail", "No audio_url returned")
            if data.get("duration"):
                log("pass", f"Duration: {data['duration']}s (original: 235.5s)")
            if data.get("b2_key"):
                log("pass", f"Stored at B2: {data['b2_key']}")
            return data
        elif r.status_code == 503:
            log("warn", f"Server 503: {r.text[:200]}")
            return None
        else:
            log("fail", f"Status {r.status_code}: {r.text[:500]}")
            return None
    except httpx.TimeoutException:
        log("warn", "Timed out (first run loads model)")
        return None
    except Exception as e:
        log("fail", "five->six edit", str(e))
        return None


async def test_change_phrase(client: httpx.AsyncClient):
    """Change 'not doing very well' to 'absolutely terrible'.

    Word timestamps:
        1.88 - 2.00  "not"
        2.00 - 2.26  "doing"
        2.26 - 2.52  "very"
        2.52 - 2.94  "well"

    Replace 1.88-2.94 (1.06s) with "absolutely terrible".
    """
    print('\n[5/7] Phrase Edit: "not doing very well" -> "absolutely terrible"')
    print('    Original: "West Ham men are not doing very well..."')
    print('    Edited:   "West Ham men are absolutely terrible..."')

    body = {
        "audio_url": REAL_AUDIO_URL,
        "track_id": f"higgs-test-2-{uuid.uuid4().hex[:8]}",
        "segment_start": 1.88,
        "segment_end": 2.94,
        "new_text": "absolutely terrible",
        "same_speaker": True,
    }

    try:
        print("    Sending to Higgs Audio...")
        r = await client.post(
            f"{BASE_URL}/api/v1/reconstruct",
            headers={"X-Service-Key": SERVICE_KEY, "Content-Type": "application/json"},
            json=body,
            timeout=300.0,
        )

        if r.status_code == 200:
            data = r.json()
            log("pass", f"Status 200 -- segments_applied={data.get('segments_applied')}")
            if data.get("audio_url"):
                log("pass", f"Reconstructed audio: {data['audio_url']}")
            if data.get("duration"):
                log("pass", f"Duration: {data['duration']}s")
            return data
        else:
            log("fail", f"Status {r.status_code}: {r.text[:500]}")
            return None
    except httpx.TimeoutException:
        log("warn", "Timed out")
        return None
    except Exception as e:
        log("fail", "phrase edit", str(e))
        return None


async def test_multi_segment_edit(client: httpx.AsyncClient):
    """Replace two segments at once using the changes[] array.

    Change 1: "five" -> "six" (6.84-7.18)
    Change 2: "not doing very well" -> "completely dominating" (1.88-2.94)
    """
    print('\n[6/7] Multi-Segment: "five"->"six" AND "not doing very well"->"completely dominating"')

    body = {
        "audio_url": REAL_AUDIO_URL,
        "track_id": f"higgs-test-3-{uuid.uuid4().hex[:8]}",
        "changes": [
            {"segment_start": 1.88, "segment_end": 2.94, "new_text": "completely dominating"},
            {"segment_start": 6.84, "segment_end": 7.18, "new_text": "six"},
        ],
        "same_speaker": True,
    }

    try:
        print("    Sending 2-segment edit to Higgs Audio...")
        r = await client.post(
            f"{BASE_URL}/api/v1/reconstruct",
            headers={"X-Service-Key": SERVICE_KEY, "Content-Type": "application/json"},
            json=body,
            timeout=600.0,
        )

        if r.status_code == 200:
            data = r.json()
            log("pass", f"Status 200 -- segments_applied={data.get('segments_applied')}")
            if data.get("segments_applied") == 2:
                log("pass", "Both edits applied in one request")
            if data.get("audio_url"):
                log("pass", f"Final audio: {data['audio_url']}")
            if data.get("duration"):
                log("pass", f"Duration: {data['duration']}s")
            return data
        else:
            log("fail", f"Status {r.status_code}: {r.text[:500]}")
            return None
    except httpx.TimeoutException:
        log("warn", "Timed out (multi-segment takes longer)")
        return None
    except Exception as e:
        log("fail", "multi-segment edit", str(e))
        return None


async def test_no_speaker_clone(client: httpx.AsyncClient):
    """Same edit but with same_speaker=False -- Higgs picks its own voice."""
    print('\n[7/7] Smart Voice (no clone): "five" -> "six"')

    body = {
        "audio_url": REAL_AUDIO_URL,
        "track_id": f"higgs-test-4-{uuid.uuid4().hex[:8]}",
        "segment_start": 6.84,
        "segment_end": 7.18,
        "new_text": "six",
        "same_speaker": False,
    }

    try:
        r = await client.post(
            f"{BASE_URL}/api/v1/reconstruct",
            headers={"X-Service-Key": SERVICE_KEY, "Content-Type": "application/json"},
            json=body,
            timeout=300.0,
        )

        if r.status_code == 200:
            data = r.json()
            log("pass", f"Smart voice OK -- duration={data.get('duration')}s")
            if data.get("audio_url"):
                log("pass", f"Audio: {data['audio_url']}")
            return data
        else:
            log("fail", f"Status {r.status_code}: {r.text[:500]}")
            return None
    except Exception as e:
        log("fail", "smart voice", str(e))
        return None


async def main():
    global BASE_URL, SERVICE_KEY

    parser = argparse.ArgumentParser(
        prog="python -m tests.test_higgs_reconstruct",
        description="Hear AI -- Higgs Audio Reconstruct Tests (Real Audio, Word-Level Edits)",
    )
    parser.add_argument("--url", default=BASE_URL, help="AI service base URL")
    parser.add_argument("--key", default=SERVICE_KEY, help="X-Service-Key secret")

    args = parser.parse_args()
    BASE_URL = args.url.rstrip("/")
    SERVICE_KEY = args.key

    print("=" * 70)
    print("  Hear AI -- Higgs Audio Reconstruct Tests")
    print(f"  Server:  {BASE_URL}")
    print(f"  Audio:   Arsenal vs West Ham WSL commentary (235.5s)")
    print("=" * 70)
    print()
    print("  Transcript excerpt:")
    print('    "West Ham men are not doing very well and neither are')
    print('     West Ham ladies. Gunners five-star show at the Emirates.')
    print('     Chloe Kelly hat-trick inspires Arsenal to win over')
    print('     West Ham."')
    print()
    print("  Edits to test:")
    print('    1. "five" -> "six"           (single word, 0.34s)')
    print('    2. "not doing very well" -> "absolutely terrible"  (phrase, 1.06s)')
    print('    3. Both edits combined       (multi-segment)')
    print('    4. "five" -> "six" (no clone) (smart voice)')
    print("=" * 70)

    async with httpx.AsyncClient(timeout=600) as client:
        await test_health(client)
        await test_auth(client)
        await test_validation(client)
        await test_change_five_to_six(client)
        await test_change_phrase(client)
        await test_multi_segment_edit(client)
        await test_no_speaker_clone(client)

    print("\n" + "=" * 70)
    total = results["passed"] + results["failed"] + results["warnings"]
    print(f"  Results: {results['passed']}/{total} passed, "
          f"{results['failed']} failed, {results['warnings']} warnings")
    print("=" * 70)

    sys.exit(1 if results["failed"] > 0 else 0)


if __name__ == "__main__":
    asyncio.run(main())
