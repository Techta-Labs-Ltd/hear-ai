import asyncio
import json
import os
import sys
import uuid

import httpx

BASE_URL    = os.getenv("AI_SERVICE_URL", "http://localhost:8000")
SERVICE_KEY = os.getenv("AI_SERVICE_SECRET", "")
BACKEND_URL = os.getenv("HEAR_BACKEND_URL", "https://dui-metric-phi-clocks.trycloudflare.com")

HEADERS = {
    "X-Service-Key": SERVICE_KEY,
    "Content-Type": "application/json",
}

TEST_RECORDING_ID = "1a62143c-2fb3-4d8f-af4b-22f3e80306e1"
TEST_TRACK_ID     = "bb2ef282-2e22-4406-bfb8-31a9921ed7ef"

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"

results = {"passed": 0, "failed": 0, "warnings": 0}


def log(status, test_name, detail=""):
    icon = PASS if status == "pass" else FAIL if status == "fail" else WARN
    suffix = f" — {detail}" if detail else ""
    print(f"  {icon} {test_name}{suffix}")
    if status == "pass":
        results["passed"] += 1
    elif status == "fail":
        results["failed"] += 1
    else:
        results["warnings"] += 1


async def test_health(client: httpx.AsyncClient):
    print("\n[1/9] Health Check")
    try:
        r = await client.get(f"{BASE_URL}/health")
        data = r.json()
        assert r.status_code == 200, f"status {r.status_code}"
        log("pass", "GET /health returns 200")
        assert data["status"] == "healthy"
        log("pass", "Status is healthy")
        if data["gpu_available"]:
            log("pass", f"GPU available: {data['gpu_name']}")
        else:
            log("warn", "GPU not available — running on CPU")
        for m in ["whisper", "demucs", "categorizer", "moderator"]:
            if m in data.get("models_loaded", []):
                log("pass", f"Model loaded: {m}")
            else:
                log("warn", f"Model not yet loaded: {m}")
    except Exception as e:
        log("fail", "Health check", str(e))


async def test_auth(client: httpx.AsyncClient):
    print("\n[2/9] Authentication")
    try:
        r = await client.post(f"{BASE_URL}/api/v1/moderate", json={"text": "hello"})
        assert r.status_code in (401, 403)
        log("pass", "Rejects request without X-Service-Key")

        r = await client.post(
            f"{BASE_URL}/api/v1/moderate",
            headers={"X-Service-Key": "wrong-key", "Content-Type": "application/json"},
            json={"text": "hello"},
        )
        assert r.status_code in (401, 403)
        log("pass", "Rejects request with wrong key")

        r = await client.post(f"{BASE_URL}/api/v1/moderate", headers=HEADERS, json={"text": "hello"})
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        log("pass", "Accepts valid X-Service-Key")
    except Exception as e:
        log("fail", "Authentication", str(e))


async def test_backend_connectivity(client: httpx.AsyncClient):
    print("\n[3/9] Backend Connectivity & Recording Fetch")
    try:
        r = await client.get(
            f"{BACKEND_URL}/api/v1/internal/recordings/{TEST_RECORDING_ID}",
            headers=HEADERS,
        )
        if r.status_code == 200:
            data = r.json()
            log("pass", f"Recording fetched: '{data.get('title')}'")
            tracks = data.get("tracks", [])
            log("pass" if tracks else "warn", f"Tracks: {len(tracks)}")
            for t in tracks:
                log("pass", f"  track={t['id'][:8]} is_enhanced={t.get('is_enhanced')} transcription={'set' if t.get('transcription') else 'null'}")
        elif r.status_code == 401:
            log("fail", "Backend auth rejected — check AI_SERVICE_SECRET matches backend")
        elif r.status_code == 404:
            log("fail", "Recording not found — check TEST_RECORDING_ID")
        else:
            log("fail", f"Backend returned {r.status_code}: {r.text[:200]}")
    except httpx.ConnectError:
        log("fail", f"Cannot reach backend at {BACKEND_URL} — check HEAR_BACKEND_URL")
    except Exception as e:
        log("fail", "Backend connectivity", str(e))


async def test_pipeline_submit(client: httpx.AsyncClient):
    print("\n[4/9] Pipeline Job Submission")
    job_id = str(uuid.uuid4())
    try:
        r = await client.post(
            f"{BASE_URL}/api/v1/process",
            headers=HEADERS,
            json={
                "job_id": job_id,
                "recording_id": TEST_RECORDING_ID,
                "job_type": "pipeline",
                "max_tags": 5,
            },
        )
        data = r.json()
        assert r.status_code == 202, f"Expected 202, got {r.status_code}: {r.text}"
        assert data["job_id"] == job_id
        log("pass", f"Job accepted: {job_id}")

        await asyncio.sleep(2)

        r2 = await client.get(f"{BASE_URL}/api/v1/jobs/{job_id}", headers=HEADERS)
        job_data = r2.json()
        assert r2.status_code == 200, f"Lookup failed: {r2.text}"
        log("pass", f"Job in DB — status: {job_data['status']}")
        if job_data.get("error"):
            log("warn", f"Early error: {job_data['error']}")

        return job_id
    except Exception as e:
        log("fail", "Pipeline submit", str(e))
        return None


async def test_job_polling(client: httpx.AsyncClient, job_id: str):
    print("\n[5/9] Pipeline Job Completion (polling up to 3 min)")
    if not job_id:
        log("fail", "Skipped — no job_id")
        return None

    max_wait = 180
    poll_interval = 5
    elapsed = 0
    status = "unknown"

    try:
        while elapsed < max_wait:
            r = await client.get(f"{BASE_URL}/api/v1/jobs/{job_id}", headers=HEADERS)
            data = r.json()
            status = data.get("status", "unknown")

            if status == "completed":
                log("pass", f"Job completed in ~{elapsed}s")
                result = data.get("result")
                if isinstance(result, str):
                    result = json.loads(result)

                transcript = None
                if result:
                    tracks = result.get("tracks", {})
                    log("pass" if tracks else "warn", f"Tracks processed: {len(tracks)}")

                    for track_id, track_data in tracks.items():
                        t = track_data.get("transcription", {})
                        if t and t.get("transcript"):
                            transcript = t["transcript"]
                            words = len(transcript.split())
                            lang = t.get("language", "?")
                            log("pass", f"Transcription — {words} words, lang={lang}")
                            log("pass", f"  Preview: {transcript[:120]}...")
                            break

                    if not transcript:
                        log("warn", "No transcription text found in result")

                return transcript

            if status == "failed":
                log("fail", f"Job failed: {data.get('error', 'unknown')}")
                return None

            print(f"    ... status={status} attempt={data.get('attempts')} ({elapsed}s/{max_wait}s)")
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        log("fail", f"Did not complete within {max_wait}s (last: {status})")
        return None
    except Exception as e:
        log("fail", "Job polling", str(e))
        return None


async def test_moderation_from_transcript(client: httpx.AsyncClient, transcript: str):
    print("\n[6/9] Moderation (from real transcript)")
    if not transcript:
        log("warn", "Skipped — no transcript from completed job")
        return

    try:
        r = await client.post(
            f"{BASE_URL}/api/v1/moderate",
            headers=HEADERS,
            json={"text": transcript},
        )
        data = r.json()
        assert r.status_code == 200, f"{r.status_code}: {r.text}"
        flagged = data.get("flagged", False)
        severity = data.get("severity", "?")
        reason = data.get("reason", "")
        log("pass", f"Moderation complete — flagged={flagged} severity={severity}")
        if flagged:
            log("warn", f"Content flagged: {reason!r}")
        else:
            log("pass", "Content is clean")
    except Exception as e:
        log("fail", "Moderation from transcript", str(e))


async def test_categorization_from_transcript(client: httpx.AsyncClient, transcript: str):
    print("\n[7/9] Categorization (from real transcript)")
    if not transcript:
        log("warn", "Skipped — no transcript from completed job")
        return

    try:
        r = await client.post(
            f"{BASE_URL}/api/v1/categorize",
            headers=HEADERS,
            json={"text": transcript, "max_tags": 5},
        )
        data = r.json()
        assert r.status_code == 200, f"{r.status_code}: {r.text}"
        log("pass", f"Tags: {data.get('tags', [])}")
        log("pass", f"Categories: {data.get('categories', [])}")
        log("pass", f"Sentiment: {data.get('sentiment', '?')}")
    except Exception as e:
        log("fail", "Categorization from transcript", str(e))


async def test_idempotency(client: httpx.AsyncClient, job_id: str):
    print("\n[8/9] Idempotency — resubmit same job_id")
    if not job_id:
        log("fail", "Skipped — no job_id")
        return
    try:
        r = await client.post(
            f"{BASE_URL}/api/v1/process",
            headers=HEADERS,
            json={"job_id": job_id, "recording_id": TEST_RECORDING_ID, "job_type": "pipeline", "max_tags": 5},
        )
        assert r.status_code == 202, f"Expected 202, got {r.status_code}: {r.text}"
        log("pass", "Resubmitting same job_id returns 202 (idempotent)")
    except Exception as e:
        log("fail", "Idempotency", str(e))


async def test_edge_cases(client: httpx.AsyncClient):
    print("\n[9/9] Edge Cases")
    try:
        r = await client.get(f"{BASE_URL}/api/v1/jobs/nonexistent-job-id", headers=HEADERS)
        assert r.status_code == 404
        log("pass", "Nonexistent job returns 404")

        r = await client.post(f"{BASE_URL}/api/v1/moderate", headers=HEADERS, json={})
        assert r.status_code == 422
        log("pass", "Missing required fields returns 422")

        r = await client.post(f"{BASE_URL}/api/v1/process", headers=HEADERS, json={"recording_id": TEST_RECORDING_ID})
        assert r.status_code == 422
        log("pass", "Missing job_id returns 422")
    except Exception as e:
        log("fail", "Edge cases", str(e))


async def main():
    global BASE_URL, SERVICE_KEY, BACKEND_URL, HEADERS
    if len(sys.argv) > 1:
        BASE_URL = sys.argv[1].rstrip("/")
    if len(sys.argv) > 2:
        SERVICE_KEY = sys.argv[2]
        HEADERS["X-Service-Key"] = SERVICE_KEY
    if len(sys.argv) > 3:
        BACKEND_URL = sys.argv[3].rstrip("/")

    print("=" * 60)
    print("  Hear AI Service — Integration Tests")
    print(f"  AI Service: {BASE_URL}")
    print(f"  Backend:    {BACKEND_URL}")
    print(f"  Recording:  {TEST_RECORDING_ID}")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=200) as client:
        await test_health(client)
        await test_auth(client)
        await test_backend_connectivity(client)
        job_id = await test_pipeline_submit(client)
        transcript = await test_job_polling(client, job_id)
        await test_moderation_from_transcript(client, transcript)
        await test_categorization_from_transcript(client, transcript)
        await test_idempotency(client, job_id)
        await test_edge_cases(client)

    print("\n" + "=" * 60)
    total = results["passed"] + results["failed"] + results["warnings"]
    print(f"  Results: {results['passed']}/{total} passed, "
          f"{results['failed']} failed, {results['warnings']} warnings")
    print("=" * 60)

    sys.exit(1 if results["failed"] > 0 else 0)


if __name__ == "__main__":
    asyncio.run(main())
