"""
Hear AI Service - Integration Test Suite
Run: python -m tests.test_api <BASE_URL> <SERVICE_KEY>
"""
import asyncio
import json
import sys
import time
import uuid

import httpx

BASE_URL = "http://localhost:8000"
SERVICE_KEY = "change-me"

HEADERS = {
    "X-Service-Key": SERVICE_KEY,
    "Content-Type": "application/json",
}

TEST_RECORDING_ID = "1a62143c-2fb3-4d8f-af4b-22f3e80306e1"
TEST_TRACK_ID     = "bb2ef282-2e22-4406-bfb8-31a9921ed7ef"
TEST_AUDIO_URL    = "http://media.hear.surf/uploads/09a072e9-ed1e-4a04-8a71-2a2960236531/20260416041448_8f4de002.wav"

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
    print("\n[1/8] Health Check")
    try:
        r = await client.get(f"{BASE_URL}/health")
        data = r.json()
        assert r.status_code == 200, f"status {r.status_code}"
        log("pass", "GET /health returns 200")

        assert data["status"] == "healthy", f"status={data['status']}"
        log("pass", "Status is healthy")

        if data["gpu_available"]:
            log("pass", f"GPU available: {data['gpu_name']}")
        else:
            log("warn", "GPU not available — running on CPU")

        models = data.get("models_loaded", [])
        for m in ["whisper", "demucs", "categorizer", "moderator"]:
            if m in models:
                log("pass", f"Model loaded: {m}")
            else:
                log("warn", f"Model not yet loaded: {m}")
    except Exception as e:
        log("fail", "Health check", str(e))


async def test_auth(client: httpx.AsyncClient):
    print("\n[2/8] Authentication")
    try:
        r = await client.post(f"{BASE_URL}/api/v1/moderate", json={"text": "hello"})
        assert r.status_code in (401, 403), f"Expected 401/403 without key, got {r.status_code}"
        log("pass", "Rejects request without X-Service-Key")

        r = await client.post(
            f"{BASE_URL}/api/v1/moderate",
            headers={"X-Service-Key": "wrong-key", "Content-Type": "application/json"},
            json={"text": "hello"},
        )
        assert r.status_code in (401, 403), f"Expected 401/403 with wrong key, got {r.status_code}"
        log("pass", "Rejects request with wrong key")

        r = await client.post(f"{BASE_URL}/api/v1/moderate", headers=HEADERS, json={"text": "hello"})
        assert r.status_code == 200, f"Expected 200 with valid key, got {r.status_code}: {r.text}"
        log("pass", "Accepts request with valid X-Service-Key")
    except Exception as e:
        log("fail", "Authentication", str(e))


async def test_moderation(client: httpx.AsyncClient):
    print("\n[3/8] Moderation")
    try:
        r = await client.post(
            f"{BASE_URL}/api/v1/moderate",
            headers=HEADERS,
            json={"text": "Welcome everyone to today's podcast. We're going to talk about music production and sound design."},
        )
        data = r.json()
        assert r.status_code == 200, f"status {r.status_code}: {r.text}"
        assert data["flagged"] is False, f"Clean content was flagged: {data}"
        log("pass", f"Clean content not flagged (severity={data.get('severity', '?')})")

        r = await client.post(
            f"{BASE_URL}/api/v1/moderate",
            headers=HEADERS,
            json={"text": "I will kill you and destroy everything you love, you stupid worthless piece of trash"},
        )
        data = r.json()
        assert r.status_code == 200, f"status {r.status_code}"
        assert data["flagged"] is True, f"Toxic content not flagged: {data}"
        log("pass", f"Toxic content flagged (severity={data.get('severity', '?')}, reason={data.get('reason', '?')!r})")

        assert "severity" in data
        log("pass", f"Severity field present: {data['severity']}")

        assert "reason" in data
        log("pass", f"Reason field present: {data['reason']!r}")

        r = await client.post(
            f"{BASE_URL}/api/v1/moderate",
            headers=HEADERS,
            json={"text": ""},
        )
        data = r.json()
        assert r.status_code == 200
        assert data["flagged"] is False
        log("pass", "Empty text handled — not flagged")

    except Exception as e:
        log("fail", "Moderation", str(e))


async def test_categorization(client: httpx.AsyncClient):
    print("\n[4/8] Categorization")
    try:
        transcript = (
            "Today we're going to talk about the latest developments in artificial intelligence. "
            "Machine learning models have been getting significantly better at understanding natural language. "
            "Companies like Google and OpenAI are pushing the boundaries of what's possible with transformer architectures. "
            "The implications for healthcare, education, and business are enormous."
        )
        r = await client.post(
            f"{BASE_URL}/api/v1/categorize",
            headers=HEADERS,
            json={"text": transcript, "max_tags": 5},
        )
        data = r.json()
        assert r.status_code == 200, f"status {r.status_code}: {r.text}"
        log("pass", "POST /api/v1/categorize returns 200")

        assert "tags" in data and isinstance(data["tags"], list)
        log("pass", f"Tags returned: {data['tags']}")

        assert "sentiment" in data
        log("pass", f"Sentiment: {data['sentiment']}")

    except Exception as e:
        log("fail", "Categorization", str(e))


async def test_pipeline_submit(client: httpx.AsyncClient):
    print("\n[5/8] Pipeline Job Submission (real recording)")
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
        log("pass", f"Pipeline job accepted: {job_id}")

        await asyncio.sleep(2)

        r2 = await client.get(f"{BASE_URL}/api/v1/jobs/{job_id}", headers=HEADERS)
        job_data = r2.json()
        assert r2.status_code == 200, f"Job status check failed: {r2.text}"
        log("pass", f"Job found in DB — status: {job_data['status']} recording: {job_data['recording_id']}")

        return job_id
    except Exception as e:
        log("fail", "Pipeline submit", str(e))
        return None


async def test_job_polling(client: httpx.AsyncClient, job_id: str):
    print("\n[6/8] Pipeline Job Completion (polling up to 3 min)")
    if not job_id:
        log("fail", "Skipped — no job_id from previous test")
        return

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
                if result:
                    tracks = result.get("tracks", {})
                    log("pass" if tracks else "warn", f"Tracks enhanced: {list(tracks.keys())}")
                    if result.get("transcription"):
                        log("pass", "Transcription present")
                    else:
                        log("warn", "No transcription in result")
                    if result.get("categorization"):
                        log("pass", f"Tags: {result['categorization'].get('tags', [])}")
                    else:
                        log("warn", "No categorization data")
                    if result.get("moderation"):
                        log("pass", f"Moderation — flagged: {result['moderation'].get('flagged')}")
                    else:
                        log("warn", "No moderation data")
                else:
                    log("warn", "Job completed but result_json is empty")
                return

            if status == "failed":
                log("fail", f"Job failed: {data.get('error', 'unknown')}")
                return

            print(f"    ... status={status} attempt={data.get('attempts')} ({elapsed}s/{max_wait}s)")
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        log("fail", f"Job did not complete within {max_wait}s (last status: {status})")
    except Exception as e:
        log("fail", "Job polling", str(e))


async def test_idempotency(client: httpx.AsyncClient, job_id: str):
    print("\n[7/8] Idempotency — resubmit same job_id")
    if not job_id:
        log("fail", "Skipped — no job_id from previous test")
        return
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
        assert r.status_code == 202, f"Expected 202, got {r.status_code}: {r.text}"
        log("pass", "Resubmitting same job_id returns 202 (idempotent)")
    except Exception as e:
        log("fail", "Idempotency", str(e))


async def test_edge_cases(client: httpx.AsyncClient):
    print("\n[8/8] Edge Cases")
    try:
        r = await client.get(f"{BASE_URL}/api/v1/jobs/nonexistent-job-id", headers=HEADERS)
        assert r.status_code == 404
        log("pass", "Nonexistent job returns 404")

        r = await client.post(f"{BASE_URL}/api/v1/moderate", headers=HEADERS, json={})
        assert r.status_code == 422
        log("pass", "Missing required fields returns 422")

        r = await client.post(
            f"{BASE_URL}/api/v1/process",
            headers=HEADERS,
            json={"recording_id": TEST_RECORDING_ID},
        )
        assert r.status_code == 422
        log("pass", "Missing job_id returns 422")

    except Exception as e:
        log("fail", "Edge cases", str(e))


async def main():
    global BASE_URL, SERVICE_KEY, HEADERS
    if len(sys.argv) > 1:
        BASE_URL = sys.argv[1].rstrip("/")
    if len(sys.argv) > 2:
        SERVICE_KEY = sys.argv[2]
        HEADERS["X-Service-Key"] = SERVICE_KEY

    print("=" * 60)
    print("  Hear AI Service — Integration Tests")
    print(f"  Target:     {BASE_URL}")
    print(f"  Recording:  {TEST_RECORDING_ID}")
    print(f"  Track:      {TEST_TRACK_ID}")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=200) as client:
        await test_health(client)
        await test_auth(client)
        await test_moderation(client)
        await test_categorization(client)
        job_id = await test_pipeline_submit(client)
        await test_job_polling(client, job_id)
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
