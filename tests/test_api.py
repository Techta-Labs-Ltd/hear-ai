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
        assert r.status_code == 200, f"Expected 200 with valid key, got {r.status_code}: {r.text}"
        log("pass", "Accepts request with valid X-Service-Key")
    except Exception as e:
        log("fail", "Authentication", str(e))


async def test_moderation(client: httpx.AsyncClient):
    print("\n[3/9] Moderation")
    try:
        r = await client.post(
            f"{BASE_URL}/api/v1/moderate",
            headers=HEADERS,
            json={"text": "Welcome everyone to today's podcast about music production and sound design."},
        )
        data = r.json()
        assert r.status_code == 200, f"{r.status_code}: {r.text}"
        assert data["flagged"] is False, f"Clean content was flagged: {data}"
        log("pass", f"Clean content not flagged (severity={data.get('severity', '?')})")

        r = await client.post(
            f"{BASE_URL}/api/v1/moderate",
            headers=HEADERS,
            json={"text": "I will kill you and destroy everything you love, you worthless piece of trash"},
        )
        data = r.json()
        assert r.status_code == 200
        assert data["flagged"] is True, f"Toxic content not flagged: {data}"
        log("pass", f"Toxic content flagged (severity={data.get('severity', '?')})")
        log("pass", f"Reason: {data.get('reason', '?')!r}")

        r = await client.post(f"{BASE_URL}/api/v1/moderate", headers=HEADERS, json={"text": ""})
        data = r.json()
        assert r.status_code == 200
        assert data["flagged"] is False
        log("pass", "Empty text handled and not flagged")
    except Exception as e:
        log("fail", "Moderation", str(e))


async def test_categorization(client: httpx.AsyncClient):
    print("\n[4/9] Categorization")
    try:
        transcript = (
            "Today we're talking about artificial intelligence and machine learning. "
            "Companies like Google and OpenAI are pushing transformer architectures. "
            "The implications for healthcare, education, and business are enormous."
        )
        r = await client.post(
            f"{BASE_URL}/api/v1/categorize",
            headers=HEADERS,
            json={"text": transcript, "max_tags": 5},
        )
        data = r.json()
        assert r.status_code == 200, f"{r.status_code}: {r.text}"
        assert "tags" in data and isinstance(data["tags"], list)
        log("pass", f"Tags: {data['tags']}")
        assert "sentiment" in data
        log("pass", f"Sentiment: {data['sentiment']}")
    except Exception as e:
        log("fail", "Categorization", str(e))


async def test_backend_connectivity(client: httpx.AsyncClient):
    print("\n[5/9] Backend Connectivity & Recording Fetch")
    try:
        url = f"{BACKEND_URL}/api/v1/internal/recordings/{TEST_RECORDING_ID}"
        r = await client.get(url, headers=HEADERS)
        if r.status_code == 200:
            data = r.json()
            log("pass", f"Backend reachable — recording '{data.get('title')}' fetched")
            tracks = data.get("tracks", [])
            log("pass" if tracks else "warn", f"Tracks in recording: {len(tracks)}")
            for t in tracks:
                log("pass", f"  track={t['id'][:8]} is_enhanced={t.get('is_enhanced')} transcription={'set' if t.get('transcription') else 'null'}")
        elif r.status_code == 401:
            log("fail", "Backend auth rejected — check AI_SERVICE_SECRET matches backend config")
        elif r.status_code == 404:
            log("fail", "Recording not found on backend — check TEST_RECORDING_ID")
        else:
            log("fail", f"Backend returned {r.status_code}: {r.text[:200]}")
    except httpx.ConnectError:
        log("fail", f"Cannot reach backend at {BACKEND_URL} — check HEAR_BACKEND_URL")
    except Exception as e:
        log("fail", "Backend connectivity", str(e))


async def test_pipeline_submit(client: httpx.AsyncClient):
    print("\n[6/9] Pipeline Job Submission (real recording)")
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
        assert r2.status_code == 200, f"Job lookup failed: {r2.text}"
        log("pass", f"Job in DB — status: {job_data['status']}")
        if job_data.get("error"):
            log("warn", f"Early error: {job_data['error']}")

        return job_id
    except Exception as e:
        log("fail", "Pipeline submit", str(e))
        return None


async def test_job_polling(client: httpx.AsyncClient, job_id: str):
    print("\n[7/9] Pipeline Job Completion (polling up to 3 min)")
    if not job_id:
        log("fail", "Skipped — no job_id")
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
                    log("pass" if tracks else "warn", f"Tracks processed: {len(tracks)}")
                    log("pass" if result.get("categorization") else "warn",
                        f"Tags: {result.get('categorization', {}).get('tags', [])}")
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
        await test_moderation(client)
        await test_categorization(client)
        await test_backend_connectivity(client)
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
