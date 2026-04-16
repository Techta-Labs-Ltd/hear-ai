"""
Hear AI Service - Integration Test Suite
Run on the server after deployment: python -m tests.test_api
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
    print("\n[1/7] Health Check")
    try:
        r = await client.get(f"{BASE_URL}/health")
        data = r.json()
        assert r.status_code == 200, f"status {r.status_code}"
        log("pass", "GET /health returns 200")

        assert data["status"] == "healthy", f"status={data['status']}"
        log("pass", "Status is healthy")

        assert data["gpu_available"] is True, "GPU not available"
        log("pass", f"GPU available: {data['gpu_name']}")

        models = data.get("models_loaded", [])
        expected = ["whisper", "demucs", "categorizer", "moderator"]
        for m in expected:
            if m in models:
                log("pass", f"Model loaded: {m}")
            else:
                log("fail", f"Model NOT loaded: {m}")
    except Exception as e:
        log("fail", "Health check", str(e))


async def test_auth(client: httpx.AsyncClient):
    print("\n[2/7] Authentication")
    try:
        r = await client.post(
            f"{BASE_URL}/api/v1/moderate",
            json={"text": "hello"},
        )
        assert r.status_code in (401, 403), f"Expected 401/403 without key, got {r.status_code}"
        log("pass", "Rejects request without X-Service-Key")

        r = await client.post(
            f"{BASE_URL}/api/v1/moderate",
            headers={"X-Service-Key": "wrong-key", "Content-Type": "application/json"},
            json={"text": "hello"},
        )
        assert r.status_code in (401, 403), f"Expected 401/403 with wrong key, got {r.status_code}"
        log("pass", "Rejects request with wrong key")

        r = await client.post(
            f"{BASE_URL}/api/v1/moderate",
            headers=HEADERS,
            json={"text": "hello"},
        )
        assert r.status_code == 200, f"Expected 200 with valid key, got {r.status_code}"
        log("pass", "Accepts request with valid key")
    except Exception as e:
        log("fail", "Authentication", str(e))


async def test_moderation(client: httpx.AsyncClient):
    print("\n[3/7] Moderation")
    try:
        r = await client.post(
            f"{BASE_URL}/api/v1/moderate",
            headers=HEADERS,
            json={"text": "This is a perfectly normal and friendly conversation about cooking."},
        )
        data = r.json()
        assert r.status_code == 200
        assert data["flagged"] is False, f"Clean text flagged: {data}"
        log("pass", "Clean text not flagged")

        r = await client.post(
            f"{BASE_URL}/api/v1/moderate",
            headers=HEADERS,
            json={"text": "I will kill you and destroy everything you love, you stupid worthless piece of trash"},
        )
        data = r.json()
        assert r.status_code == 200
        assert data["flagged"] is True, f"Toxic text not flagged: {data}"
        log("pass", "Toxic text flagged correctly")

        assert "categories" in data and "scores" in data
        log("pass", "Returns categories and scores")

        if "openai" in data and data["openai"].get("scores"):
            log("pass", "OpenAI moderation layer active")
        else:
            log("warn", "OpenAI moderation layer inactive (no OPENAI_API_KEY?)")

    except Exception as e:
        log("fail", "Moderation", str(e))


async def test_categorization(client: httpx.AsyncClient):
    print("\n[4/7] Categorization")
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
        assert r.status_code == 200
        log("pass", "POST /api/v1/categorize returns 200")

        assert "tags" in data and isinstance(data["tags"], list)
        log("pass", f"Tags returned: {data['tags']}")

        assert "categories" in data and isinstance(data["categories"], list)
        log("pass", f"Categories returned: {data['categories']}")

        assert "sentiment" in data
        log("pass", f"Sentiment: {data['sentiment']}")

        assert "confidence_scores" in data
        log("pass", f"Confidence scores: {len(data['confidence_scores'])} entries")

        r2 = await client.post(
            f"{BASE_URL}/api/v1/categorize",
            headers=HEADERS,
            json={"text": transcript, "custom_tags": ["ai-research", "tech-news"], "max_tags": 5},
        )
        data2 = r2.json()
        assert r2.status_code == 200
        log("pass", f"Custom tags accepted, result tags: {data2['tags']}")

    except Exception as e:
        log("fail", "Categorization", str(e))


async def test_pipeline_submit(client: httpx.AsyncClient):
    print("\n[5/7] Pipeline Job Submission")
    job_id = str(uuid.uuid4())
    try:
        r = await client.post(
            f"{BASE_URL}/api/v1/process",
            headers=HEADERS,
            json={
                "job_id": job_id,
                "recording_id": f"test-rec-{uuid.uuid4().hex[:8]}",
                "job_type": "pipeline",
                "max_tags": 5,
            },
        )
        data = r.json()
        assert r.status_code == 202, f"Expected 202, got {r.status_code}"
        assert data["job_id"] == job_id
        log("pass", f"Pipeline job accepted: {job_id}")

        await asyncio.sleep(2)

        r2 = await client.get(
            f"{BASE_URL}/api/v1/jobs/{job_id}",
            headers=HEADERS,
        )
        job_data = r2.json()
        assert r2.status_code == 200
        log("pass", f"Job status: {job_data['status']}")

        return job_id
    except Exception as e:
        log("fail", "Pipeline submit", str(e))
        return None


async def test_job_polling(client: httpx.AsyncClient, job_id: str):
    print("\n[6/7] Pipeline Job Completion (polling)")
    if not job_id:
        log("fail", "Skipped — no job_id from previous test")
        return

    max_wait = 120
    poll_interval = 5
    elapsed = 0

    try:
        while elapsed < max_wait:
            r = await client.get(
                f"{BASE_URL}/api/v1/jobs/{job_id}",
                headers=HEADERS,
            )
            data = r.json()
            status = data.get("status", "unknown")

            if status == "completed":
                log("pass", f"Job completed in ~{elapsed}s")
                result = data.get("result")
                if result:
                    if isinstance(result, str):
                        result = json.loads(result)

                    if result.get("transcription"):
                        log("pass", f"Transcription present ({len(result['transcription'])} chars)")
                    else:
                        log("warn", "No transcription in result")

                    if result.get("enhancement"):
                        log("pass", "Enhancement data present")
                    else:
                        log("warn", "No enhancement data")

                    if result.get("categorization"):
                        log("pass", f"Tags: {result['categorization'].get('tags', [])}")
                    else:
                        log("warn", "No categorization data")

                    if result.get("moderation"):
                        log("pass", f"Flagged: {result['moderation'].get('flagged', '?')}")
                    else:
                        log("warn", "No moderation data")
                else:
                    log("warn", "Job completed but no result JSON")
                return

            if status == "failed":
                log("fail", f"Job failed: {data.get('error', 'unknown')}")
                return

            print(f"    ... status={status}, waiting ({elapsed}s/{max_wait}s)")
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        log("fail", f"Job did not complete within {max_wait}s (last status: {status})")
    except Exception as e:
        log("fail", "Job polling", str(e))


async def test_edge_cases(client: httpx.AsyncClient):
    print("\n[7/7] Edge Cases")
    try:
        r = await client.post(
            f"{BASE_URL}/api/v1/moderate",
            headers=HEADERS,
            json={"text": ""},
        )
        data = r.json()
        assert r.status_code == 200
        assert data["flagged"] is False
        log("pass", "Empty text moderation handled")

        r = await client.post(
            f"{BASE_URL}/api/v1/categorize",
            headers=HEADERS,
            json={"text": "", "max_tags": 5},
        )
        data = r.json()
        assert r.status_code == 200
        assert data["tags"] == []
        log("pass", "Empty text categorization handled")

        r = await client.get(
            f"{BASE_URL}/api/v1/jobs/nonexistent-job-id",
            headers=HEADERS,
        )
        assert r.status_code == 404
        log("pass", "Nonexistent job returns 404")

        r = await client.post(
            f"{BASE_URL}/api/v1/moderate",
            headers=HEADERS,
            json={},
        )
        assert r.status_code == 422
        log("pass", "Missing required fields returns 422")

    except Exception as e:
        log("fail", "Edge cases", str(e))


async def main():
    if len(sys.argv) > 1:
        global BASE_URL, SERVICE_KEY, HEADERS
        BASE_URL = sys.argv[1].rstrip("/")
    if len(sys.argv) > 2:
        SERVICE_KEY = sys.argv[2]
        HEADERS["X-Service-Key"] = SERVICE_KEY

    print("=" * 60)
    print("  Hear AI Service — Integration Tests")
    print(f"  Target: {BASE_URL}")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=180) as client:
        await test_health(client)
        await test_auth(client)
        await test_moderation(client)
        await test_categorization(client)
        job_id = await test_pipeline_submit(client)
        await test_job_polling(client, job_id)
        await test_edge_cases(client)

    print("\n" + "=" * 60)
    total = results["passed"] + results["failed"] + results["warnings"]
    print(f"  Results: {results['passed']}/{total} passed, "
          f"{results['failed']} failed, {results['warnings']} warnings")
    print("=" * 60)

    sys.exit(1 if results["failed"] > 0 else 0)


if __name__ == "__main__":
    asyncio.run(main())
