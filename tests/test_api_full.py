"""Comprehensive API test"""
import asyncio, json, sys, time, httpx

BASE = "http://localhost:8000"
SECRET = "8c361a305f40bfc53b165b41c274b00fdba004445e8473bc0e04da8d0093aca4"
H = {"X-Service-Key": SECRET, "Content-Type": "application/json"}

PASS = 0
FAIL = 0

def ok(name, status, body=""):
    global PASS, FAIL
    if 200 <= status < 300:
        PASS += 1
        print(f"  ✓ [{status}] {name}")
    else:
        FAIL += 1
        print(f"  ✗ [{status}] {name}  body={str(body)[:120]}")

def ok_json(name, status, data, checks=None):
    global PASS, FAIL
    if 200 <= status < 300:
        if checks:
            for k, v in checks.items():
                actual = data.get(k)
                if actual != v and not (isinstance(v, type) and isinstance(actual, v)):
                    FAIL += 1
                    print(f"  ✗ [{status}] {name} — missing/failed check: {k}={actual} (expected {v})")
                    return
        PASS += 1
        print(f"  ✓ [{status}] {name}")
    else:
        FAIL += 1
        print(f"  ✗ [{status}] {name}  body={str(data)[:120]}")

async def main():
    async with httpx.AsyncClient(timeout=10) as c:
        print("HEALTH")
        r = await c.get(f"{BASE}/health")
        d = r.json()
        ok_json("health", r.status_code, d, {
            "status": "healthy", "gpu_available": True, "redis_status": "connected"
        })

        print("\nQUEUE STATS")
        r = await c.get(f"{BASE}/api/v1/queue/stats", headers=H)
        d = r.json()
        print(f"  queued={d.get('queued')} active={d.get('active')} total={d.get('total')}")
        ok_json("stats", r.status_code, d, {"queued": int, "active": int})

        base_queued = d.get("queued", 0)

        print("\nSUBMIT JOBS")
        jobs = [
            ("test-tx-01", "transcription", {"edited_transcript": "Hello world."}),
            ("test-pipe-01", "pipeline", {"edited_transcript": "Hello world."}),
            ("test-tag-01", "audio_tag", {}),
        ]
        for jid, jtype, extra in jobs:
            r = await c.post(f"{BASE}/api/v1/process", headers=H, json={
                "job_id": jid, "track_id": f"track-{jid}", "job_type": jtype, **extra
            })
            ok_json(f"submit {jtype}", r.status_code, r.json(), {"status": "accepted"})

        print("\nQUEUE STATS (after submit)")
        r = await c.get(f"{BASE}/api/v1/queue/stats", headers=H)
        d = r.json()
        ok_json("stats after submit", r.status_code, d,
                {"queued": base_queued + 3, "active": int, "total": base_queued + 3})

        print("\nJOB STATUS")
        for jid in ["test-tx-01", "test-pipe-01", "test-tag-01"]:
            r = await c.get(f"{BASE}/api/v1/jobs/{jid}", headers=H)
            d = r.json()
            ok_json(f"job {jid}", r.status_code, d, {"job_id": jid})

        print("\nCANCEL A JOB")
        r = await c.post(f"{BASE}/api/v1/jobs/test-tag-01/cancel", headers=H)
        ok_json("cancel", r.status_code, r.json(), {"cancelled": True})

        print("\nQUEUE STATS (after cancel)")
        r = await c.get(f"{BASE}/api/v1/queue/stats", headers=H)
        d = r.json()
        print(f"  queued={d.get('queued')} (was {base_queued + 3})")
        ok_json("stats after cancel", r.status_code, d, {"queued": int})

        print("\nCANCEL NONEXISTENT")
        r = await c.post(f"{BASE}/api/v1/jobs/nonexistent/cancel", headers=H)
        ok_json("cancel nonexistent 404", r.status_code, r.json(), {"detail": "Job not found"})

        print("\nUNAUTHORIZED")
        r = await c.get(f"{BASE}/api/v1/queue/stats")
        ok_json("unauthorized 401", r.status_code, r.json(), {"detail": "Authorization required"})

        print(f"\n{'='*50}")
        print(f"  RESULTS: {PASS} passed, {FAIL} failed")
        print(f"{'='*50}")
        if FAIL > 0:
            sys.exit(1)

asyncio.run(main())
