"""Run reconstruction on a local audio file and download results.

Tests zero-shot voice cloning with Fish Speech TTS.
Uses the HRA Elects New Leaders to Challenge Council.wav audio.

Usage:
    python tests/test_reconstruct_local.py
"""

import time
import httpx
import os
import sys
import uuid

BASE_URL = os.getenv("AI_SERVICE_URL", "http://localhost:8000")
SERVICE_KEY = os.getenv("AI_SERVICE_SECRET", "8c361a305f40bfc53b165b41c274b00fdba004445e8473bc0e04da8d0093aca4")

LOCAL_AUDIO_URL = "http://127.0.0.1:8765/test_audio.wav"
OUTPUT_DIR = "/workspace/hear-ai/reconstruct_output"

PASS = "\033[92m"
FAIL = "\033[91m"
END = "\033[0m"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    client = httpx.Client(timeout=600)
    passed = 0
    failed = 0

    # Health check
    print("=" * 60)
    print("  Hear AI - Reconstruction Test (Zero-Shot Voice Cloning)")
    print("  Audio: HRA Elects New Leaders to Challenge Council.wav")
    print("=" * 60)
    r = client.get(f"{BASE_URL}/health")
    health = r.json()
    print(f"\n  Server: {health['status']}")
    print(f"  GPU:    {health.get('gpu_name', 'N/A')}")
    print(f"  Models: {health.get('models_loaded', [])}")

    # Test 1: Single word edit with voice cloning
    print("\n" + "-" * 50)
    print("[1/5] Word Edit + Voice Clone: 'hello world'")
    print(f"      Segment: 1.0s - 2.0s (same_speaker=True)")

    t0 = time.time()
    r = client.post(
        f"{BASE_URL}/api/v1/reconstruct",
        headers={"X-Service-Key": SERVICE_KEY, "Content-Type": "application/json"},
        json={
            "audio_url": LOCAL_AUDIO_URL,
            "track_id": f"local-1-{uuid.uuid4().hex[:8]}",
            "segment_start": 1.0,
            "segment_end": 2.0,
            "new_text": "hello world",
            "same_speaker": True,
        },
        timeout=300.0,
    )

    if r.status_code != 200:
        print(f"      {FAIL}FAILED{END}: HTTP {r.status_code} - {r.text[:300]}")
        failed += 1
    else:
        data = r.json()
        preview_id = data.get("preview_id", "")
        audio_url = data.get("preview_audio_url", "")
        duration = data.get("preview_duration", 0)
        qm = data.get("quality_metrics", {})
        elapsed = time.time() - t0
        print(f"      {PASS}OK{END} ({elapsed:.1f}s, {duration}s audio)")
        print(f"      preview_id:    {preview_id}")
        print(f"      preview_url:   {audio_url}")
        print(f"      quality:       DNSMOS={qm.get('dnsmos_ovr',0):.2f}  "
              f"loudness={qm.get('loudness_match_db',0):.1f}dB  "
              f"clipping={qm.get('clipping_detected',False)}")
        passed += 1

    # Test 2: Confirm and download
    print("\n" + "-" * 50)
    print("[2/5] Confirm Preview + Download Result")
    if preview_id:
        r2 = client.post(
            f"{BASE_URL}/api/v1/reconstruct/confirm",
            headers={"X-Service-Key": SERVICE_KEY, "Content-Type": "application/json"},
            json={"preview_id": preview_id, "track_id": f"local-1-{uuid.uuid4().hex[:8]}"},
            timeout=60.0,
        )
        if r2.status_code != 200:
            print(f"      {FAIL}FAILED{END}: HTTP {r2.status_code} - {r2.text[:300]}")
            failed += 1
        else:
            final_url = r2.json().get("audio_url", "")
            print(f"      final_url: {final_url}")
            r3 = client.get(final_url, timeout=60.0)
            path = os.path.join(OUTPUT_DIR, "word_edit_cloned.mp3")
            with open(path, "wb") as f:
                f.write(r3.content)
            print(f"      {PASS}Downloaded{END}: {path} ({len(r3.content)/1024:.1f} KB)")
            passed += 1

    # Test 3: Phrase edit with voice cloning
    print("\n" + "-" * 50)
    print("[3/5] Phrase Edit + Voice Clone: 'zero shot voice cloning test'")
    print(f"      Segment: 1.0s - 3.0s (same_speaker=True)")

    t0 = time.time()
    r = client.post(
        f"{BASE_URL}/api/v1/reconstruct",
        headers={"X-Service-Key": SERVICE_KEY, "Content-Type": "application/json"},
        json={
            "audio_url": LOCAL_AUDIO_URL,
            "track_id": f"local-3-{uuid.uuid4().hex[:8]}",
            "segment_start": 1.0,
            "segment_end": 3.0,
            "new_text": "zero shot voice cloning test",
            "same_speaker": True,
        },
        timeout=300.0,
    )
    if r.status_code == 200:
        data = r.json()
        elapsed = time.time() - t0
        print(f"      {PASS}OK{END} ({elapsed:.1f}s, {data.get('preview_duration',0)}s)")
        print(f"      preview_id:  {data['preview_id']}")
        print(f"      preview_url: {data.get('preview_audio_url','')}")
        passed += 1
    else:
        print(f"      {FAIL}FAILED{END}: HTTP {r.status_code} - {r.text[:300]}")
        failed += 1

    # Test 4: Multi-segment edit with voice cloning
    print("\n" + "-" * 50)
    print("[4/5] Multi-Segment + Voice Clone: 2 segments at once")
    print(f"      Segments: [1.0-1.5] + [3.0-3.5] (same_speaker=True)")

    t0 = time.time()
    r = client.post(
        f"{BASE_URL}/api/v1/reconstruct",
        headers={"X-Service-Key": SERVICE_KEY, "Content-Type": "application/json"},
        json={
            "audio_url": LOCAL_AUDIO_URL,
            "track_id": f"local-4-{uuid.uuid4().hex[:8]}",
            "changes": [
                {"segment_start": 1.0, "segment_end": 1.5, "new_text": "first segment edited"},
                {"segment_start": 3.0, "segment_end": 3.5, "new_text": "second segment edited"},
            ],
            "same_speaker": True,
        },
        timeout=600.0,
    )
    if r.status_code == 200:
        data = r.json()
        elapsed = time.time() - t0
        print(f"      {PASS}OK{END} ({elapsed:.1f}s, {data.get('preview_duration',0)}s)")
        print(f"      segments_applied: {data.get('segments_applied',0)}")
        print(f"      preview_id:       {data['preview_id']}")
        passed += 1
    else:
        print(f"      {FAIL}FAILED{END}: HTTP {r.status_code} - {r.text[:300]}")
        failed += 1

    # Test 5: No clone mode (same_speaker=False) — still uses reference audio
    print("\n" + "-" * 50)
    print("[5/5] Smart Voice (no clone): 'test' -> 'generic voice'")
    print(f"      Segment: 1.0s - 2.0s (same_speaker=False)")

    t0 = time.time()
    r = client.post(
        f"{BASE_URL}/api/v1/reconstruct",
        headers={"X-Service-Key": SERVICE_KEY, "Content-Type": "application/json"},
        json={
            "audio_url": LOCAL_AUDIO_URL,
            "track_id": f"local-5-{uuid.uuid4().hex[:8]}",
            "segment_start": 1.0,
            "segment_end": 2.0,
            "new_text": "generic voice test",
            "same_speaker": False,
        },
        timeout=300.0,
    )
    if r.status_code == 200:
        data = r.json()
        elapsed = time.time() - t0
        print(f"      {PASS}OK{END} ({elapsed:.1f}s, {data.get('preview_duration',0)}s)")
        print(f"      preview_id:  {data['preview_id']}")
        print(f"      preview_url: {data.get('preview_audio_url','')}")
        passed += 1
    else:
        print(f"      {FAIL}FAILED{END}: HTTP {r.status_code} - {r.text[:300]}")
        failed += 1

    # Summary
    print("\n" + "=" * 60)
    total = passed + failed
    print(f"  Results: {PASS}{passed}{END}/{total} passed, {FAIL}{failed}{END} failed")
    print("=" * 60)
    print("\n  Output files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath) / 1024
        print(f"    {f} ({size:.1f} KB)")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
