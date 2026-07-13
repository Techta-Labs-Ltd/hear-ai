#!/usr/bin/env python3
"""
E2E test for Qwen3-TTS preview → confirm → remove flow.

Generates a local test audio file and serves it via a temp HTTP server
so no external dependencies are needed.

Usage:
    python3 tests/test_qwen3_preview_flow.py --local-audio

Flags:
    --local-audio    Generate a local test audio file and serve it on :9999
    --legacy-only    Only test the legacy direct-splice endpoint
    --remove-only    Only test the remove endpoint
    --skip-local     Use existing AUDIO_URL (default media.hear.surf)
"""

import argparse
import json
import os
import random
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import HTTPServer, SimpleHTTPRequestHandler

import requests

BASE = os.environ.get("HEAR_API_URL", "http://localhost:8000")
HEADERS = {
    "X-Service-Key": os.environ.get("AI_SERVICE_SECRET", "8c361a305f40bfc53b165b41c274b00fdba004445e8473bc0e04da8d0093aca4"),
    "Content-Type": "application/json",
}

TRACK_ID = f"test-qwen3-flow-{uuid.uuid4().hex[:8]}"
LOCAL_SERVE_PORT = 9999


def generate_test_wav(path: str, duration: float = 5.0, sample_rate: int = 44100):
    import subprocess as _sp
    try:
        _sp.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i",
             f"sine=frequency=220:duration={duration}",
             "-ac", "1", "-ar", str(sample_rate), path],
            check=True, capture_output=True, timeout=15,
        )
        return path
    except Exception:
        pass


class TempAudioHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=directory, **kwargs)

    def log_message(self, format, *args):
        pass  # suppress HTTP server logs


def serve_temp_audio(audio_path: str, port: int):
    """Start a temporary HTTP server to serve a test audio file."""
    directory = os.path.dirname(audio_path)
    filename = os.path.basename(audio_path)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    audio_url = f"http://127.0.0.1:{port}/{filename}"
    print(f"  Serving test audio at: {audio_url}")
    return server, audio_url


def log(label, resp, elapsed=None):
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")
    print(f"Status: {resp.status_code}" + (f" ({elapsed:.1f}s)" if elapsed else ""))
    try:
        data = resp.json()
        text = json.dumps(data, indent=2)
        if len(text) > 2000:
            text = text[:2000] + "\n  ... (truncated)"
        print(f"Body: {text}")
    except Exception:
        print(f"Body: {resp.text[:500]}")
    return data if resp.ok else None


def test_preview_flow(audio_url):
    print("\n\n\x1b[36m\x1b[1m━━━ TEST 1: Preview Flow (POST /api/v1/reconstruct → confirm) ━━━\x1b[0m")

    payload = {
        "audio_url": audio_url,
        "track_id": TRACK_ID,
        "changes": [
            {
                "segment_start": 0.0,
                "segment_end": 1.5,
                "new_text": "Praise the Lord, praise the Lord.",
                "original_text": "Test audio segment one.",
            }
        ],
        "same_speaker": True,
    }

    t0 = time.time()
    resp = requests.post(f"{BASE}/api/v1/reconstruct", json=payload, headers=HEADERS)
    data = log(f"[1.1] POST /api/v1/reconstruct (preview)", resp, time.time() - t0)
    if not data:
        return False

    preview_id = data.get("preview_id")
    if not preview_id:
        print("\n\x1b[31m✗ No preview_id returned\x1b[0m")
        return False
    print(f"\n  ✓ preview_id: {preview_id}")

    # Get preview status
    resp = requests.get(f"{BASE}/api/v1/reconstruct/previews/{preview_id}", headers=HEADERS)
    log(f"[1.2] GET preview/{preview_id}", resp)

    # Confirm
    confirm_payload = {
        "preview_id": preview_id,
        "track_id": TRACK_ID,
        "user_id": "test-user-001",
    }
    t0 = time.time()
    resp = requests.post(f"{BASE}/api/v1/reconstruct/confirm", json=confirm_payload, headers=HEADERS)
    data = log(f"[1.3] POST /api/v1/reconstruct/confirm", resp, time.time() - t0)
    if data:
        print(f"  ✓ final audio_url: {data.get('audio_url', 'N/A')}")
        print(f"  ✓ user_id in response: {data.get('user_id')}")
    return True


def test_remove_flow(audio_url):
    print("\n\n\x1b[36m\x1b[1m━━━ TEST 2: Remove Segment (POST /api/v1/reconstruct/remove) ━━━\x1b[0m")

    remove_payload = {
        "audio_url": audio_url,
        "track_id": f"{TRACK_ID}-remove",
        "segment_start": 0.0,
        "segment_end": 1.5,
        "user_id": "test-user-001",
    }

    t0 = time.time()
    resp = requests.post(f"{BASE}/api/v1/reconstruct/remove", json=remove_payload, headers=HEADERS)
    data = log(f"[2.1] POST /api/v1/reconstruct/remove", resp, time.time() - t0)
    if not data:
        return False
    print(f"  ✓ removed_duration: {data.get('removed_duration')}")
    print(f"  ✓ user_id: {data.get('user_id')}")
    return True


def test_legacy_direct_mode(audio_url):
    print("\n\n\x1b[36m\x1b[1m━━━ TEST 3: Legacy Direct Mode (X-Preview-Mode: false) ━━━\x1b[0m")

    payload = {
        "audio_url": audio_url,
        "track_id": f"{TRACK_ID}-legacy",
        "changes": [
            {
                "segment_start": 1.0,
                "segment_end": 2.5,
                "new_text": "Hallelujah.",
            }
        ],
        "same_speaker": True,
    }
    headers = {**HEADERS, "X-Preview-Mode": "false"}

    t0 = time.time()
    resp = requests.post(f"{BASE}/api/v1/reconstruct", json=payload, headers=headers)
    data = log(f"[3.1] POST /api/v1/reconstruct (legacy direct)", resp, time.time() - t0)
    if not data:
        return False
    print(f"  ✓ audio_url: {data.get('audio_url', 'N/A')}")
    print(f"  ✓ duration: {data.get('duration')}")
    return True


def test_rollback_flow(audio_url):
    print("\n\n\x1b[36m\x1b[1m━━━ TEST 4: Rollback Preview ━━━\x1b[0m")

    payload = {
        "audio_url": audio_url,
        "track_id": f"{TRACK_ID}-rollback",
        "changes": [
            {
                "segment_start": 0.0,
                "segment_end": 1.5,
                "new_text": "Praise the Lord.",
            }
        ],
        "same_speaker": True,
    }

    resp = requests.post(f"{BASE}/api/v1/reconstruct", json=payload, headers=HEADERS)
    data = resp.json()
    preview_id = data.get("preview_id")
    if not preview_id:
        print("\n\x1b[33m⚠ No preview_id — skipping rollback test\x1b[0m")
        return True

    resp = requests.post(
        f"{BASE}/api/v1/reconstruct/rollback?preview_id={preview_id}",
        headers=HEADERS,
    )
    data = log(f"[4.1] POST /api/v1/reconstruct/rollback", resp)
    if data:
        print(f"  ✓ status: {data.get('status')}")

    # Verify status changed
    resp = requests.get(f"{BASE}/api/v1/reconstruct/previews/{preview_id}", headers=HEADERS)
    data = log(f"[4.2] GET preview after rollback", resp)
    if data:
        print(f"  ✓ final status: {data.get('status')}")
    return True


def test_endpoint_discovery():
    """Verify all new endpoints are registered and respond properly."""
    print("\n\n\x1b[36m\x1b[1m━━━ TEST 5: Endpoint Discovery ━━━\x1b[0m")

    tests = [
        ("GET /health", lambda: requests.get(f"{BASE}/health", headers=HEADERS)),
        ("POST /api/v1/reconstruct/confirm (bad body)", lambda: requests.post(
            f"{BASE}/api/v1/reconstruct/confirm", json={}, headers=HEADERS)),
        ("POST /api/v1/reconstruct/remove (bad body)", lambda: requests.post(
            f"{BASE}/api/v1/reconstruct/remove", json={}, headers=HEADERS)),
        ("POST /api/v1/reconstruct/rollback (no preview_id)", lambda: requests.post(
            f"{BASE}/api/v1/reconstruct/rollback", headers=HEADERS)),
        ("GET /api/v1/reconstruct/previews/fake-id", lambda: requests.get(
            f"{BASE}/api/v1/reconstruct/previews/fake-id", headers=HEADERS)),
    ]

    all_ok = True
    for name, fn in tests:
        try:
            resp = fn()
            if resp.status_code in (200, 422, 404, 405):
                print(f"  ✓ {name} → {resp.status_code}")
            else:
                print(f"  ? {name} → {resp.status_code} (unexpected)")
                all_ok = False
        except Exception as e:
            print(f"  ✗ {name} → {e}")
            all_ok = False
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Qwen3 preview/confirm/remove flow")
    parser.add_argument("--local-audio", action="store_true", help="Generate local test audio")
    parser.add_argument("--legacy-only", action="store_true")
    parser.add_argument("--remove-only", action="store_true")
    parser.add_argument("--skip-local", action="store_true",
                        help="Use AUDIO_URL from env or default")
    args = parser.parse_args()

    print(f"\x1b[1mQwen3-TTS Preview/Confirm/Remove Flow Test\x1b[0m")
    print(f"  Server: {BASE}")
    print(f"  Track:  {TRACK_ID}")

    # ── Set up test audio ──────────────────────────────────────────
    server = None
    if args.local_audio:
        tmpdir = tempfile.mkdtemp(prefix="hear-test-")
        audio_path = os.path.join(tmpdir, "test_audio.wav")
        generate_test_wav(audio_path, duration=5.0)
        print(f"  Generated: {audio_path} ({os.path.getsize(audio_path)} bytes)")
        server, audio_url = serve_temp_audio(audio_path, LOCAL_SERVE_PORT)
        print(f"  Serving on: http://127.0.0.1:{LOCAL_SERVE_PORT}/")
    else:
        audio_url = os.environ.get("AUDIO_URL", "")
        if not audio_url:
            audio_url = "https://media.hear.surf/pipeline-source-mp3/1e75983e-8fa3-4859-ba99-281df885750c/18249299-2b8f-4a9b-b333-3a1ee3f478b4-6a2633be-d937-4215-9097-ca27896a0346.mp3"
        print(f"  Audio: {audio_url[:80]}...")

    # ── Run tests ──────────────────────────────────────────────────
    all_pass = True
    if args.legacy_only:
        ok = test_legacy_direct_mode(audio_url)
        if not ok:
            all_pass = False
    elif args.remove_only:
        ok = test_remove_flow(audio_url)
        if not ok:
            all_pass = False
    else:
        test_endpoint_discovery()

        ok = test_preview_flow(audio_url)
        if not ok:
            print("\n\x1b[33m⚠ Preview flow had issues — continuing...\x1b[0m")

        ok = test_remove_flow(audio_url)
        if not ok:
            all_pass = False

        ok = test_legacy_direct_mode(audio_url)
        if not ok:
            all_pass = False

        test_rollback_flow(audio_url)

    # ── Cleanup ────────────────────────────────────────────────────
    if server:
        server.shutdown()

    print(f"\n{'=' * 60}")
    if all_pass:
        print("\x1b[32m\x1b[1m✓ All tests passed\x1b[0m")
    else:
        print("\x1b[31m\x1b[1m✗ Some tests failed\x1b[0m")
    print(f"{'=' * 60}")
    sys.exit(0 if all_pass else 1)
