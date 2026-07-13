"""Test pipeline reconstruct API with 4 large realistic edits.

Uses /api/v1/process (async pipeline) instead of /api/v1/reconstruct (direct).
Polls job status until completion, then downloads and transcribes the output.
"""
import requests
import time
import json
import uuid
import os
import subprocess
import threading
import http.server
import functools

BASE = "http://localhost:8000"
HEADERS = {
    "X-Service-Key": "8c361a305f40bfc53b165b41c274b00fdba004445e8473bc0e04da8d0093aca4",
    "Content-Type": "application/json",
}
OUTPUT_DIR = "/workspace/reconstruct_output"

# Serve local file via HTTP
LOCAL_SOURCE = "/workspace/reconstruct_output/v3_transcript_context.mp3"
SERVE_DIR = "/workspace/reconstruct_output"
SERVE_PORT = 8889
AUDIO_URL = f"http://localhost:{SERVE_PORT}/v3_transcript_context.mp3"

changes = [
    {
        "segment_start": 0.0,
        "segment_end": 4.5,
        "original_text": "This article is telling us what happened in Tindale at our number of years",
        "new_text": "This week's article is telling us everything that happened in Tynedale after a significant number of years of local history and community events that shaped the region",
    },
    {
        "segment_start": 42.0,
        "segment_end": 50.0,
        "original_text": "and Fading Schools and an executive head teacher was a point to turn around the fort during June",
        "new_text": "and Farding School, an executive head teacher was appointed in June to turn around the fortunes of both schools, bringing new leadership and a fresh vision for education in the community",
    },
    {
        "segment_start": 62.0,
        "segment_end": 72.0,
        "original_text": "One of Tyendale's best loan firms was pursuing a claim against the Council for a business rape rebate following months of roadworks outside its hex and premises",
        "new_text": "One of Tynedale's best known firms was actively pursuing a compensation claim against the County Council for a business rates rebate following months of disruptive roadworks directly outside its Hexham premises, which the business owner claimed had cost thousands of pounds in lost revenue",
    },
    {
        "segment_start": 100.0,
        "segment_end": 108.0,
        "original_text": "Hexham MP Geoffrey Rippham told the coround that he had no regrets about getting involved in a House of Commons Frecar",
        "new_text": "Hexham MP Geoffrey Rippon told the court that he absolutely had no regrets whatsoever about getting personally involved in a House of Commons fracas that had captured national attention",
    },
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Start local file server
print(f"Starting local file server on port {SERVE_PORT}...")
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=SERVE_DIR)
httpd = http.server.HTTPServer(("0.0.0.0", SERVE_PORT), handler)
server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
server_thread.start()
print(f"    Serving {SERVE_DIR} at http://localhost:{SERVE_PORT}/")

# Verify file is accessible
try:
    test_resp = requests.head(AUDIO_URL, timeout=3)
    print(f"    Source file accessible: {test_resp.status_code}")
except Exception as e:
    print(f"    WARNING: Could not reach local file: {e}")

print()
print("=" * 70)
print("PIPELINE RECONSTRUCT TEST (4 large edits)")
print("Endpoint: POST /api/v1/process (async)")
print("=" * 70)
for i, c in enumerate(changes):
    wc_old = len(c.get("original_text", "").split())
    wc_new = len(c["new_text"].split())
    print(f"  Change {i+1}: [{c['segment_start']}-{c['segment_end']}s] {wc_old}w -> {wc_new}w")
    print(f"    new: \"{c['new_text'][:70]}...\"")

# Submit via pipeline API
job_id = str(uuid.uuid4())
payload = {
    "job_id": job_id,
    "track_id": f"pipeline-large-{uuid.uuid4().hex[:8]}",
    "job_type": "reconstruct",
    "audio_url": AUDIO_URL,
    "changes": changes,
    "same_speaker": True,
}

print(f"\n[1] Submitting job {job_id}...")
t0 = time.time()
resp = requests.post(f"{BASE}/api/v1/process", json=payload, headers=HEADERS)
print(f"    Response ({resp.status_code}): {json.dumps(resp.json())}")

if resp.status_code != 202:
    print(f"FAILED to submit: {json.dumps(resp.json(), indent=2)}")
    httpd.shutdown()
    exit(1)

# Poll until complete
print(f"\n[2] Polling job status...")
for attempt in range(120):
    time.sleep(3)
    poll = requests.get(f"{BASE}/api/v1/jobs/{job_id}", headers=HEADERS)
    pdata = poll.json()
    status = pdata.get("status", "unknown")
    stage = pdata.get("current_stage", "")

    if attempt % 5 == 0 or status in ("completed", "failed", "cancelled"):
        elapsed = time.time() - t0
        print(f"    [{elapsed:.0f}s] status={status} stage={stage}")

    if status in ("completed", "failed", "cancelled"):
        print(f"\n    Final status: {status}")

        if status == "completed":
            result = pdata.get("result", {})
            aurl = result.get("reconstructed_audio", {}).get("audio_url") or result.get("audio_url", "")
            segs_applied = result.get("segments_applied", "?")
            print(f"    Segments applied: {segs_applied}")
            print(f"    Audio URL: {aurl}")

            if aurl:
                print(f"\n[3] Downloading output...")
                rd = requests.get(aurl, timeout=60)
                out_path = os.path.join(OUTPUT_DIR, "pipeline_large_4seg.mp3")
                with open(out_path, "wb") as f:
                    f.write(rd.content)
                total_elapsed = time.time() - t0
                print(f"    Saved: {out_path} ({len(rd.content)/1024:.0f} KB)")
                print(f"    Total time: {total_elapsed:.1f}s")

                # Transcribe edit regions
                print(f"\n[4] Transcribing edit regions...")
                try:
                    import torchaudio
                    import torch
                    import tempfile
                    import wave
                    import whisperx

                    model = whisperx.load_model("base", device="cuda", compute_type="float16", language="en")
                    w, sr = torchaudio.load(out_path)
                    print(f"    Duration: {w.shape[1]/sr:.1f}s")

                    regions = [
                        (0.0, 12.0, "Edit 1"),
                        (38.0, 60.0, "Edit 2"),
                        (58.0, 82.0, "Edit 3"),
                        (95.0, 118.0, "Edit 4"),
                    ]
                    for start, end, desc in regions:
                        s = int(start * sr)
                        e = min(int(end * sr), w.shape[1])
                        clip = w[:, s:e]
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                            with wave.open(tmp.name, "wb") as wf:
                                wf.setnchannels(1)
                                wf.setsampwidth(2)
                                wf.setframerate(sr)
                                pcm = (clip.squeeze(0).clamp(-1.0, 1.0) * 32767.0).to(torch.int16).numpy().tobytes()
                                wf.writeframes(pcm)
                            audio = whisperx.load_audio(tmp.name)
                            result = model.transcribe(audio, batch_size=4)
                            text = " ".join(s["text"] for s in result["segments"]).strip()
                            os.unlink(tmp.name)
                        print(f"\n    [{start:.0f}-{end:.0f}s] {desc}:")
                        print(f"      {text}")
                except Exception as e:
                    print(f"    Transcription failed: {e}")
                    print(f"    Download the file manually to verify: {aurl}")
        else:
            print(f"    Error: {pdata.get('error', 'unknown')}")
            print(json.dumps(pdata, indent=2, default=str))
        break

httpd.shutdown()
print(f"\n{'='*70}")
print(f"Output: {OUTPUT_DIR}/pipeline_large_4seg.mp3")
print(f"{'='*70}")
