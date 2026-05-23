import requests
import time
import subprocess
import os

# Start a simple HTTP server to serve the local test file
server_process = subprocess.Popen(
    ["python3", "-m", "http.server", "8080"],
    cwd="/workspace/higgs-audio/examples/voice_prompts",
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

# Give the server a moment to start
time.sleep(1)

try:
    print("Sending API request to reconstruct...")
    # The secret is from the .env file
    headers = {
        "X-Service-Key": "8c361a305f40bfc53b165b41c274b00fdba004445e8473bc0e04da8d0093aca4",
        "Content-Type": "application/json"
    }

    payload = {
        "audio_url": "http://127.0.0.1:8080/belinda.wav",
        "track_id": "test_track_123",
        "same_speaker": True,
        "changes": [
            {
                # We replace a tiny portion of the audio (0.5s to 1.5s)
                # with a new phrase to test the generation and splicing.
                "segment_start": 0.5,
                "segment_end": 1.5,
                "new_text": "This is a brand new test phrase"
            }
        ]
    }

    response = requests.post("http://localhost:8000/api/v1/reconstruct", json=payload, headers=headers)
    
    print(f"Status Code: {response.status_code}")
    try:
        print(response.json())
    except:
        print(response.text)
        
finally:
    server_process.terminate()
