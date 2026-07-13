import sys
sys.path.insert(0, '/workspace/hear-ai')

import httpx
import boto3
import tempfile
import os
import subprocess
from app.config import settings


FISH_SPEECH_URL = settings.FISH_SPEECH_TTS_SERVER_URL.rstrip("/")


def generate_speech(text: str) -> bytes:
    payload = {
        "text": text,
        "format": "wav",
        "references": [],
    }
    resp = httpx.post(
        f"{FISH_SPEECH_URL}/v1/tts",
        json=payload,
        timeout=180.0,
    )
    resp.raise_for_status()
    return resp.content


def wav_to_mp3(wav_path: str, title: str) -> str:
    mp3_fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
    os.close(mp3_fd)
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", wav_path,
            "-ac", "2", "-ar", "44100",
            "-c:a", "libmp3lame",
            "-b:a", "128k",
            "-write_xing", "0",
            "-id3v2_version", "3",
            "-metadata", f"title={title}",
            "-metadata", "artist=Hear",
            "-f", "mp3",
            mp3_path,
        ],
        check=True, capture_output=True,
    )
    return mp3_path


def upload_to_b2(local_path: str, remote_key: str, content_type: str) -> str:
    client = boto3.client(
        "s3",
        endpoint_url=settings.B2_ENDPOINT_URL,
        aws_access_key_id=settings.B2_KEY_ID,
        aws_secret_access_key=settings.B2_APPLICATION_KEY,
    )
    client.upload_file(
        local_path,
        settings.B2_BUCKET_NAME,
        remote_key,
        ExtraArgs={"ContentType": content_type},
    )
    return f"{settings.B2_ENDPOINT_URL}/{settings.B2_BUCKET_NAME}/{remote_key}"


feedback_text = (
    "If you enjoyed this, say 'Open Test Development' to rate it, "
    "leave feedback, and support the creator. Up next, the next track."
)

outro_text = (
    "You've reached the end of this content. To keep listening, "
    "open Hear and discover more from your favorite creators. "
    "If you enjoyed what you heard, say 'Open Test Development' to rate it, "
    "leave feedback, and follow the creator. Thanks for listening with Hear."
)

PREFIX = "alexa-playback"

print("Generating Feedback Wrapper...")
fb_wav = generate_speech(feedback_text)
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    f.write(fb_wav)
    fb_wav_path = f.name
fb_mp3 = wav_to_mp3(fb_wav_path, "Feedback Wrapper")
os.unlink(fb_wav_path)
fb_key = f"{PREFIX}/feedback_wrapper.mp3"
fb_url = upload_to_b2(fb_mp3, fb_key, "audio/mpeg")
os.unlink(fb_mp3)
print(f"  -> {fb_url}")

print("Generating Playback Outro...")
ot_wav = generate_speech(outro_text)
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    f.write(ot_wav)
    ot_wav_path = f.name
ot_mp3 = wav_to_mp3(ot_wav_path, "Playback Outro")
os.unlink(ot_wav_path)
ot_key = f"{PREFIX}/playback_outro.mp3"
ot_url = upload_to_b2(ot_mp3, ot_key, "audio/mpeg")
os.unlink(ot_mp3)
print(f"  -> {ot_url}")

print()
print("=== MP3 URLS ===")
print(f"Feedback Wrapper: {fb_url}")
print(f"Playback Outro:   {ot_url}")
