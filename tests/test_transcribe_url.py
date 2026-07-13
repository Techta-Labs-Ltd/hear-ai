"""Transcribe an audio URL using the transcriber directly and print word timestamps.

Usage:
    python tests/test_transcribe_url.py <audio_url>
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


async def main():
    audio_url = sys.argv[1] if len(sys.argv) > 1 else None
    if not audio_url:
        print("Usage: python test_transcribe_url.py <audio_url>")
        sys.exit(1)

    print(f"Audio URL: {audio_url[:100]}...")
    print()

    # Import services
    from app.services.registry import transcriber
    from app.core.downloader import download_audio

    # Download
    print("Downloading audio...")
    audio_path = await download_audio(audio_url)
    print(f"Downloaded to: {audio_path}")

    # Transcribe
    print("Transcribing with Faster-Whisper...")
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    result = await transcriber.transcribe(audio_bytes)

    transcript = result.get("transcript", "")
    language = result.get("language", "unknown")
    duration = result.get("duration", 0)
    words = result.get("words", [])

    print(f"Language: {language}")
    print(f"Duration: {duration:.1f}s")
    print(f"\nFull transcript:\n{transcript}\n")

    if words:
        print("Word-level timestamps:")
        print(f"{'Start':>8s} - {'End':>8s}   Word")
        print("-" * 40)
        for w in words:
            start = w.get("start", 0)
            end = w.get("end", 0)
            word = w.get("word", "")
            print(f"{start:8.2f} - {end:8.2f}   {word}")

    # Cleanup
    if os.path.exists(audio_path):
        os.remove(audio_path)


if __name__ == "__main__":
    asyncio.run(main())
