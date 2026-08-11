import asyncio
import json
from hear.core.downloader import download_audio
from hear.services.registry import transcriber

AUDIO_URL = "https://media.hear.surf/pipeline-source-mp3/8ad18866-e1c4-4055-9064-151958b0f8c3/1faaf4fe-717c-4180-a304-1b13743b7d0f-824a7377-e31e-4858-86bb-2505bf2c659e.mp3"

async def main():
    print("Downloading audio...")
    tmp_path = await download_audio(AUDIO_URL, suffix=".wav")
    print(f"Downloaded to {tmp_path}")
    
    print("Transcribing...")
    with open(tmp_path, "rb") as f:
        wav_bytes = f.read()
    
    transcriber.load()
    result = await transcriber.transcribe(wav_bytes)
    
    with open("/workspace/hear-ai/tests/transcript_dump.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print("Transcript saved to /workspace/hear-ai/tests/transcript_dump.json")

if __name__ == "__main__":
    asyncio.run(main())
