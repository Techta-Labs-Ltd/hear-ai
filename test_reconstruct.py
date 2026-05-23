import asyncio
import os
import sys

# Add hear-ai to path so imports work from here
sys.path.insert(0, "/workspace/hear-ai")

# Force offline mode for the test
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from app.services.synthesizer import SpeechSynthesizer
from app.core.storage import storage

async def main():
    print("Initializing SpeechSynthesizer...")
    synth = SpeechSynthesizer()
    synth.load()
    
    if not synth.higgs_available:
        print("ERROR: Higgs Audio is not available!")
        return
        
    print("Higgs Audio loaded successfully!")
    print("Running a mock reconstruct job...")
    
    # We use a short sample audio file
    original_audio = "/workspace/higgs-audio/examples/voice_prompts/en_woman.wav"
    
    # We'll replace seconds 1.0 to 2.5 with a new generated speech
    changes = [
        {
            "segment_start": 1.0,
            "segment_end": 2.5,
            "new_text": "This is a successful offline reconstruct test."
        }
    ]
    
    # Mock storage upload so we don't upload test artifacts to your production cloud bucket
    original_upload = storage.upload_file
    def mock_upload(local_path, b2_key):
        print(f"-> MOCKED UPLOAD: Saved reconstructed audio locally to: {local_path}")
        return f"mock_url://{b2_key}"
    
    storage.upload_file = mock_upload
    
    try:
        result = await synth.reconstruct_segments(
            original_audio_path=original_audio,
            track_id="test-offline-reconstruct-123",
            changes=changes,
            same_speaker=True
        )
        
        print("\n=== RECONSTRUCT TEST PASSED ===")
        print(f"Generated Audio Duration: {result.duration}s")
        print(f"Mocked Cloud URL: {result.audio_url}")
        
    except Exception as e:
        print(f"\n=== RECONSTRUCT TEST FAILED ===")
        import traceback
        traceback.print_exc()
    finally:
        storage.upload_file = original_upload

if __name__ == "__main__":
    asyncio.run(main())
