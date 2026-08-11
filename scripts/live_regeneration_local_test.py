"""Exercise live Fish Speech reconstruction without uploading the result to B2."""

import asyncio
import shutil
from pathlib import Path

import ray
from ray import serve

from hear.services.model_client import RayModelClient, set_model_client
from hear.services.reconstruction.synthesizer import SpeechSynthesizer


INPUT = Path("/tmp/hear-regeneration-test.wav")
OUTPUT = Path("/tmp/hear-regeneration-output.mp3")


class LocalStorage:
    bucket_name = "local-test"

    @staticmethod
    def key(*parts: str) -> str:
        return "/".join(parts)

    def upload_file(self, local_path: str, _key: str, _content_type: str) -> str:
        shutil.copy2(local_path, OUTPUT)
        return OUTPUT.as_uri()


async def run_test() -> None:
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    ray.init(address="auto", namespace="serve")
    try:
        fish_handle = serve.get_deployment_handle("fish_speech", app_name="hear")
        set_model_client(RayModelClient({"fish_speech": fish_handle}))
        synthesizer = SpeechSynthesizer()
        synthesizer.load()
        result = await synthesizer.reconstruct_segments(
            original_audio_path=str(INPUT),
            track_id="live-reconstruct-track",
            storage=LocalStorage(),
            changes=[
                {
                    "segment_start": 5.0,
                    "segment_end": 8.0,
                    "new_text": "This is a live audio regeneration test.",
                    "original_text": "A huge team effort has resulted.",
                }
            ],
            same_speaker=True,
            job_id="live-reconstruct-job",
            run_id="live-reconstruct-run",
        )
        if not OUTPUT.is_file() or OUTPUT.stat().st_size == 0:
            raise RuntimeError(f"No local regeneration output was produced: {OUTPUT}")
        print(f"RESULT={result}")
        print(f"OUTPUT={OUTPUT} SIZE={OUTPUT.stat().st_size}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    asyncio.run(run_test())
