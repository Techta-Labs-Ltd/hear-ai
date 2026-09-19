"""Exercise live Fish Speech reconstruction without uploading the result to B2."""

import asyncio
import os
import shutil
from pathlib import Path

import ray
from ray import serve

from hear.config import settings
from hear.services.model_client import RayModelClient, set_model_client
from hear.services.reconstruction.synthesizer import SpeechSynthesizer

AUDIO_ROOT = Path(settings.HEAR_TEMP_DIR)
INPUT = Path(
    os.environ.get(
        "HEAR_REGENERATION_INPUT",
        str(AUDIO_ROOT / "hear-regeneration-test.wav"),
    )
)
OUTPUT = Path(
    os.environ.get(
        "HEAR_REGENERATION_OUTPUT",
        str(AUDIO_ROOT / "hear-regeneration-output.mp3"),
    )
)
SEGMENT_START = float(os.environ.get("HEAR_REGENERATION_START", "5.0"))
SEGMENT_END = float(os.environ.get("HEAR_REGENERATION_END", "17.0"))
NEW_TEXT = os.environ.get(
    "HEAR_REGENERATION_TEXT",
    (
        "This is a longer live audio regeneration test designed to verify that "
        "the replacement keeps the original speaker's pace, pitch, tone, and "
        "natural transition when it is joined back into the surrounding recording."
    ),
)
ORIGINAL_TEXT = os.environ.get(
    "HEAR_REGENERATION_ORIGINAL_TEXT",
    "A longer section of the original recording is replaced for this live check.",
)


class LocalStorage:
    bucket_name = "local-test"

    @staticmethod
    def key(*parts: str) -> str:
        return "/".join(parts)

    def upload_file(self, local_path: str, _key: str, _content_type: str) -> str:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, OUTPUT)
        return OUTPUT.as_uri()


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


async def run_test() -> None:
    if await asyncio.to_thread(_file_size, INPUT) == 0:
        raise FileNotFoundError(INPUT)

    ray.init(address="auto", namespace="serve")
    try:
        fish_handle = serve.get_deployment_handle("fish_speech", app_name="hear")
        transcription_handle = serve.get_deployment_handle(
            "transcription", app_name="hear"
        )
        set_model_client(
            RayModelClient({
                "fish_speech": fish_handle,
                "transcription": transcription_handle,
            })
        )
        synthesizer = SpeechSynthesizer()
        synthesizer.load()
        result = await synthesizer.reconstruct_segments(
            original_audio_path=str(INPUT),
            track_id="live-reconstruct-track",
            storage=LocalStorage(),
            changes=[
                {
                    "segment_start": SEGMENT_START,
                    "segment_end": SEGMENT_END,
                    "new_text": NEW_TEXT,
                    "original_text": ORIGINAL_TEXT,
                }
            ],
            same_speaker=True,
            job_id="live-reconstruct-job",
            run_id="live-reconstruct-run",
        )
        output_size = await asyncio.to_thread(_file_size, OUTPUT)
        if output_size == 0:
            raise RuntimeError(f"No local regeneration output was produced: {OUTPUT}")
        print(f"RESULT={result}")
        print(f"OUTPUT={OUTPUT} SIZE={output_size}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    asyncio.run(run_test())
