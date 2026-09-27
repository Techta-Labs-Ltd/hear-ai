from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
from pathlib import Path

from hear.config import RuntimeSettings
from hear.execution.native import NativeExecutor
from hear.inference.manifest import ModelManifest
from hear.runtime.roles import WorkerRole


class LocalTranscriptionCommand:
    @staticmethod
    def parse_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        parser.add_argument("audio", type=Path)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--model-root", type=Path)
        parser.add_argument("--language", default="en")
        return parser.parse_args()

    @classmethod
    async def run(cls, args: argparse.Namespace) -> int:
        root = Path(__file__).parent.parent
        audio_path = args.audio.resolve(strict=True)
        settings = RuntimeSettings.from_environment(dict(os.environ))
        model_root = args.model_root or settings.model_root
        manifest = ModelManifest(root / "hear" / "model_manifest.json")
        missing = (
            *manifest.license_blockers(WorkerRole.TRANSCRIPTION),
            *manifest.validate_local(model_root, WorkerRole.TRANSCRIPTION),
        )
        if missing:
            raise RuntimeError("transcription models are not ready: " + ", ".join(missing))

        QwenAsrEngine = importlib.import_module("hear.inference.qwen_asr").QwenAsrEngine
        TranscriptionService = importlib.import_module(
            "hear.services.transcription.service"
        ).TranscriptionService

        native = NativeExecutor("local-transcription")
        engine = None
        try:
            engine = QwenAsrEngine(
                model_path=model_root / "qwen3-asr-1.7b",
                aligner_path=model_root / "qwen3-forced-aligner",
                cache_dir=model_root,
                temp_dir=settings.temp_dir,
                dtype=settings.qwen_asr_dtype,
                device_map=settings.qwen_asr_device_map,
                vad_onset=settings.whisper_vad_onset,
                vad_offset=settings.whisper_vad_offset,
                max_batch_size=settings.whisper_batch_size,
                long_audio_batch_size=settings.whisper_long_audio_batch_size,
                chunk_seconds=settings.whisper_chunk_seconds,
            )
            service = TranscriptionService(
                engine,
                chunk_seconds=settings.whisper_chunk_seconds,
                batch_size=settings.whisper_batch_size,
                long_audio_batch_size=settings.whisper_long_audio_batch_size,
                native=native,
            )
            result = await service.transcribe_file(
                str(audio_path),
                language=args.language,
            )
            output = args.output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({"audio": str(audio_path), "output": str(output), **result}))
        finally:
            if engine is not None:
                await engine.close()
            await native.close()
        return 0

    @classmethod
    def main(cls) -> int:
        return asyncio.run(cls.run(cls.parse_args()))


if __name__ == "__main__":
    raise SystemExit(LocalTranscriptionCommand.main())
