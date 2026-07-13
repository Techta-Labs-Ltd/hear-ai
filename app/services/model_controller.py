import asyncio
import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

STAGE_MODELS: dict[str, set[str]] = {
    "downloading":           set(),
    "transcribing":          {"transcription"},
    "moderating":            {"small_models", "?llm"},
    "categorizing":          {"small_models", "?llm"},
    "discovering":           {"?llm"},
    "enhancing":             {"deepfilternet", "mossformer2"},
    "reconstructing":        {"fish_speech"},
    "rebuilding_audio":      {"fish_speech"},
    "reconstructing_edits":  {"fish_speech"},
    "diffing_transcript":    set(),
    "audio_tagging":         {"transcription"},
}

FISH_SPEECH_API: str = "http://localhost:8080"


class ModelController:
    def __init__(self, server: object = None) -> None:
        self._ref: dict[str, int] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    def set_server(self, server: object) -> None:
        pass

    def _models_for(self, stage: str) -> set[str]:
        return STAGE_MODELS.get(stage, set())

    @staticmethod
    def _model_name(entry: str) -> str:
        return entry.lstrip("?")

    @staticmethod
    def _is_optional(entry: str) -> bool:
        return entry.startswith("?")

    async def prepare(self, stage: str) -> None:
        models = self._models_for(stage)
        if not models:
            return
        async with self._lock:
            for entry in models:
                name = self._model_name(entry)
                if name == "fish_speech":
                    if self._ref.get(name, 0) == 0:
                        await self._do_resume_fish_speech()
                    self._ref[name] = self._ref.get(name, 0) + 1
                    continue
                if self._ref.get(name, 0) > 0:
                    self._ref[name] += 1
                    continue
                self._ref[name] = 1

    async def release(self, stage: str, next_stage: Optional[str] = None) -> None:
        current = self._models_for(stage)
        future = self._models_for(next_stage) if next_stage else set()
        current_names = {self._model_name(m) for m in current} if current else set()
        future_names = {self._model_name(m) for m in future} if future else set()

        to_decr = current_names - future_names - {"fish_speech"}

        async with self._lock:
            for name in to_decr:
                self._ref[name] = max(0, self._ref.get(name, 0) - 1)

            if "fish_speech" in current_names and "fish_speech" not in future_names:
                self._ref["fish_speech"] = max(0, self._ref.get("fish_speech", 0) - 1)
                if self._ref.get("fish_speech", 0) == 0:
                    await self._do_suspend_fish_speech()

            if "fish_speech" in future_names and "fish_speech" not in current_names:
                if self._ref.get("fish_speech", 0) == 0:
                    await self._do_resume_fish_speech()
                self._ref["fish_speech"] = self._ref.get("fish_speech", 0) + 1

    def _do_suspend_fish_speech(self) -> None:
        try:
            r = httpx.post(f"{FISH_SPEECH_API}/v1/model/suspend", timeout=30)
            if r.status_code == 200:
                logger.info("Fish Speech suspended")
        except Exception as e:
            logger.warning("Fish Speech suspend failed: %s", e)

    async def _do_resume_fish_speech(self) -> None:
        try:
            loop = asyncio.get_event_loop()
            def _resume() -> int:
                r = httpx.post(f"{FISH_SPEECH_API}/v1/model/resume", timeout=60)
                return r.status_code
            status = await loop.run_in_executor(None, _resume)
            if status == 200:
                logger.info("Fish Speech resumed")
        except Exception as e:
            logger.warning("Fish Speech resume failed: %s", e)

    async def shutdown(self) -> None:
        async with self._lock:
            self._ref.clear()


def _gpu_free_mem() -> float:
    try:
        import torch
        return torch.cuda.mem_get_info()[0] / (1024 * 1024)
    except Exception:
        return 0.0
