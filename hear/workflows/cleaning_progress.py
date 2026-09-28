"""Bounded thread-to-async progress without exposing arbitrary model messages."""

import asyncio


class CleaningProgress:
    STAGES = {"denoising": 30.0, "sound_cleanup": 55.0, "mastering": 75.0}

    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.queue: asyncio.Queue[tuple[str, float]] = asyncio.Queue(maxsize=8)

    def publish(self, stage: str, value: float) -> None:
        if stage in self.STAGES:
            self.loop.call_soon_threadsafe(self._put, stage)

    def _put(self, stage: str) -> None:
        if self.queue.full():
            self.queue.get_nowait()
        self.queue.put_nowait((stage, self.STAGES[stage]))
