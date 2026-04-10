import asyncio

import torch

from app.config import settings


class GPUManager:
    def __init__(self):
        self._semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_GPU_JOBS)
        self._active_jobs = 0
        self._queued_jobs = 0

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def is_available(self) -> bool:
        return torch.cuda.is_available()

    @property
    def gpu_name(self) -> str:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "CPU"

    @property
    def active_jobs(self) -> int:
        return self._active_jobs

    @property
    def queued_jobs(self) -> int:
        return self._queued_jobs

    async def acquire(self):
        self._queued_jobs += 1
        await self._semaphore.acquire()
        self._queued_jobs -= 1
        self._active_jobs += 1

    async def release(self):
        self._active_jobs -= 1
        self._semaphore.release()


gpu = GPUManager()
