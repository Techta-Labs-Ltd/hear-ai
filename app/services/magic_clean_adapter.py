from dataclasses import asdict, is_dataclass

from app.services.enhancer import AudioEnhancer


class MagicCleanEnhancer:
    def __init__(self, inner: AudioEnhancer | None = None):
        self._inner = inner or AudioEnhancer()

    def load(self):
        if not self._inner.is_loaded:
            self._inner.load()

    @property
    def is_loaded(self) -> bool:
        return self._inner.is_loaded

    async def enhance(
        self,
        input_path: str,
        track_id: str,
        job_id: str,
        ai_job_id: str | None = None,
        ai_run_id: str | None = None,
        on_stage=None,
        on_progress=None,
        **kwargs,
    ) -> dict:
        if on_stage:
            on_stage("enhancing")
        if on_progress:
            on_progress("enhancing", 0, 1, 0.0)

        result = await self._inner.enhance(
            input_path=input_path,
            track_id=track_id,
            job_id=job_id,
            ai_job_id=ai_job_id,
            ai_run_id=ai_run_id,
        )

        if on_progress:
            on_progress("output_fade", 0, 1, 100.0)

        return asdict(result) if is_dataclass(result) else result
