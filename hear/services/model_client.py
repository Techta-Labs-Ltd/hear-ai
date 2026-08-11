import asyncio
import concurrent.futures
import json
import logging
import threading

logger = logging.getLogger(__name__)

_client = None


def set_model_client(client: "RayModelClient") -> None:
    global _client
    _client = client


def get_model_client() -> "RayModelClient":
    if _client is None:
        raise RuntimeError("RayModelClient not initialized")
    return _client


class RayModelClient:
    def __init__(self, handles: dict[str, object]) -> None:
        self._handles = dict(handles)

    def _get_handle(self, name: str):
        try:
            return self._handles[name]
        except KeyError as exc:
            raise RuntimeError(f"Ray model handle was not injected: {name}") from exc

    def _resolve_sync(self, ref):
        try:
            loop = asyncio.get_running_loop()
            fut = concurrent.futures.Future()
            def _resolve():
                try:
                    fut.set_result(ref.result())
                except Exception as e:
                    fut.set_exception(e)
            t = threading.Thread(target=_resolve, daemon=True)
            t.start()
            return fut.result()
        except RuntimeError:
            return ref.result()

    @staticmethod
    async def _resolve(ref):
        return await ref

    async def transcribe(self, audio_bytes: bytes, batch_size: int = 36) -> dict:
        handle = self._get_handle("transcription")
        raw = await handle.transcribe.remote(audio_bytes, batch_size)
        return json.loads(raw)

    async def small_model_infer(
        self, model_name: str, text: str, candidates=None,
    ) -> dict:
        handle = self._get_handle("small_models")
        request: dict = {"model_name": model_name, "text": text, "candidates": candidates}
        return await handle.remote(request)

    async def moderate(self, text: str) -> dict:
        return await self.small_model_infer("toxic_bert", text)

    async def sentiment(self, text: str) -> dict:
        return await self.small_model_infer("sentiment", text)

    async def generate_speech(
        self,
        *,
        text: str,
        max_new_tokens: int = 1024,
        references: list[dict] | None = None,
        reference_id: str | None = None,
        language: str = "en",
    ) -> bytes:
        return await self._get_handle("fish_speech").generate_speech.remote(
            text,
            max_new_tokens,
            references,
            reference_id,
            language,
        )

    def transcribe_sync(self, audio_bytes: bytes, batch_size: int = 36) -> dict:
        handle = self._get_handle("transcription")
        return json.loads(self._resolve_sync(handle.transcribe.remote(audio_bytes, batch_size)))

    def moderate_sync(self, text: str) -> dict:
        req = {"model_name": "toxic_bert", "text": text, "candidates": None}
        return self._resolve_sync(self._get_handle("small_models").remote(req))

    def nli_sync(self, text: str, candidates: list[str], hypothesis_template: str = None) -> dict:
        req = {"model_name": "nli", "text": text, "candidates": candidates, "hypothesis_template": hypothesis_template}
        return self._resolve_sync(self._get_handle("small_models").remote(req))

    def sentiment_sync(self, text: str) -> dict:
        req = {"model_name": "sentiment", "text": text, "candidates": None}
        return self._resolve_sync(self._get_handle("small_models").remote(req))

    def llm_generate_sync(self, messages: list[dict], max_tokens: int = 512) -> str:
        return self._resolve_sync(self._get_handle("llm").generate.remote(messages, max_tokens))
