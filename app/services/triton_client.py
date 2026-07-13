import asyncio
import json
import logging
from typing import Optional

from ray import serve

logger = logging.getLogger(__name__)

_client: Optional["RayModelClient"] = None


def set_triton_client(client: "RayModelClient") -> None:
    global _client
    _client = client


def get_triton_client() -> "RayModelClient":
    if _client is None:
        raise RuntimeError("RayModelClient not initialized")
    return _client


class RayModelClient:
    def __init__(self) -> None:
        self._handles: dict[str, object] = {}

    def _get_handle(self, name: str):
        if name not in self._handles:
            self._handles[name] = serve.get_deployment_handle(name, "default")
        return self._handles[name]

    def _resolve_sync(self, ref):
        import concurrent.futures
        try:
            loop = asyncio.get_running_loop()
            fut = concurrent.futures.Future()
            def _resolve():
                try:
                    fut.set_result(ref.result())
                except Exception as e:
                    fut.set_exception(e)
            import threading
            t = threading.Thread(target=_resolve, daemon=True)
            t.start()
            return fut.result()
        except RuntimeError:
            return ref.result()

    @staticmethod
    async def _resolve(ref):
        return await ref

    async def transcribe_async(self, audio_bytes: bytes, batch_size: int = 36) -> dict:
        handle = self._get_handle("transcription")
        raw = await handle.transcribe.remote(audio_bytes, batch_size)
        return json.loads(raw)

    async def transcribe(self, audio_bytes: bytes, batch_size: int = 36) -> dict:
        handle = self._get_handle("transcription")
        raw = await handle.transcribe.remote(audio_bytes, batch_size)
        return json.loads(raw)

    async def llm_generate(self, messages: list[dict], max_tokens: int = 512) -> str:
        handle = self._get_handle("llm")
        return await handle.generate.remote(messages, max_tokens)

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

    async def nli(self, text: str, candidates: list[str], hypothesis_template: str = None) -> dict:
        handle = self._get_handle("small_models")
        request: dict = {
            "model_name": "nli", "text": text,
            "candidates": candidates, "hypothesis_template": hypothesis_template,
        }
        return await handle.remote(request)

    async def enhance_audio(
        self, model_name: str, audio_bytes: bytes, sample_rate: int = 48000, atten_lim_db: float = 12.0,
    ) -> bytes:
        if model_name == "deepfilternet":
            return await self._get_handle("deepfilternet").enhance.remote(audio_bytes, sample_rate, atten_lim_db)
        elif model_name == "mossformer2":
            return await self._get_handle("mossformer2").enhance.remote(audio_bytes, sample_rate)
        return b""

    def transcribe_sync(self, audio_bytes: bytes, batch_size: int = 36) -> dict:
        handle = self._get_handle("transcription")
        return json.loads(self._resolve_sync(handle.transcribe.remote(audio_bytes, batch_size)))

    def enhance_audio_sync(self, model_name: str, audio_bytes: bytes, sample_rate: int = 48000, atten_lim_db: float = 12.0) -> bytes:
        if model_name == "deepfilternet":
            return self._resolve_sync(self._get_handle("deepfilternet").enhance.remote(audio_bytes, sample_rate, atten_lim_db))
        elif model_name == "mossformer2":
            return self._resolve_sync(self._get_handle("mossformer2").enhance.remote(audio_bytes, sample_rate))
        return b""

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

    async def unload_all(self) -> None:
        self._handles.clear()
