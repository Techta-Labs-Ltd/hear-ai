from __future__ import annotations

from typing import Protocol


class SmallModelsProtocol(Protocol):
    def infer_sync(
        self,
        model_name: str,
        text: str,
        candidates: list[str] | None = None,
        hypothesis_template: str | None = None,
        *,
        multi_label: bool = False,
    ) -> dict: ...

    def toxicity_batch_sync(self, texts: list[str]) -> list[dict]: ...


class TextGenerationProtocol(Protocol):
    @property
    def is_available(self) -> bool: ...

    def generate_sync(self, messages: list[dict], max_tokens: int) -> str: ...


class SpeechGenerationProtocol(Protocol):
    async def generate_speech(
        self,
        *,
        text: str,
        max_new_tokens: int = 1024,
        references: list[dict] | None = None,
        reference_id: str | None = None,
        language: str = "en",
        seed: int | None = None,
    ) -> bytes: ...


class LocalInferenceClient:
    def __init__(
        self,
        *,
        small_models: SmallModelsProtocol | None = None,
        text_generation: TextGenerationProtocol | None = None,
        speech_generation: SpeechGenerationProtocol | None = None,
    ) -> None:
        self._small_models = small_models
        self._text_generation = text_generation
        self._speech_generation = speech_generation

    def moderate_sync(self, text: str) -> dict:
        return self._require_small_models().infer_sync("toxic_bert", text)

    def moderate_batch_sync(self, texts: list[str]) -> list[dict]:
        return self._require_small_models().toxicity_batch_sync(texts)

    def nli_sync(
        self,
        text: str,
        candidates: list[str],
        hypothesis_template: str | None = None,
        *,
        multi_label: bool = False,
    ) -> dict:
        return self._require_small_models().infer_sync(
            "nli",
            text,
            candidates,
            hypothesis_template,
            multi_label=multi_label,
        )

    def sentiment_sync(self, text: str) -> dict:
        return self._require_small_models().infer_sync("sentiment", text)

    def llm_generate_sync(self, messages: list[dict], max_tokens: int = 512) -> str:
        if self._text_generation is None or not self._text_generation.is_available:
            raise RuntimeError("text_generation_disabled")
        return self._text_generation.generate_sync(messages, max_tokens)

    @property
    def llm_available(self) -> bool:
        return self._text_generation is not None and self._text_generation.is_available

    async def generate_speech(
        self,
        *,
        text: str,
        max_new_tokens: int = 1024,
        references: list[dict] | None = None,
        reference_id: str | None = None,
        language: str = "en",
        seed: int | None = None,
    ) -> bytes:
        if self._speech_generation is None:
            raise RuntimeError("speech_generation_disabled")
        return await self._speech_generation.generate_speech(
            text=text,
            max_new_tokens=max_new_tokens,
            references=references,
            reference_id=reference_id,
            language=language,
            seed=seed,
        )

    def _require_small_models(self) -> SmallModelsProtocol:
        if self._small_models is None:
            raise RuntimeError("small_models_disabled")
        return self._small_models
