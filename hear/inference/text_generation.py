from __future__ import annotations

from pathlib import Path
from threading import Lock


class DisabledTextGenerationEngine:
    @property
    def is_available(self) -> bool:
        return False

    def generate_sync(self, messages: list[dict], max_tokens: int) -> str:
        raise RuntimeError("text_generation_disabled")


class VllmTextGenerationEngine:
    def __init__(
        self,
        model_path: Path,
        *,
        gpu_memory_utilization: float = 0.82,
        max_model_len: int = 8192,
    ) -> None:
        from vllm import LLM

        self._lock = Lock()
        self._model = LLM(
            model=str(model_path),
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )

    @property
    def is_available(self) -> bool:
        return True

    def generate_sync(self, messages: list[dict], max_tokens: int) -> str:
        from vllm import SamplingParams

        with self._lock:
            outputs = self._model.chat(
                messages,
                sampling_params=SamplingParams(
                    max_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                ),
                use_tqdm=False,
            )
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("text_generation_empty_result")
        return outputs[0].outputs[0].text.strip()
