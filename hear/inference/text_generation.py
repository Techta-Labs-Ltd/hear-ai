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
        gpu_memory_gib: float,
        max_model_len: int = 8192,
    ) -> None:
        from vllm import LLM

        self._lock = Lock()
        self._model = LLM(
            model=str(model_path),
            trust_remote_code=True,
            gpu_memory_utilization=self.utilization_for(gpu_memory_gib),
            max_model_len=max_model_len,
        )

    @staticmethod
    def utilization_for(gpu_memory_gib: float, total_bytes: int | None = None) -> float:
        # vLLM only takes a fraction of the card; convert so the budget is card-independent
        # and the rest of the pipeline (ASR, small models) keeps its share.
        if total_bytes is None:
            import torch

            total_bytes = torch.cuda.get_device_properties(0).total_memory
        utilization = gpu_memory_gib * 1024**3 / total_bytes
        if not 0 < utilization <= 0.9:
            raise ValueError("qwen_llm_memory_budget_does_not_fit_gpu")
        return round(utilization, 4)

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
