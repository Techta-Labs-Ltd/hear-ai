import gc
import logging
from typing import Any

import torch
from ray import serve
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from hear.config import settings
from hear.core.blocking import NativeWorker

logger = logging.getLogger(__name__)

@serve.deployment(
    name="small_models",
    ray_actor_options={"num_gpus": 0.10, "num_cpus": 0.3},
    num_replicas=1,
    max_ongoing_requests=1,
    max_queued_requests=4,
    health_check_period_s=10,
    health_check_timeout_s=30,
)
class SmallModelsDeployment:
    def __init__(self) -> None:
        self._worker = NativeWorker("small-models")
        logger.info("Loading toxic-bert ...")
        self._toxic = pipeline(
            "text-classification",
            model=settings.TOXIC_MODEL_PATH,
            device=0,
            model_kwargs={"local_files_only": True},
        )
        logger.info("Loading sentiment model ...")
        self._sentiment = pipeline(
            "sentiment-analysis",
            model=settings.SENTIMENT_MODEL_PATH,
            device=0,
            model_kwargs={"local_files_only": True},
        )
        logger.info("Loading NLI model ...")
        self._nli = pipeline(
            "zero-shot-classification",
            model=settings.NLI_MODEL_PATH,
            device=0,
            model_kwargs={"local_files_only": True},
        )
        logger.info("small_models all loaded")

    async def __call__(self, request: dict) -> dict:
        return await self._worker.run(self._infer, request)

    def _infer(self, request: dict) -> dict:
        model_name: str = request.get("model_name", "")
        text: str = request.get("text", "")
        candidates: list[str] | None = request.get("candidates")
        hypothesis_template: str | None = request.get("hypothesis_template")
        try:
            if model_name == "toxic_bert":
                result = self._toxic(text[:512], truncation=True)
                return {"labels": [r["label"] for r in result], "scores": [r["score"] for r in result]}
            elif model_name == "sentiment":
                result = self._sentiment(text[:512], truncation=True)
                return {"labels": [r["label"] for r in result], "scores": [r["score"] for r in result]}
            elif model_name == "nli":
                nli_kwargs: dict[str, Any] = {}
                if hypothesis_template:
                    nli_kwargs["hypothesis_template"] = hypothesis_template
                result = self._nli(text[:1024], candidates or [], **nli_kwargs)
                return {"labels": result["labels"], "scores": result["scores"]}
            raise ValueError("unsupported_small_model")
        except Exception as e:
            logger.error("small_models inference error: %s", e)
            raise

    async def close(self) -> None:
        if hasattr(self, "_worker"):
            await self._worker.close()
        self._release()

    def _release(self) -> None:
        for attr in ("_toxic", "_sentiment", "_nli"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        torch.cuda.empty_cache()

    def __del__(self) -> None:
        if hasattr(self, "_worker"):
            self._worker.shutdown()
        self._release()


@serve.deployment(
    name="llm",
    ray_actor_options={"num_gpus": 0.25, "num_cpus": 0.5},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 1,
        "target_num_ongoing_requests_per_replica": 1,
    },
    max_ongoing_requests=1,
    max_queued_requests=4,
    health_check_period_s=30,
    health_check_timeout_s=600,
    graceful_shutdown_timeout_s=60,
)
class LLMDeployment:
    def __init__(self) -> None:
        self._worker = NativeWorker("llm")
        logger.info("Loading Qwen2.5-7B-Instruct (4-bit quantized) ...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            settings.LLM_MODEL_PATH,
            local_files_only=True,
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            settings.LLM_MODEL_PATH,
            device_map="auto",
            quantization_config=quantization_config,
            local_files_only=True,
        )
        logger.info("LLM ready")

    async def generate(self, messages: list[dict], max_tokens: int) -> str:
        return await self._worker.run(self._generate, messages, max_tokens)

    def _generate(self, messages: list[dict], max_tokens: int) -> str:
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = self._model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=0.7, top_p=0.9, do_sample=True,
            )
        response = self._tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
        )
        return response.strip()

    async def generate_batch(self, items: list[tuple[list[dict], int]]) -> list[str]:
        if len(items) > 8:
            raise ValueError("llm_batch_too_large")
        results = []
        for messages, max_tokens in items:
            results.append(await self.generate(messages, max_tokens))
        return results

    async def close(self) -> None:
        if hasattr(self, "_worker"):
            await self._worker.close()
        self._release()

    def _release(self) -> None:
        for attr in ("_model", "_tokenizer"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        torch.cuda.empty_cache()

    def __del__(self) -> None:
        if hasattr(self, "_worker"):
            self._worker.shutdown()
        self._release()
