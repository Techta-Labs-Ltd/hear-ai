import asyncio
import json

import httpx
import torch
from transformers import pipeline as hf_pipeline

from app.config import settings

MODERATION_MODEL = "unitary/toxic-bert"


class ModerationService:
    def __init__(self):
        self._classifier = None
        self._device = 0 if torch.cuda.is_available() else -1

    def load(self):
        self._classifier = hf_pipeline(
            "text-classification",
            model=MODERATION_MODEL,
            device=self._device,
            top_k=None,
        )

    @property
    def is_loaded(self) -> bool:
        return self._classifier is not None

    async def moderate(self, text: str, blocked_keywords: list[str] = None) -> dict:
        if not text or not text.strip():
            return {"flagged": False, "categories": {}, "scores": {}, "openai": {}, "blocked_words": []}

        loop = asyncio.get_event_loop()
        local_task = loop.run_in_executor(None, self._classify_local, text)
        openai_task = self._classify_openai(text)
        keyword_task = loop.run_in_executor(None, self._check_blocked_keywords, text, blocked_keywords or [])

        local_result, openai_result, blocked_result = await asyncio.gather(local_task, openai_task, keyword_task)

        flagged = local_result["flagged"] or openai_result.get("flagged", False) or blocked_result["flagged"]

        return {
            "flagged": flagged,
            "categories": local_result["categories"],
            "scores": local_result["scores"],
            "openai": openai_result,
            "blocked_words": blocked_result["matched"],
        }

    def _classify_local(self, text: str) -> dict:
        results = self._classifier(text[:512])
        if isinstance(results, list) and results and isinstance(results[0], list):
            results = results[0]

        categories = {}
        scores = {}
        for item in results:
            label = item["label"].lower()
            score = round(item["score"], 4)
            scores[label] = score
            categories[label] = score > 0.5

        return {
            "flagged": any(categories.values()),
            "categories": categories,
            "scores": scores,
        }

    async def _classify_openai(self, text: str) -> dict:
        if not settings.OPENAI_API_KEY:
            return {"flagged": False, "categories": {}, "scores": {}}

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    f"{settings.OPENAI_BASE_URL}/moderations",
                    headers={
                        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={"input": text[:4000]},
                )
                response.raise_for_status()
                data = response.json()

            result = data["results"][0]
            return {
                "flagged": result["flagged"],
                "categories": result["categories"],
                "scores": {k: round(v, 4) for k, v in result["category_scores"].items()},
            }
        except Exception:
            return {"flagged": False, "categories": {}, "scores": {}}

    def _check_blocked_keywords(self, text: str, blocked_keywords: list[str]) -> dict:
        if not blocked_keywords:
            return {"flagged": False, "matched": []}

        text_lower = text.lower()
        matched = [kw for kw in blocked_keywords if kw.lower() in text_lower]

        return {
            "flagged": len(matched) > 0,
            "matched": matched,
        }
