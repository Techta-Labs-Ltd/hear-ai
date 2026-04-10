import asyncio
import json
import re
from collections import Counter
from typing import Optional

import httpx
import torch
from transformers import pipeline as hf_pipeline

from app.config import settings
from app.core.category_loader import category_loader

PRETRAINED_BASE = "cross-encoder/nli-distilroberta-base"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


class CategorizationService:
    def __init__(self):
        self._zero_shot = None
        self._sentiment = None
        self._device = 0 if torch.cuda.is_available() else -1

    def load(self):
        self._zero_shot = hf_pipeline(
            "zero-shot-classification",
            model=PRETRAINED_BASE,
            device=self._device,
            multi_label=True,
        )
        self._sentiment = hf_pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL,
            device=self._device,
        )

    @property
    def is_loaded(self) -> bool:
        return self._zero_shot is not None

    async def categorize(
        self,
        transcript: str,
        segments: Optional[list[dict]] = None,
        custom_tags: Optional[list[str]] = None,
        max_tags: int = 8,
    ) -> dict:
        if not transcript or not transcript.strip():
            return {
                "tags": [],
                "categories": [],
                "confidence_scores": {},
                "sentiment": "neutral",
            }

        if custom_tags:
            for tag in custom_tags:
                category_loader.add_tag(tag)

        data = category_loader.data
        loop = asyncio.get_event_loop()

        layer1_task = loop.run_in_executor(
            None, self._keyword_layer, transcript, segments or [], data.keyword_rules
        )
        layer2_task = loop.run_in_executor(
            None, self._zero_shot_layer, transcript, data.all_labels
        )
        layer3_task = self._openai_layer(transcript, data.categories, data.tags)
        sentiment_task = loop.run_in_executor(None, self._get_sentiment, transcript)

        layer1, layer2, layer3, sentiment = await asyncio.gather(
            layer1_task, layer2_task, layer3_task, sentiment_task
        )
        merged = self._merge(layer1, layer2, layer3, data.tags, data.categories, max_tags)

        for tag in merged["tags"]:
            category_loader.add_tag(tag)

        return {
            "tags": merged["tags"],
            "categories": merged["categories"],
            "confidence_scores": merged["confidence_scores"],
            "sentiment": sentiment,
        }

    def _keyword_layer(self, transcript: str, segments: list[dict], keyword_rules: dict) -> dict:
        text_lower = transcript.lower()
        scores = {}
        for pattern, tag in keyword_rules.items():
            matches = len(re.findall(pattern.lower(), text_lower))
            if matches > 0:
                scores[tag] = min(1.0, matches * 0.15 + 0.4)
        if segments:
            seg_counter = Counter()
            for seg in segments:
                seg_text = seg.get("text", "").lower()
                for pattern, tag in keyword_rules.items():
                    if re.search(pattern.lower(), seg_text):
                        seg_counter[tag] += 1
            for tag, count in seg_counter.items():
                density = count / max(len(segments), 1)
                scores[tag] = round(scores.get(tag, 0) * 0.6 + density * 0.4, 4)
        return {"scores": scores}

    def _zero_shot_layer(self, transcript: str, labels: list[str]) -> dict:
        if not labels:
            return {"scores": {}}
        output = self._zero_shot(
            transcript[:1024],
            labels,
            hypothesis_template="This audio recording is about {}.",
        )
        scores = dict(zip(output["labels"], output["scores"]))
        return {"scores": scores}

    async def _openai_layer(self, transcript: str, categories: list[str], tags: list[str]) -> dict:
        if not settings.OPENAI_API_KEY:
            return {"scores": {}, "suggested_tags": []}

        cat_list = ", ".join(categories[:30])
        tag_list = ", ".join(tags[:50])

        prompt = (
            f"Analyze this audio transcript and return a JSON object with two keys:\n"
            f"1. \"categories\": pick the most relevant from [{cat_list}] with confidence 0-1\n"
            f"2. \"tags\": pick the most relevant from [{tag_list}] with confidence 0-1, "
            f"plus suggest up to 3 new tags if none fit well (prefix with #)\n\n"
            f"Transcript:\n{transcript[:2000]}\n\n"
            f"Return ONLY valid JSON, no explanation."
        )

        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.post(
                    f"{settings.OPENAI_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": settings.OPENAI_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,
                        "max_tokens": 500,
                    },
                )
                response.raise_for_status()
                data = response.json()

            content = data["choices"][0]["message"]["content"].strip()
            content = content.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(content)

            scores = {}
            for cat, conf in parsed.get("categories", {}).items():
                scores[cat] = round(float(conf), 4)
            for tag, conf in parsed.get("tags", {}).items():
                scores[tag] = round(float(conf), 4)

            suggested = [t for t in parsed.get("tags", {}).keys() if t.startswith("#")]

            return {"scores": scores, "suggested_tags": suggested}
        except Exception:
            return {"scores": {}, "suggested_tags": []}

    def _get_sentiment(self, transcript: str) -> str:
        try:
            result = self._sentiment(transcript[:512])[0]
            label = result["label"].lower()
            if "positive" in label:
                return "positive"
            if "negative" in label:
                return "negative"
            return "neutral"
        except Exception:
            return "neutral"

    def _merge(
        self,
        layer1: dict,
        layer2: dict,
        layer3: dict,
        all_tags: list[str],
        all_categories: list[str],
        max_tags: int,
    ) -> dict:
        l1 = layer1.get("scores", {})
        l2 = layer2.get("scores", {})
        l3 = layer3.get("scores", {})

        known_tags = set(all_tags)
        for tag in l3:
            if tag.startswith("#") and tag not in known_tags:
                known_tags.add(tag)

        merged_tag_scores = {}
        for tag in known_tags:
            s1 = l1.get(tag, 0)
            s2 = l2.get(tag, 0)
            s3 = l3.get(tag, 0)
            if s3 > 0:
                merged_tag_scores[tag] = round(s1 * 0.25 + s2 * 0.35 + s3 * 0.40, 4)
            else:
                merged_tag_scores[tag] = round(s1 * 0.45 + s2 * 0.55, 4)

        tags = sorted(
            [t for t, s in merged_tag_scores.items() if s >= 0.35],
            key=lambda t: merged_tag_scores[t],
            reverse=True,
        )[:max_tags]

        cat_scores = {}
        for c in all_categories:
            s2 = l2.get(c, 0)
            s3 = l3.get(c, 0)
            if s3 > 0:
                cat_scores[c] = round(s2 * 0.4 + s3 * 0.6, 4)
            else:
                cat_scores[c] = round(s2, 4)

        categories = [c for c, s in cat_scores.items() if s >= 0.45]
        if not categories and all_categories:
            categories = [max(all_categories, key=lambda c: cat_scores.get(c, 0))]

        return {
            "tags": tags,
            "categories": categories[:3],
            "confidence_scores": {**merged_tag_scores, **cat_scores},
        }
