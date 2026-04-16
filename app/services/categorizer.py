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
from app.core.platform_settings import fetch_platform_settings

PRETRAINED_BASE = "cross-encoder/nli-distilroberta-base"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

_STOPWORDS = {
    "the", "and", "for", "with", "has", "was", "are", "his", "her", "they",
    "that", "this", "from", "have", "been", "said", "also", "only", "when",
    "into", "after", "their", "there", "were", "what", "which", "about",
    "will", "would", "could", "should", "over", "some", "all", "more",
    "than", "then", "just", "each", "even", "him", "had", "not", "but",
    "out", "who", "two", "time", "very", "our", "here", "where", "both",
    "other", "than", "those", "these", "him", "its", "year", "years",
}


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
                "new_tags_added": [],
                "new_categories_added": [],
                "settings_applied": False,
            }

        platform = await fetch_platform_settings()
        settings_applied = bool(platform.auto_tag_keywords or platform.blocked_keywords)

        combined_custom = list(custom_tags or [])
        for kw in platform.auto_tag_keywords:
            if kw and kw not in combined_custom:
                combined_custom.append(kw)

        if combined_custom:
            for tag in combined_custom:
                category_loader.add_tag(tag)

        data = category_loader.data
        loop = asyncio.get_event_loop()

        # Step 1: keyword layer first — its results guide the zero-shot tag pool
        layer1 = await loop.run_in_executor(
            None, self._keyword_layer, transcript, segments or [], data.keyword_rules
        )

        # Step 2: zero-shot on categories + a curated tag pool built from keyword hits + transcript words
        tag_pool = self._build_tag_pool(transcript, data.tags, layer1["scores"])

        layer2_cat, layer2_tag, layer3, sentiment = await asyncio.gather(
            loop.run_in_executor(None, self._zero_shot_labels, transcript, data.categories),
            loop.run_in_executor(None, self._zero_shot_labels, transcript, tag_pool),
            self._openai_layer(transcript, data.categories, data.tags),
            loop.run_in_executor(None, self._get_sentiment, transcript),
        )

        merged = self._merge(layer1, layer2_cat, layer2_tag, layer3, data.tags, data.categories, max_tags)

        new_tags_added: list[str] = []
        for tag in merged["tags"]:
            normalised = tag if tag.startswith("#") else f"#{tag}"
            if normalised not in data.tags:
                category_loader.add_tag(normalised)
                new_tags_added.append(normalised)
            elif tag not in data.tags:
                category_loader.add_tag(tag)
                new_tags_added.append(tag)

        new_categories_added: list[str] = []
        for suggested_cat in layer3.get("suggested_categories", []):
            clean = suggested_cat.lstrip("#").strip().title()
            if clean and clean not in data.categories:
                category_loader.add_category(clean)
                new_categories_added.append(clean)

        if platform.blocked_keywords:
            merged["tags"] = [
                t for t in merged["tags"]
                if not any(bk in t.lower() for bk in platform.blocked_keywords)
            ]

        return {
            "tags": merged["tags"],
            "categories": merged["categories"],
            "confidence_scores": merged["confidence_scores"],
            "sentiment": sentiment,
            "new_tags_added": new_tags_added,
            "new_categories_added": new_categories_added,
            "settings_applied": settings_applied,
        }

    # ------------------------------------------------------------------

    def _extract_transcript_words(self, transcript: str) -> set[str]:
        """Return significant lowercase words from the transcript (length > 4, not stopwords)."""
        words = set()
        for w in re.split(r"[\s\.,;:!?\-\"\'()]+", transcript):
            w = w.lower().strip()
            if len(w) > 3 and w not in _STOPWORDS:
                words.add(w)
        return words

    def _build_tag_pool(self, transcript: str, all_tags: list[str], keyword_scores: dict) -> list[str]:
        """
        Build an intelligent tag candidate pool:
        1. Tags already matched by the keyword layer (guaranteed relevant)
        2. Tags whose label words appear in the transcript
        3. Fill remaining slots with a spread across all tag sections
        """
        tx_words = self._extract_transcript_words(transcript)
        priority: list[str] = []
        seen: set[str] = set()

        # Priority 1: keyword layer hits
        for tag in all_tags:
            if tag in keyword_scores and tag not in seen:
                priority.append(tag)
                seen.add(tag)

        # Priority 2: tags whose own text overlaps with transcript words
        for tag in all_tags:
            if tag in seen:
                continue
            tag_words = set(re.findall(r"[a-z]+", tag.lower()))
            if tag_words & tx_words:
                priority.append(tag)
                seen.add(tag)

        # Fill remaining slots: spread evenly across the full tag list
        remaining = [t for t in all_tags if t not in seen]
        fill_slots = max(0, 120 - len(priority))
        if fill_slots and remaining:
            step = max(1, len(remaining) // fill_slots)
            filler = [remaining[i] for i in range(0, len(remaining), step)][:fill_slots]
            priority.extend(filler)

        return priority

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

    _ZS_TEMPLATE = "This audio recording is about {}."

    def _zero_shot_labels(self, transcript: str, labels: list[str]) -> dict:
        if not labels:
            return {"scores": {}}
        output = self._zero_shot(transcript[:1024], labels, hypothesis_template=self._ZS_TEMPLATE)
        return {"scores": dict(zip(output["labels"], output["scores"]))}

    async def _openai_layer(self, transcript: str, categories: list[str], tags: list[str]) -> dict:
        if not settings.OPENAI_API_KEY:
            return {"scores": {}, "suggested_tags": [], "suggested_categories": []}

        prompt = (
            f"Analyze this audio transcript and return a JSON object with three keys:\n"
            f"1. \"categories\": pick the most relevant from [{', '.join(categories[:30])}] with confidence 0-1\n"
            f"2. \"tags\": pick the most relevant from [{', '.join(tags[:50])}] with confidence 0-1, "
            f"plus suggest up to 3 new tags if none fit well (prefix with #)\n"
            f"3. \"new_categories\": list up to 2 new category names (plain strings, no #) "
            f"if the content clearly belongs to a category not in the list\n\n"
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

            return {
                "scores": scores,
                "suggested_tags": [t for t in parsed.get("tags", {}).keys() if t.startswith("#")],
                "suggested_categories": [
                    c.strip() for c in parsed.get("new_categories", [])
                    if isinstance(c, str) and c.strip()
                ],
            }
        except Exception:
            return {"scores": {}, "suggested_tags": [], "suggested_categories": []}

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

    _TAG_THRESHOLD = 0.35
    _CAT_THRESHOLD = 0.40

    def _merge(
        self,
        layer1: dict,
        layer2_cat: dict,
        layer2_tag: dict,
        layer3: dict,
        all_tags: list[str],
        all_categories: list[str],
        max_tags: int,
    ) -> dict:
        l1  = layer1.get("scores", {})
        l2c = layer2_cat.get("scores", {})
        l2t = layer2_tag.get("scores", {})
        l3  = layer3.get("scores", {})

        known_tags = set(all_tags)
        for tag in l3:
            if tag.startswith("#") and tag not in known_tags:
                known_tags.add(tag)

        merged_tag_scores: dict[str, float] = {}
        for tag in known_tags:
            s1 = l1.get(tag, 0)
            s2 = l2t.get(tag, 0)
            s3 = l3.get(tag, 0)
            if s3 > 0:
                merged_tag_scores[tag] = round(s1 * 0.25 + s2 * 0.35 + s3 * 0.40, 4)
            elif s1 > 0 and s2 > 0:
                merged_tag_scores[tag] = round(s1 * 0.45 + s2 * 0.55, 4)
            elif s1 > 0:
                merged_tag_scores[tag] = round(s1 * 0.80, 4)
            else:
                merged_tag_scores[tag] = round(s2, 4)

        ranked_tags = sorted(merged_tag_scores.items(), key=lambda x: x[1], reverse=True)

        tags = [t for t, s in ranked_tags if s >= self._TAG_THRESHOLD][:max_tags]
        if not tags and ranked_tags:
            tags = [t for t, _ in ranked_tags[:min(max_tags, 2)]]

        cat_scores: dict[str, float] = {}
        for c in all_categories:
            s2 = l2c.get(c, 0)
            s3 = l3.get(c, 0)
            cat_scores[c] = round(s2 * 0.4 + s3 * 0.6, 4) if s3 > 0 else round(s2, 4)

        categories = [c for c, s in cat_scores.items() if s >= self._CAT_THRESHOLD]
        if not categories and all_categories:
            categories = [max(all_categories, key=lambda c: cat_scores.get(c, 0))]

        print(f"[CATEGORIZER] top_tag_scores={ranked_tags[:8]}")
        print(f"[CATEGORIZER] top_cat_scores={sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)[:5]}")

        return {
            "tags": tags,
            "categories": categories[:3],
            "confidence_scores": {**merged_tag_scores, **cat_scores},
        }
