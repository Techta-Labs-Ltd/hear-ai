import asyncio
import json
import re
from collections import Counter
from typing import Optional

import httpx
import torch
import warnings
from transformers import pipeline as hf_pipeline
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning, message=".*clean_up_tokenization_spaces.*")
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

        layer1 = await loop.run_in_executor(
            None, self._keyword_layer, transcript, segments or [], data.keyword_rules
        )

        tag_pool = self._build_tag_pool(transcript, data.tags, layer1["scores"])

        # Zero-shot calls MUST run sequentially — the model is NOT thread-safe.
        # Running two zero-shot inferences concurrently corrupts results.
        layer2_cat = await loop.run_in_executor(
            None, self._zero_shot_labels, transcript, data.categories
        )
        layer2_tag = await loop.run_in_executor(
            None, self._zero_shot_labels, transcript, tag_pool
        )

        # OpenAI (network call) and sentiment (different model) can run in parallel
        layer3, sentiment = await asyncio.gather(
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
        words = set()
        for w in re.split(r"[\s\.,;:!?\-\"\'()]+", transcript):
            w = w.lower().strip()
            if len(w) > 3 and w not in _STOPWORDS:
                words.add(w)
        return words

    def _build_tag_pool(self, transcript: str, all_tags: list[str], keyword_scores: dict) -> list[str]:
        tx_words = self._extract_transcript_words(transcript)
        priority: list[str] = []
        seen: set[str] = set()

        for tag in all_tags:
            if tag in keyword_scores and tag not in seen:
                priority.append(tag)
                seen.add(tag)

        for tag in all_tags:
            if tag in seen:
                continue
            tag_words = set(re.findall(r"[a-z]+", tag.lower()))
            if tag_words & tx_words:
                priority.append(tag)
                seen.add(tag)

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
            "You are an intelligent content categorization system.\n\n"
            "Your task is to analyze a transcript and generate:\n"
            "1. Up to 5 highly relevant tags (with # prefix)\n"
            "2. Up to 5 accurate categories (no # prefix)\n\n"
            "Rules:\n"
            "- Base your output ONLY on the core subject of the transcript.\n"
            "- Focus on the main themes, not incidental mentions.\n"
            "- Do NOT include unrelated or weakly related categories.\n"
            "- Avoid generic categories like \"Energy\", \"Technology\", or \"Business\" unless they are clearly central.\n"
            "- Prioritize specificity and relevance over broadness.\n\n"
            "Tag Guidelines:\n"
            "- Tags must be concise and descriptive (e.g., #Wildlife, #Photography, #Awards)\n"
            "- Prefer commonly used, human-readable tags\n"
            "- Avoid duplicates or near-duplicates\n\n"
            "Category Guidelines:\n"
            "- Categories should represent high-level domains (e.g., Wildlife, Photography, Film, Nature, Awards)\n"
            "- Only include categories that are strongly supported by the transcript\n"
            "- Do NOT infer categories that are not clearly present\n\n"
            "CRITICAL EXCEPTION:\n"
            "- If the transcript is very short (e.g., under 15 words), purely conversational, a sudden threat, or lacks any distinct topic, you MUST return empty arrays [] for both tags and categories. Do not force tags.\n\n"
            "Output format (STRICT JSON):\n"
            "{\n"
            "  \"tags\": [\"#Tag1\", \"#Tag2\", \"#Tag3\", \"#Tag4\", \"#Tag5\"],\n"
            "  \"categories\": [\"Category1\", \"Category2\", \"Category3\", \"Category4\", \"Category5\"],\n"
            "  \"confidence\": \"low | medium | high\"\n"
            "}\n\n"
            f"Transcript:\n{transcript[:3000]}"
        )

        try:
            async with httpx.AsyncClient(timeout=45) as client:
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
                        "max_tokens": 300,
                    },
                )
                response.raise_for_status()
                data = response.json()

            content = data["choices"][0]["message"]["content"].strip()
            content = content.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(content)

            confidence_map = {"low": 0.5, "medium": 0.75, "high": 0.95}
            confidence = confidence_map.get(str(parsed.get("confidence", "medium")).lower(), 0.75)

            scores: dict[str, float] = {}

            for tag in parsed.get("tags", []):
                tag = tag.strip()
                if tag.startswith("#") and len(tag) > 1:
                    scores[tag] = confidence

            suggested_categories: list[str] = []
            for cat in parsed.get("categories", []):
                cat = cat.strip()
                if not cat:
                    continue
                scores[cat] = confidence
                if cat not in categories:
                    suggested_categories.append(cat)

            return {
                "scores": scores,
                "suggested_tags": [t for t in parsed.get("tags", []) if str(t).startswith("#")],
                "suggested_categories": suggested_categories,
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

        has_openai = len(l3) > 0

        merged_tag_scores: dict[str, float] = {}
        for tag in known_tags:
            s1 = l1.get(tag, 0)
            s2 = l2t.get(tag, 0)
            s3 = l3.get(tag, 0)
            
            if has_openai:
                score = (s1 * 0.4) + (s2 * 0.2) + (s3 * 0.6)
                if s1 > 0 and s3 > 0: score += 0.15
                elif s2 > 0 and s3 > 0: score += 0.10
            else:
                score = (s1 * 0.6) + (s2 * 0.4)
                if s1 > 0 and s2 > 0: score += 0.15
                
            merged_tag_scores[tag] = round(min(1.0, score), 4)

        ranked_tags = sorted(merged_tag_scores.items(), key=lambda x: x[1], reverse=True)

        tags = [t for t, s in ranked_tags if s >= self._TAG_THRESHOLD][:max_tags]
        if not tags and ranked_tags:
            tags = [t for t, _ in ranked_tags[:min(max_tags, 2)]]

        cat_scores: dict[str, float] = {}
        for c in all_categories:
            s2 = l2c.get(c, 0)
            s3 = l3.get(c, 0)
            
            if has_openai:
                score = (s2 * 0.3) + (s3 * 0.7)
                if s2 > 0 and s3 > 0: score += 0.15
            else:
                score = s2 * 1.0
                
            cat_scores[c] = round(min(1.0, score), 4)

        ranked_cats = sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)
        categories = [c for c, s in ranked_cats if s >= self._CAT_THRESHOLD][:3]
        if not categories and ranked_cats:
            categories = [c for c, _ in ranked_cats[:1]]

        print(f"[CATEGORIZER] top_tag_scores={ranked_tags[:8]}")
        print(f"[CATEGORIZER] top_cat_scores={sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)[:5]}")

        return {
            "tags": tags,
            "categories": categories,
            "confidence_scores": {**merged_tag_scores, **cat_scores},
        }
