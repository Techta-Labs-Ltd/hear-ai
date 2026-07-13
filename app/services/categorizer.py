import asyncio
import json
import logging
import re
from collections import Counter, defaultdict
from typing import Optional

import httpx
import warnings
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning, message=".*clean_up_tokenization_spaces.*")
from app.config import settings
from app.core.category_loader import (
    category_loader,
    is_hierarchical_taxonomy_path,
    _taxonomy_path_to_tag,
)
from app.core.content_context import (
    assistive_tech_narrative,
    tech_history_narrative,
    wildlife_media_narrative,
)
from app.core.discovery_taxonomy import discovery_taxonomy_loader
from app.core.platform_settings import fetch_platform_settings
from app.services.llm_service import get_llm_service
from app.services.triton_client import get_triton_client

logger = logging.getLogger(__name__)

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

    async def categorize(
        self,
        transcript: str,
        segments: Optional[list[dict]] = None,
        custom_tags: Optional[list[str]] = None,
        max_tags: int = 8,
        per_track_transcripts: Optional[dict[str, str]] = None,
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
                category_loader.add_tag(self._normalize_tag(tag))

        data = category_loader.data
        catalog_cats, catalog_tags = self._expanded_catalog_labels(data)
        loop = asyncio.get_event_loop()

        # ── Multi-track: process each track independently and merge ─────────────
        active_tracks = {
            k: v for k, v in (per_track_transcripts or {}).items()
            if v and v.strip()
        }
        if len(active_tracks) > 1:
            return await self._categorize_multi_track(
                track_texts=active_tracks,
                data=data,
                platform=platform,
                settings_applied=settings_applied,
                max_tags=max_tags,
            )

        layer1 = await loop.run_in_executor(
            None, self._keyword_layer, transcript, segments or [], data.keyword_rules
        )

        tag_pool = self._build_tag_pool(transcript, catalog_tags, layer1["scores"])

        layer2_cat = await loop.run_in_executor(
            None, self._zero_shot_labels, transcript, catalog_cats
        )
        tag_labels = list(tag_pool) if tag_pool else []
        layer2_tag = await loop.run_in_executor(
            None, self._zero_shot_labels, transcript, tag_labels
        ) if tag_labels else {"scores": {}}
        context_cats = self._build_context_category_shortlist(
            transcript,
            catalog_cats,
            layer1["scores"],
            layer2_cat.get("scores", {}),
        )
        nli_top = self._top_nli_categories(layer2_cat.get("scores", {}), limit=6)

        if get_llm_service().is_available:
            try:
                qwen_out = await self._categorize_qwen_primary(
                    transcript,
                    data=data,
                    catalog_tags=catalog_tags,
                    layer1_scores=layer1["scores"],
                    zero_shot_scores=layer2_cat.get("scores", {}),
                    context_cats=context_cats,
                    tag_pool=tag_pool,
                    nli_top=nli_top,
                    max_tags=max_tags,
                    platform=platform,
                    settings_applied=settings_applied,
                    loop=loop,
                )
                if qwen_out is not None:
                    return qwen_out
            except Exception as exc:
                logger.warning("[CATEGORIZER] Qwen failed (%s) — falling back to NLI pipeline", exc)

        # OpenAI layer — skipped when no key set
        if settings.OPENAI_API_KEY:
            layer3, sentiment = await asyncio.gather(
                self._openai_layer(transcript, data.categories, data.tags),
                loop.run_in_executor(None, self._get_sentiment, transcript),
            )
        else:
            layer3 = {"scores": {}, "suggested_tags": [], "suggested_categories": []}
            sentiment = await loop.run_in_executor(None, self._get_sentiment, transcript)

        merged = self._merge(layer1, layer2_cat, layer2_tag, layer3, data.tags, data.categories, max_tags)
        merged["tags"] = self._normalize_tags(merged["tags"])[:max_tags]

        new_tags_added = []
        for tag in merged["tags"]:
            if tag not in self._normalize_tags(data.tags):
                category_loader.add_tag(tag)
                new_tags_added.append(tag)

        new_categories_added: list[str] = []
        for suggested_cat in layer3.get("suggested_categories", []):
            clean = suggested_cat.lstrip("#").strip().title()
            if clean and clean not in data.categories:
                category_loader.add_category(clean)
                new_categories_added.append(clean)

        if platform.blocked_keywords:
            merged["tags"] = self._filter_blocked_tags(merged["tags"], platform.blocked_keywords)
        merged["tags"] = self._ensure_non_empty_tags(merged["tags"], merged["categories"], transcript, max_tags)
        merged["tags"], merged["categories"] = self._sanitize_categorization_labels(
            transcript, merged["tags"], merged["categories"]
        )
        merged["tags"], merged["categories"] = self._apply_editorial_rules(
            transcript,
            merged["tags"],
            merged["categories"],
            max_tags,
        )
        merged["tags"] = self._normalize_tags(merged["tags"])[:max_tags]
        merged["categories"] = merged["categories"][:3]
        persisted_tags, persisted_cats = category_loader.ensure_labels(
            merged["tags"], merged["categories"]
        )
        for t in persisted_tags:
            if t not in new_tags_added:
                new_tags_added.append(t)
        for c in persisted_cats:
            if c not in new_categories_added:
                new_categories_added.append(c)

        return {
            "tags": merged["tags"],
            "categories": merged["categories"],
            "confidence_scores": merged["confidence_scores"],
            "sentiment": sentiment,
            "new_tags_added": new_tags_added,
            "new_categories_added": new_categories_added,
            "settings_applied": settings_applied,
            "llm_used": False,
            "categorizer_mode": "nli",
        }

    # ------------------------------------------------------------------
    # Multi-track
    # ------------------------------------------------------------------

    async def _categorize_multi_track(
        self,
        track_texts: dict[str, str],
        data,
        platform,
        settings_applied: bool,
        max_tags: int,
    ) -> dict:
        """Analyse each track independently then merge results.

        This prevents the longest track (e.g. 3-minute football commentary)
        from drowning out shorter ones (e.g. a 30-second recipe intro or
        a music intro). Every track contributes its own tags and categories.
        """
        loop = asyncio.get_event_loop()
        catalog_cats, catalog_tags = self._expanded_catalog_labels(data)
        all_tags: list[str] = []
        all_categories: list[str] = []
        all_sentiments: list[str] = []
        confidence_scores: dict[str, float] = {}
        new_tags_added: list[str] = []
        new_categories_added: list[str] = []
        llm_was_used = False
        per_track: dict[str, dict] = {}

        for track_id, t_text in track_texts.items():
            if not t_text or not t_text.strip():
                continue

            logger.info(
                "[CATEGORIZER] Multi-track: analysing track %s (%d words)",
                track_id[:16], len(t_text.split()),
            )

            # Keyword layer — always runs, gives Qwen context
            layer1 = await loop.run_in_executor(
                None, self._keyword_layer, t_text, [], data.keyword_rules
            )
            tag_pool = self._build_tag_pool(t_text, catalog_tags, layer1["scores"])
            layer2_cat = await loop.run_in_executor(
                None, self._zero_shot_labels, t_text, catalog_cats
            )
            zs_scores = layer2_cat.get("scores", {})
            context_cats = self._build_context_category_shortlist(
                t_text, catalog_cats, layer1["scores"], zs_scores
            )
            nli_top = self._top_nli_categories(zs_scores, limit=6)

            if get_llm_service().is_available:
                try:
                    qwen_track = await self._categorize_qwen_primary(
                        t_text,
                        data=data,
                        catalog_tags=catalog_tags,
                        layer1_scores=layer1["scores"],
                        zero_shot_scores=zs_scores,
                        context_cats=context_cats,
                        tag_pool=tag_pool,
                        nli_top=nli_top,
                        max_tags=max_tags,
                        platform=platform,
                        settings_applied=settings_applied,
                        loop=loop,
                    )
                    if qwen_track is None:
                        raise RuntimeError("qwen_primary returned no result")
                    t_tags = qwen_track["tags"]
                    t_cats = qwen_track["categories"]
                    track_new_tags = qwen_track.get("new_tags_added", [])
                    track_new_cats = qwen_track.get("new_categories_added", [])
                    t_sent = qwen_track.get("sentiment", "neutral")

                    per_track[track_id] = {"tags": t_tags, "categories": t_cats, "sentiment": t_sent}

                    for tag in t_tags:
                        if tag not in all_tags:
                            all_tags.append(tag)
                            confidence_scores[tag] = 0.85
                    for nt in track_new_tags:
                        if nt not in new_tags_added:
                            new_tags_added.append(nt)
                    for cat in t_cats:
                        if cat not in all_categories:
                            all_categories.append(cat)
                    for nc in track_new_cats:
                        if nc not in new_categories_added:
                            new_categories_added.append(nc)
                    all_sentiments.append(t_sent)
                    llm_was_used = True
                    continue
                except Exception as exc:
                    logger.warning(
                        "[CATEGORIZER] Qwen failed for track %s (%s) — using NLI",
                        track_id[:16], exc,
                    )

            layer2_tag: dict = {"scores": {}}
            layer3: dict = {"scores": {}, "suggested_tags": [], "suggested_categories": []}
            nli_merged = self._merge(layer1, layer2_cat, layer2_tag, layer3, data.tags, data.categories, max_tags)
            t_sent = await loop.run_in_executor(None, self._get_sentiment, t_text)

            t_tags = self._normalize_tags(nli_merged.get("tags", []))
            t_cats = self._finalize_categories(
                t_text,
                nli_merged.get("categories", []),
                zs_scores,
                max_categories=3,
            )
            t_tags, t_cats = self._apply_editorial_rules(t_text, t_tags, t_cats, max_tags)
            per_track[track_id] = {"tags": t_tags, "categories": t_cats, "sentiment": t_sent}

            for tag in t_tags:
                if tag not in all_tags:
                    all_tags.append(tag)
                    confidence_scores[tag] = nli_merged.get("confidence_scores", {}).get(tag, 0.5)
            for cat in t_cats:
                if cat not in all_categories:
                    all_categories.append(cat)
            all_sentiments.append(t_sent)

        # Apply blocked_keywords filter to global list AND per-track breakdown
        if platform.blocked_keywords:
            bk = platform.blocked_keywords
            all_tags = self._filter_blocked_tags(all_tags, bk)
            for tid in per_track:
                per_track[tid]["tags"] = self._filter_blocked_tags(per_track[tid]["tags"], bk)
        all_tags = self._ensure_non_empty_tags(all_tags, all_categories, " ".join(track_texts.values()), max_tags)

        final_sentiment = (
            Counter(all_sentiments).most_common(1)[0][0]
            if all_sentiments else "neutral"
        )

        logger.info(
            "[CATEGORIZER] Multi-track merge: tags=%s categories=%s per_track_count=%d",
            all_tags[:max_tags], all_categories, len(per_track),
        )

        return {
            "tags": all_tags[:max_tags],
            "categories": all_categories,
            "confidence_scores": confidence_scores,
            "sentiment": final_sentiment,
            "new_tags_added": new_tags_added,
            "new_categories_added": new_categories_added,
            "settings_applied": settings_applied,
            "llm_used": llm_was_used,
            "categorizer_mode": "qwen_primary" if llm_was_used else "nli",
            "per_track": per_track,
        }

    def _integrate_llm_categorization(
        self,
        llm_result: dict,
        *,
        catalog_tags: list[str],
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        catalog_norm = {t.lower() for t in self._normalize_tags(catalog_tags)}
        tags_from_list = self._normalize_tags(
            list(llm_result.get("tags", [])) + list(llm_result.get("new_tags", []))
        )
        categories = [
            c.strip()
            for c in list(llm_result.get("categories", []))
            + list(llm_result.get("new_categories", []))
            if isinstance(c, str) and c.strip()
        ]
        seen_cat: set[str] = set()
        deduped_cats: list[str] = []
        for c in categories:
            low = c.lower()
            if low in seen_cat:
                continue
            seen_cat.add(low)
            deduped_cats.append(c)
        categories = deduped_cats
        tags_from_list, categories = self._sanitize_categorization_labels(
            "", tags_from_list, categories
        )

        new_tags_added: list[str] = []
        new_categories_added: list[str] = []

        def _register_new_tag(raw: str) -> None:
            normalised = self._normalize_tag(raw)
            if not normalised or normalised in new_tags_added:
                return
            category_loader.add_tag(normalised)
            new_tags_added.append(normalised)

        for tag in llm_result.get("new_tags", []):
            if isinstance(tag, str):
                _register_new_tag(tag)

        for cat in llm_result.get("new_categories", []):
            if isinstance(cat, str) and cat.strip():
                c = cat.strip()
                category_loader.add_category(c)
                if c not in new_categories_added:
                    new_categories_added.append(c)
                if c not in categories:
                    categories.append(c)

        for tag in tags_from_list:
            if tag.lower() not in catalog_norm:
                _register_new_tag(tag)

        merged_tags: list[str] = []
        seen: set[str] = set()
        for tag in tags_from_list + new_tags_added:
            key = tag.lower()
            if key in seen:
                continue
            seen.add(key)
            merged_tags.append(tag)

        merged_tags = self._normalize_tags(tags_from_list + new_tags_added)
        return merged_tags, categories, new_tags_added, new_categories_added

    def _expanded_catalog_labels(self, data) -> tuple[list[str], list[str]]:
        cats: list[str] = []
        tags: list[str] = []
        seen_c: set[str] = set()
        seen_t: set[str] = set()
        for c in category_loader.flat_catalog_categories():
            key = c.strip().lower()
            if c.strip() and key not in seen_c:
                seen_c.add(key)
                cats.append(c.strip())
        for t in data.tags:
            nt = self._normalize_tag(t)
            if nt and nt.lower() not in seen_t:
                seen_t.add(nt.lower())
                tags.append(nt)
        return cats, tags

    async def _categorize_qwen_primary(
        self,
        transcript: str,
        *,
        data,
        catalog_tags: list[str],
        layer1_scores: dict,
        zero_shot_scores: dict,
        context_cats: list[str],
        tag_pool: list[str],
        nli_top: list[str],
        max_tags: int,
        platform,
        settings_applied: bool,
        loop,
        max_categories: int = 3,
    ) -> dict | None:
        taxonomy_paths = list(discovery_taxonomy_loader.data.paths)
        llm_result = await loop.run_in_executor(
            None,
            lambda: get_llm_service().categorize(
                transcript,
                context_cats[:50],
                tag_pool[:120],
                layer1_scores,
                max_categories=max_categories,
                nli_top_categories=nli_top,
                taxonomy_paths=taxonomy_paths,
            ),
        )
        tags, categories, new_tags_added, new_categories_added = self._integrate_llm_categorization(
            llm_result, catalog_tags=catalog_tags
        )
        tags, categories = self._sanitize_categorization_labels(
            transcript, tags, categories
        )
        tags, categories = self._trail_validate_categorization(
            transcript,
            tags,
            categories,
            zero_shot_scores,
            layer1_scores,
            max_tags=max_tags,
            max_categories=max_categories,
        )
        if platform.blocked_keywords:
            tags = self._filter_blocked_tags(tags, platform.blocked_keywords)
        tags = self._ensure_non_empty_tags(tags, categories, transcript, max_tags)
        tags = self._normalize_tags(tags)[:max_tags]
        categories = categories[:max_categories]
        persisted_tags, persisted_cats = category_loader.ensure_labels(tags, categories)
        for t in persisted_tags:
            if t not in new_tags_added:
                new_tags_added.append(t)
        for c in persisted_cats:
            if c not in new_categories_added:
                new_categories_added.append(c)
        confidence_scores = {t: 0.9 for t in tags}
        return {
            "tags": tags,
            "categories": categories,
            "confidence_scores": confidence_scores,
            "sentiment": llm_result.get("sentiment", "neutral"),
            "new_tags_added": new_tags_added,
            "new_categories_added": new_categories_added,
            "settings_applied": settings_applied,
            "llm_used": True,
            "categorizer_mode": "qwen_primary",
        }

    _ASSISTIVE_FLAT_CATEGORIES = frozenset(
        {
            "accessibility",
            "smart glasses",
            "assistive technology",
            "visual impairment",
            "guide dogs",
            "independence and daily living",
            "blindness and visual impairment",
        }
    )
    _ASSISTIVE_TAG_SLUG_PREFIXES = (
        "#accessibility",
        "#assistive",
        "#smart-glasses",
        "#smartglasses",
        "#guidedog",
        "#guide-dog",
        "#visual-impair",
        "#visually-impair",
    )
    _ENVIRONMENT_COMMUNITY_TERMS = (
        "river",
        "chalk stream",
        "over-extracted",
        "preservation society",
        "water company",
        "water companies",
        "town council",
        "allocated",
        "consultant",
        "stakeholders",
        "sewage treatment",
        "reservoir",
        "infrastructure project",
        "darent",
        "sevenoaks",
        "charity",
        "funded a project",
        "improving the flow",
    )

    def _environment_community_context(self, text: str) -> bool:
        hits = sum(1 for term in self._ENVIRONMENT_COMMUNITY_TERMS if term in text)
        return hits >= 2

    def _sanitize_categorization_labels(
        self,
        transcript: str,
        tags: list[str],
        categories: list[str],
    ) -> tuple[list[str], list[str]]:
        """Tags must be #hashtags; taxonomy paths belong in discovery, not categorization output."""
        text = (transcript or "").lower()
        assistive = self._assistive_tech_narrative_context(text)
        out_tags: list[str] = []
        seen_t: set[str] = set()
        out_cats: list[str] = []
        seen_c: set[str] = set()

        for raw in categories or []:
            cat = re.sub(r"\s+", " ", str(raw or "").strip())
            if not cat:
                continue
            if is_hierarchical_taxonomy_path(cat):
                if not assistive:
                    continue
                cat = cat.split(" > ")[-1].strip()
            low = cat.lower()
            if not assistive and low in self._ASSISTIVE_FLAT_CATEGORIES:
                continue
            if low in seen_c:
                continue
            seen_c.add(low)
            out_cats.append(cat)
            slug = _taxonomy_path_to_tag(raw) if is_hierarchical_taxonomy_path(str(raw)) else ""
            if slug and slug.lower() not in seen_t:
                seen_t.add(slug.lower())
                out_tags.append(slug)

        for raw in tags or []:
            if is_hierarchical_taxonomy_path(str(raw)):
                if not assistive:
                    continue
                slug = _taxonomy_path_to_tag(str(raw))
                if slug and slug.lower() not in seen_t:
                    seen_t.add(slug.lower())
                    out_tags.append(slug)
                continue
            norm = self._normalize_tag(str(raw))
            if not norm:
                continue
            if not assistive and any(
                norm.lower().startswith(p) for p in self._ASSISTIVE_TAG_SLUG_PREFIXES
            ):
                continue
            if norm.lower() in seen_t:
                continue
            seen_t.add(norm.lower())
            out_tags.append(norm)

        return out_tags, out_cats

    def _strip_unrelated_assistive_labels(
        self, text: str, tags: list[str], categories: list[str]
    ) -> tuple[list[str], list[str]]:
        if self._assistive_tech_narrative_context(text):
            return tags, categories
        tags = [
            t
            for t in self._normalize_tags(tags)
            if not any(t.lower().startswith(p) for p in self._ASSISTIVE_TAG_SLUG_PREFIXES)
        ]
        categories = [
            c
            for c in categories
            if c.strip().lower() not in self._ASSISTIVE_FLAT_CATEGORIES
            and not is_hierarchical_taxonomy_path(c)
        ]
        return tags, categories

    def _trail_validate_categorization(
        self,
        transcript: str,
        tags: list[str],
        categories: list[str],
        zero_shot_scores: dict,
        keyword_scores: dict,
        *,
        max_tags: int,
        max_categories: int = 3,
    ) -> tuple[list[str], list[str]]:
        """Light guardrails after Qwen — fill gaps only, do not override good LLM picks."""
        text = (transcript or "").lower()
        tags, categories = self._sanitize_categorization_labels(transcript, tags, categories)
        tags, categories = self._strip_unrelated_assistive_labels(text, tags, categories)
        normalized_tags = self._normalize_tags(tags)
        normalized_categories = [c.strip() for c in (categories or []) if c and c.strip()]

        normalized_categories = self._strip_clinical_categories_for_wildlife_media(
            text, normalized_categories
        )
        normalized_tags, normalized_categories = self._refine_assistive_tech_labels(
            text, normalized_tags, normalized_categories
        )
        normalized_tags, normalized_categories = self._rebalance_subject_over_format(
            transcript, normalized_tags, normalized_categories
        )

        if not normalized_categories and zero_shot_scores:
            ranked = sorted(zero_shot_scores.items(), key=lambda x: x[1], reverse=True)
            for cat, score in ranked:
                if score < 0.45:
                    break
                if cat.lower() in self._FORMAT_CATEGORIES:
                    continue
                if (
                    self._wildlife_media_context(text)
                    and not self._vet_clinical_context(text)
                    and cat.strip().lower() in self._CLINICAL_ANIMAL_CATEGORIES
                ):
                    continue
                if (
                    self._assistive_tech_narrative_context(text)
                    and cat.strip().lower() in self._MISLEADING_CATS_ASSISTIVE
                ):
                    continue
                normalized_categories.append(cat)
                if len(normalized_categories) >= max_categories:
                    break

        if not normalized_tags and keyword_scores:
            for tag, score in sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True):
                if score < 0.35:
                    break
                if (
                    self._assistive_tech_narrative_context(text)
                    and self._normalize_tag(tag).lower() in self._MISLEADING_TAGS_ASSISTIVE
                ):
                    continue
                t = self._normalize_tag(tag)
                if t and t not in normalized_tags:
                    normalized_tags.append(t)
                if len(normalized_tags) >= max_tags:
                    break

        normalized_tags, normalized_categories = self._refine_assistive_tech_labels(
            text, normalized_tags, normalized_categories
        )
        normalized_tags, normalized_categories = self._boost_wildlife_media_labels(
            text, normalized_tags, normalized_categories
        )
        normalized_tags, normalized_categories = self._boost_environment_community_labels(
            text, normalized_tags, normalized_categories
        )
        normalized_tags, normalized_categories = self._strip_unrelated_assistive_labels(
            text, normalized_tags, normalized_categories
        )
        normalized_tags, normalized_categories = self._strip_misleading_tags_for_context(
            text, normalized_tags, normalized_categories
        )
        return self._normalize_tags(normalized_tags)[:max_tags], normalized_categories[:max_categories]

    def _normalize_tag(self, tag: str) -> str:
        if not tag:
            return ""
        clean = str(tag).strip().lower()
        clean = re.sub(r"\s+", "-", clean)
        clean = clean.lstrip("#")
        clean = re.sub(r"[^a-z0-9_\-]", "", clean)
        if not clean:
            return ""
        return f"#{clean}"

    def _normalize_tags(self, tags: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for tag in tags or []:
            normalised = self._normalize_tag(tag)
            if normalised and normalised not in seen:
                out.append(normalised)
                seen.add(normalised)
        return out

    def _filter_blocked_tags(self, tags: list[str], blocked_keywords: list[str]) -> list[str]:
        blocked = [b.lower().strip() for b in (blocked_keywords or []) if b and b.strip()]
        if not blocked:
            return tags
        return [t for t in tags if not any(bk in t.lower() for bk in blocked)]

    def _ensure_non_empty_tags(self, tags: list[str], categories: list[str], transcript: str, max_tags: int) -> list[str]:
        normalised = self._normalize_tags(tags)
        if normalised:
            return normalised[:max_tags]
        category_tags = [self._normalize_tag(c) for c in categories if c]
        category_tags = [t for t in category_tags if t]
        if category_tags:
            return self._normalize_tags(category_tags)[:max_tags]
        words = [w for w in re.findall(r"[a-zA-Z]{4,}", transcript.lower()) if w not in _STOPWORDS]
        if words:
            return self._normalize_tags([words[0]])[:max_tags]
        return []

    _FORMAT_CATEGORIES = frozenset(
        {"podcast", "documentary", "entertainment", "lifestyle", "opinion", "media"}
    )
    _FORMAT_TAGS = frozenset({"#podcast", "#radio", "#broadcast", "#streaming"})
    _CLINICAL_ANIMAL_CATEGORIES = frozenset(
        {
            "veterinary",
            "dentistry",
            "pediatrics",
            "oncology",
            "cardiology",
            "neurology",
            "psychiatry",
            "dermatology",
            "optometry",
            "surgery",
        }
    )
    _VET_CLINICAL_TERMS = (
        "veterinar", "vet clinic", "animal hospital", "pet hospital", "neutering",
        "spay", "vaccination", "deworming", "euthanias", "surgical", "diagnos",
        "antibiotic", "prescription", "kennel cough", "flea treatment",
    )
    _WILDLIFE_MEDIA_TERMS = (
        "wildlife", "wildlife photographer", "wildlife award", "wildlife trust",
        "photographer", "photo competition", "photography competition",
        "graphic designer", "won the", "has won", "award for",
        "country park", "rewilding", "great crested", "grebe",
        "great crested grebe", "bird", "birds", "chicks", "mating",
        "national park", "nature reserve", "sanctuary", "video titled",
    )
    _MISLEADING_TAGS_TECH_HISTORY = frozenset(
        {"#wildlife", "#photography", "#animals", "#nature", "#accessibility", "#guidedogs"}
    )
    _ASSISTIVE_TECH_NARRATIVE_TERMS = (
        "guide dog", "guide dogs", "visually impaired", "visual impairment",
        "sighted people", "sighted person", "meta ray", "ray-ban", "smart glasses",
        "hey meta", "orion", "assistive", "independence isn't", "independence is about",
        "blind", "partially sighted", "object recognition", "ai assistant",
    )
    _MISLEADING_TAGS_ASSISTIVE = frozenset(
        {"#wildlife", "#nature", "#animals", "#technology", "#ai", "#environment"}
    )
    _MISLEADING_CATS_ASSISTIVE = frozenset(
        {"wildlife", "nature", "animals", "technology", "environment", "science"}
    )

    def _rebalance_subject_over_format(
        self,
        transcript: str,
        tags: list[str],
        categories: list[str],
    ) -> tuple[list[str], list[str]]:
        text = (transcript or "").lower()
        normalized_tags = self._normalize_tags(tags)
        normalized_categories = [c.strip() for c in (categories or []) if c and c.strip()]

        subject_rules: list[tuple[tuple[str, ...], str, str, int]] = [
            (
                (
                    "music", "song", "remix", "band", "album", "lyrics", "melody",
                    "singer", "orchestral", "joan of arc", "soundtrack", "single",
                    "ep", "guitar", "piano", "concert", "vinyl",
                ),
                "Music",
                "#music",
                2,
            ),
            (
                ("game", "gaming", "xbox", "playstation", "nintendo", "esports", "truman adventure"),
                "Gaming",
                "#gaming",
                2,
            ),
            (
                ("football", "soccer", "rugby", "cricket", "tennis", "match", "goal", "league"),
                "Sports",
                "#sports",
                3,
            ),
            (
                (
                    "wildlife", "photographer", "photography", "photo competition",
                    "won the", "has won", "award for", "wildlife trust", "grebe",
                    "great crested", "rewilding", "country park", "graphic designer",
                ),
                "Wildlife",
                "#wildlife",
                2,
            ),
            (
                (
                    "guide dog", "guide dogs", "visually impaired", "sighted people",
                    "meta ray", "ray-ban", "smart glasses", "hey meta", "orion",
                    "assistive", "independence isn't", "independence is about",
                ),
                "Personal lived experience",
                "#accessibility",
                2,
            ),
            (
                (
                    "river", "chalk stream", "preservation society", "water company",
                    "town council", "over-extracted", "improving the flow", "darent",
                    "sevenoaks", "allocated", "stakeholders", "sewage treatment",
                ),
                "Environment",
                "#environment",
                2,
            ),
            (
                (
                    "minidisc", "mini disc", "betamax", "walkman", "format war",
                    "magneto-optical", "netmd", "himd", "digital compact cassette",
                    "a-trac", "mixtape", "audio format", "compact cassette",
                ),
                "Technology",
                "#technology",
                2,
            ),
        ]

        for terms, category, tag, min_hits in subject_rules:
            hits = sum(1 for term in terms if term in text)
            if hits < min_hits:
                continue
            if category not in normalized_categories:
                normalized_categories.insert(0, category)
            tag_norm = self._normalize_tag(tag)
            if tag_norm and tag_norm not in normalized_tags:
                normalized_tags.insert(0, tag_norm)
            normalized_categories = [
                c for c in normalized_categories
                if c.lower() not in self._FORMAT_CATEGORIES or c.lower() == category.lower()
            ]
            normalized_tags = [
                t for t in normalized_tags
                if t.lower() not in self._FORMAT_TAGS or t.lower() == tag_norm
            ]
            break

        return normalized_tags, normalized_categories

    def _boost_environment_community_labels(
        self, text: str, tags: list[str], categories: list[str]
    ) -> tuple[list[str], list[str]]:
        if not self._environment_community_context(text):
            return tags, categories
        if self._assistive_tech_narrative_context(text):
            return tags, categories
        normalized_tags = self._normalize_tags(tags)
        normalized_categories = [c.strip() for c in categories if c and c.strip()]
        for tag in ("#environment", "#community", "#water", "#charity"):
            t = self._normalize_tag(tag)
            if t and t not in normalized_tags:
                normalized_tags.insert(0, t)
        for cat in ("Environment", "Community", "Charity", "News"):
            if cat not in normalized_categories:
                normalized_categories.insert(0, cat)
        normalized_tags, normalized_categories = self._strip_unrelated_assistive_labels(
            text, normalized_tags, normalized_categories
        )
        return normalized_tags, normalized_categories

    def _wildlife_media_context(self, text: str) -> bool:
        return wildlife_media_narrative(text)

    def _tech_history_context(self, text: str) -> bool:
        return tech_history_narrative(text)

    def _strip_misleading_tags_for_context(
        self, text: str, tags: list[str], categories: list[str]
    ) -> tuple[list[str], list[str]]:
        if self._assistive_tech_narrative_context(text):
            return tags, categories
        tags = self._normalize_tags(tags)
        categories = [c.strip() for c in categories if c and c.strip()]
        if self._tech_history_context(text):
            tags = [t for t in tags if t.lower() not in self._MISLEADING_TAGS_TECH_HISTORY]
            categories = [
                c
                for c in categories
                if c.strip().lower()
                not in self._ASSISTIVE_FLAT_CATEGORIES
                | {"wildlife", "photography", "animals", "nature"}
            ]
        elif not self._wildlife_media_context(text):
            tags = [
                t
                for t in tags
                if t.lower()
                not in {"#wildlife", "#photography", "#animals", "#nature", "#awards"}
            ]
            categories = [
                c
                for c in categories
                if c.strip().lower() not in {"wildlife", "photography", "animals", "nature"}
            ]
        return tags, categories

    def _vet_clinical_context(self, text: str) -> bool:
        return any(term in text for term in self._VET_CLINICAL_TERMS)

    def _strip_clinical_categories_for_wildlife_media(
        self, text: str, categories: list[str]
    ) -> list[str]:
        if not self._wildlife_media_context(text) or self._vet_clinical_context(text):
            return categories
        return [
            c
            for c in categories
            if c.strip().lower() not in self._CLINICAL_ANIMAL_CATEGORIES
        ]

    def _assistive_tech_narrative_context(self, text: str) -> bool:
        return assistive_tech_narrative(text)

    def _refine_assistive_tech_labels(
        self, text: str, tags: list[str], categories: list[str]
    ) -> tuple[list[str], list[str]]:
        if not self._assistive_tech_narrative_context(text):
            return tags, categories
        normalized_tags = self._normalize_tags(tags)
        normalized_categories = [c.strip() for c in categories if c and c.strip()]
        normalized_tags = [
            t for t in normalized_tags if t.lower() not in self._MISLEADING_TAGS_ASSISTIVE
        ]
        normalized_categories = [
            c
            for c in normalized_categories
            if c.strip().lower() not in self._MISLEADING_CATS_ASSISTIVE
        ]
        for tag in (
            "#accessibility",
            "#guidedogs",
            "#assistivetechnology",
            "#smartglasses",
            "#humanconnection",
            "#disability",
        ):
            t = self._normalize_tag(tag)
            if t and t not in normalized_tags:
                normalized_tags.insert(0, t)
        for cat in (
            "Personal lived experience",
            "Accessibility",
            "Human connection",
            "Lifestyle",
        ):
            if cat not in normalized_categories:
                normalized_categories.insert(0, cat)
        return normalized_tags, normalized_categories

    def _boost_wildlife_media_labels(
        self, text: str, tags: list[str], categories: list[str]
    ) -> tuple[list[str], list[str]]:
        if self._assistive_tech_narrative_context(text):
            return tags, categories
        if not self._wildlife_media_context(text) or self._vet_clinical_context(text):
            return tags, categories
        normalized_tags = self._normalize_tags(tags)
        normalized_categories = [c.strip() for c in categories if c and c.strip()]
        for tag in ("#wildlife", "#photography", "#awards"):
            t = self._normalize_tag(tag)
            if t and t not in normalized_tags:
                normalized_tags.insert(0, t)
        for cat in ("Wildlife", "News", "Photography"):
            if cat not in normalized_categories:
                normalized_categories.insert(0, cat)
        return normalized_tags, normalized_categories

    def _apply_editorial_rules(
        self,
        transcript: str,
        tags: list[str],
        categories: list[str],
        max_tags: int,
    ) -> tuple[list[str], list[str]]:
        text = (transcript or "").lower()
        tags, categories = self._sanitize_categorization_labels(transcript, tags, categories)
        normalized_tags = self._normalize_tags(tags)
        normalized_categories = [c.strip() for c in (categories or []) if c and c.strip()]
        normalized_tags, normalized_categories = self._strip_unrelated_assistive_labels(
            text, normalized_tags, normalized_categories
        )
        normalized_tags, normalized_categories = self._rebalance_subject_over_format(
            transcript, normalized_tags, normalized_categories
        )
        normalized_categories = self._strip_clinical_categories_for_wildlife_media(
            text, normalized_categories
        )
        normalized_tags, normalized_categories = self._refine_assistive_tech_labels(
            text, normalized_tags, normalized_categories
        )
        normalized_tags, normalized_categories = self._boost_wildlife_media_labels(
            text, normalized_tags, normalized_categories
        )
        normalized_tags, normalized_categories = self._boost_environment_community_labels(
            text, normalized_tags, normalized_categories
        )
        sports_terms = (
            "football", "soccer", "rugby", "tennis", "cricket", "match", "league",
            "goal", "championship", "tournament", "athlete", "golf",
        )
        obituary_terms = (
            "obituary", "obituaries", "died", "died peacefully", "funeral",
            "celebration of her life", "celebration of his life", "flowers welcome",
            "donations may be made", "in memory", "passed away", "aged ",
        )
        religion_terms = ("church", "faith", "worship", "catholic", "prayer")
        obituary_hits = sum(1 for term in obituary_terms if term in text)
        sports_hits = sum(1 for term in sports_terms if term in text)
        if obituary_hits >= 2:
            boosts = ["#obituary", "#localnews", "#community"]
            if any(term in text for term in religion_terms):
                boosts.append("#religion")
            for tag in boosts:
                t = self._normalize_tag(tag)
                if t and t not in normalized_tags:
                    normalized_tags.insert(0, t)
            if "Obituaries" not in normalized_categories:
                normalized_categories.insert(0, "Obituaries")
            if "News" not in normalized_categories:
                normalized_categories.append("News")
            if "Community" not in normalized_categories:
                normalized_categories.append("Community")
            if sports_hits < 3:
                normalized_tags = [t for t in normalized_tags if t != "#sports"]
        normalized_tags, normalized_categories = self._strip_unrelated_assistive_labels(
            text, normalized_tags, normalized_categories
        )
        normalized_tags, normalized_categories = self._strip_misleading_tags_for_context(
            text, normalized_tags, normalized_categories
        )
        return self._normalize_tags(normalized_tags)[:max_tags], normalized_categories[:3]

    # ------------------------------------------------------------------

    def _extract_transcript_words(self, transcript: str) -> set[str]:
        words = set()
        for w in re.split(r"[\s\.,;:!?\-\"\'()]+", transcript):
            w = w.lower().strip()
            if len(w) > 3 and w not in _STOPWORDS:
                words.add(w)
        return words

    def _rank_categories(self, categories: list[str], keyword_scores: dict[str, float]) -> list[str]:
        hit: list[str] = []
        rest: list[str] = []
        for cat in categories:
            if f"#{cat}" in keyword_scores or cat in keyword_scores:
                hit.append(cat)
            else:
                rest.append(cat)
        return hit + rest

    def _top_nli_categories(self, zero_shot_scores: dict, *, limit: int = 6) -> list[str]:
        ranked = sorted(zero_shot_scores.items(), key=lambda x: x[1], reverse=True)
        out: list[str] = []
        for cat, score in ranked:
            if score < 0.2:
                break
            if cat.lower() in self._FORMAT_CATEGORIES and score < 0.5:
                continue
            out.append(cat)
            if len(out) >= limit:
                break
        return out

    def _build_context_category_shortlist(
        self,
        transcript: str,
        all_categories: list[str],
        keyword_scores: dict,
        zero_shot_scores: dict,
        *,
        limit: int = 50,
    ) -> list[str]:
        tx_words = self._extract_transcript_words(transcript)
        canonical = {c.lower(): c for c in all_categories}
        scores: dict[str, float] = defaultdict(float)

        for cat, zs in zero_shot_scores.items():
            key = cat.lower()
            if key in canonical:
                scores[canonical[key]] += float(zs) * 0.55

        for tag, kw in keyword_scores.items():
            tag_clean = tag.lstrip("#").lower()
            for key, name in canonical.items():
                if key == tag_clean or key in tag_clean or tag_clean in key:
                    scores[name] += float(kw) * 0.35

        for cat in all_categories:
            cat_words = set(re.findall(r"[a-z]+", cat.lower()))
            overlap = len(cat_words & tx_words)
            if overlap:
                scores[cat] += overlap * 0.12

        for fmt in self._FORMAT_CATEGORIES:
            for cat in list(scores):
                if cat.lower() == fmt and scores[cat] < 0.45:
                    scores[cat] *= 0.3

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ordered = [c for c, s in ranked if s >= 0.08]
        subject_scored = [c for c in ordered if c.lower() not in self._FORMAT_CATEGORIES]
        format_scored = [c for c in ordered if c.lower() in self._FORMAT_CATEGORIES]
        seen: set[str] = set()
        shortlist: list[str] = []

        def _append(candidates: list[str]) -> None:
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    shortlist.append(c)

        _append(subject_scored)
        _append([c for c in all_categories if c.lower() not in self._FORMAT_CATEGORIES])
        _append(format_scored)
        _append(all_categories)
        return shortlist[:limit]

    def _finalize_categories(
        self,
        transcript: str,
        categories: list[str],
        zero_shot_scores: dict,
        *,
        max_categories: int = 3,
    ) -> list[str]:
        cats = [c.strip() for c in (categories or []) if isinstance(c, str) and c.strip()]
        if not cats and zero_shot_scores:
            ranked = sorted(zero_shot_scores.items(), key=lambda x: x[1], reverse=True)
            cats = [c for c, s in ranked if s >= 0.28][:max_categories]

        text = (transcript or "").lower()
        ranked = sorted(zero_shot_scores.items(), key=lambda x: x[1], reverse=True)
        for cat, score in ranked[:8]:
            if score < 0.32:
                continue
            if cat.lower() in self._FORMAT_CATEGORIES:
                continue
            if (
                self._wildlife_media_context(text)
                and not self._vet_clinical_context(text)
                and cat.strip().lower() in self._CLINICAL_ANIMAL_CATEGORIES
            ):
                continue
            if cat not in cats:
                cats.insert(0, cat)

        _, cats = self._rebalance_subject_over_format(transcript, [], cats)
        cats = self._strip_clinical_categories_for_wildlife_media(text, cats)
        _, cats = self._boost_wildlife_media_labels(text, [], cats)
        return cats[:max_categories]

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
            # Wrap each pipe-separated alternative with word boundaries so that
            # short tokens like 'AI', 'car', 'ant', 'bee' don't match as substrings
            # inside unrelated words (e.g. 'AI' inside 'again', 'ant' inside 'advantage').
            bounded = "|".join(f"\\b{p.strip()}\\b" for p in pattern.lower().split("|"))
            matches = len(re.findall(bounded, text_lower))
            if matches > 0:
                scores[tag] = min(1.0, matches * 0.15 + 0.4)
        if segments:
            seg_counter = Counter()
            for seg in segments:
                seg_text = seg.get("text", "").lower()
                for pattern, tag in keyword_rules.items():
                    bounded = "|".join(f"\\b{p.strip()}\\b" for p in pattern.lower().split("|"))
                    if re.search(bounded, seg_text):
                        seg_counter[tag] += 1
            for tag, count in seg_counter.items():
                density = count / max(len(segments), 1)
                scores[tag] = round(scores.get(tag, 0) * 0.6 + density * 0.4, 4)
        return {"scores": scores}

    _ZS_TEMPLATE = "This audio recording is about {}."

    def _zero_shot_labels(self, transcript: str, labels: list[str]) -> dict:
        if not labels:
            return {"scores": {}}
        output = get_triton_client().nli_sync(transcript[:1024], labels, hypothesis_template=self._ZS_TEMPLATE)
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
            result = get_triton_client().sentiment_sync(transcript[:512])
        except Exception:
            return "neutral"
        if isinstance(result, list) and result:
            result = result[0]
        if not isinstance(result, dict):
            return "neutral"
        label = result.get("label", "").lower()
        if "positive" in label:
            return "positive"
        if "negative" in label:
            return "negative"
        return "neutral"

    _TAG_THRESHOLD = 0.50
    _CAT_THRESHOLD = 0.35

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
            elif s2 > 0:
                score = (s1 * 0.4) + (s2 * 0.6)
            else:
                score = s1 * 1.0

            merged_tag_scores[tag] = round(min(1.0, score), 4)

        ranked_tags = sorted(merged_tag_scores.items(), key=lambda x: x[1], reverse=True)

        tags = [t for t, s in ranked_tags if s >= self._TAG_THRESHOLD][:max_tags]

        cat_scores: dict[str, float] = {}
        for c in all_categories:
            s1 = l1.get(f"#{c}", 0)
            s2 = l2c.get(c, 0)
            s3 = l3.get(c, 0)

            if has_openai:
                score = (s1 * 0.2) + (s2 * 0.3) + (s3 * 0.7)
                if (s1 > 0 or s2 > 0) and s3 > 0:
                    score += 0.10
            else:
                score = (s1 * 0.4) + (s2 * 0.6)

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
