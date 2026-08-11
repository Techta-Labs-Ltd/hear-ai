import asyncio
import logging
import re
import warnings
from collections import Counter, defaultdict

from transformers import logging as hf_logging

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning, message=".*clean_up_tokenization_spaces.*")
from hear.core.category_loader import (
    _taxonomy_path_to_tag,
    category_loader,
    is_hierarchical_taxonomy_path,
)
from hear.core.content_context import (
    assistive_tech_narrative,
    filter_freeform_tag_labels,
    wildlife_media_narrative,
)
from hear.core.discovery_taxonomy import discovery_taxonomy_loader
from hear.core.platform_settings import fetch_platform_settings
from hear.models.database import CategoryTrainingExample, SessionLocal
from hear.services.llm import get_llm_service
from hear.services.model_client import get_model_client
from hear.training import categorizer_infer

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "the", "and", "for", "with", "has", "was", "are", "his", "her", "they",
    "that", "this", "from", "have", "been", "said", "also", "only", "when",
    "into", "after", "their", "there", "were", "what", "which", "about",
    "will", "would", "could", "should", "over", "some", "all", "more",
    "than", "then", "just", "each", "even", "him", "had", "not", "but",
    "out", "who", "two", "time", "very", "our", "here", "where", "both",
    "other", "those", "these", "its", "year", "years",
}

_FORMAT_CATEGORIES = frozenset(
    {"podcast", "documentary", "entertainment", "lifestyle", "opinion", "media"}
)


class CategorizationService:

    _FORMAT_TAGS = frozenset({"#podcast", "#radio", "#broadcast", "#streaming"})

    async def categorize(
        self,
        transcript: str,
        segments: list[dict] | None = None,
        custom_tags: list[str] | None = None,
        max_tags: int = 8,
        per_track_transcripts: dict[str, str] | None = None,
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

        # blocked_keywords/auto_tag_keywords are pushed once via
        # Pipeline.UpdatePlatformSettings over gRPC and persisted
        # there -- no network call and no re-persisting needed on every categorize().
        platform = await fetch_platform_settings()
        settings_applied = bool(platform.auto_tag_keywords or platform.blocked_keywords)


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

        sentiment = await loop.run_in_executor(None, self._get_sentiment, transcript)
        tag_labels = list(tag_pool) if tag_pool else []
        layer2_tag = (
            await loop.run_in_executor(None, self._zero_shot_labels, transcript, tag_labels)
            if tag_labels
            else {"scores": {}}
        )

        merged = self._merge(layer1, layer2_cat, layer2_tag, data.tags, data.categories, max_tags)
        merged["tags"] = self._normalize_tags(merged["tags"])[:max_tags]

        new_tags_added = []
        for tag in merged["tags"]:
            if tag not in self._normalize_tags(data.tags):
                category_loader.add_tag(tag)
                new_tags_added.append(tag)

        new_categories_added: list[str] = []

        merged["tags"], merged["categories"], _ = await loop.run_in_executor(
            None, self._apply_blocked_keywords, transcript, merged["tags"], merged["categories"], platform.blocked_keywords,
        )
        merged["tags"] = self._ensure_non_empty_tags(merged["tags"], merged["categories"], transcript, max_tags)
        merged["tags"], merged["categories"] = self._sanitize_categorization_labels(
            merged["tags"], merged["categories"]
        )
        merged["categories"] = self._finalize_categories(
            transcript, merged["categories"], layer2_cat.get("scores", {}), max_categories=3,
        )
        merged["tags"], merged["categories"] = self._apply_editorial_rules(
            transcript, merged["tags"], merged["categories"], max_tags
        )
        merged["tags"] = self._normalize_tags(merged["tags"])[:max_tags]
        persisted_tags, persisted_cats = category_loader.ensure_labels(
            merged["tags"], merged["categories"]
        )
        for t in persisted_tags:
            if t not in new_tags_added:
                new_tags_added.append(t)
        for c in persisted_cats:
            if c not in new_categories_added:
                new_categories_added.append(c)

        self._log_auto_example(transcript, merged["categories"], merged["tags"])
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

    def _log_auto_example(self, transcript: str, categories: list[str], tags: list[str]) -> None:
        """Every real categorize() call becomes a weakly-labeled training example,
        distinct from higher-trust backend-verified webhook examples (source=.grpc.)."""
        if not transcript or not transcript.strip():
            return
        db = SessionLocal()
        try:
            for cat in (categories or [None]):
                db.add(CategoryTrainingExample(
                    source="auto_categorized",
                    event_type="categorized",
                    text=transcript[:4000],
                    category=cat,
                    tags=tags or [],
                    label=None,
                    raw_payload=None,
                ))
            db.commit()
        except Exception:
            logger.warning("[CATEGORIZER] failed to log auto training example", exc_info=True)
            db.rollback()
        finally:
            db.close()

    def _log_keyword_examples(self, keywords: list[str], label: str) -> None:
        if not keywords:
            return
        db = SessionLocal()
        try:
            for kw in keywords:
                db.add(CategoryTrainingExample(
                    source="auto_categorized",
                    event_type=f"keyword_{label}",
                    text=kw,
                    category=None,
                    tags=[kw],
                    label=label,
                    raw_payload=None,
                ))
            db.commit()
        except Exception:
            logger.warning("[CATEGORIZER] failed to log keyword training example", exc_info=True)
            db.rollback()
        finally:
            db.close()

    def _trained_model_fallback(self, target: str, transcript: str, limit: int) -> list[str]:
        """Predictions from the Ray Train checkpoint (app/training/categorizer_train.py).
        Used two ways: as a hint fed into the Qwen prompt on every call, and as a
        last-resort answer when the keyword/NLI/Qwen pipeline found nothing. A no-op
        returning [] until a checkpoint has been trained via POST /api/v1/admin/train-categorizer."""
        if not transcript:
            return []
        scores = categorizer_infer.predict(target, transcript)
        if not scores:
            return []
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picked = [label for label, score in ranked if score >= 0.3][:limit]
        if picked:
            logger.info("[CATEGORIZER] trained_model_fallback target=%s picked=%s", target, picked)
        return picked

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
            nli_merged = self._merge(layer1, layer2_cat, layer2_tag, data.tags, data.categories, max_tags)
            t_sent = await loop.run_in_executor(None, self._get_sentiment, t_text)

            t_tags = self._normalize_tags(nli_merged.get("tags", []))
            t_cats = self._finalize_categories(
                t_text,
                nli_merged.get("categories", []),
                zs_scores,
                max_categories=3,
            )
            t_tags, t_cats = self._sanitize_categorization_labels(t_tags, t_cats)
            t_tags, t_cats = self._apply_editorial_rules(
                t_text, t_tags, t_cats, max_tags
            )
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
            full_text = " ".join(track_texts.values())
            all_tags, all_categories, _ = await loop.run_in_executor(
                None, self._apply_blocked_keywords, full_text, all_tags, all_categories, platform.blocked_keywords,
            )
            for tid, t_text in track_texts.items():
                if tid not in per_track:
                    continue
                kept_tags, kept_cats, _ = await loop.run_in_executor(
                    None, self._apply_blocked_keywords, t_text, per_track[tid]["tags"], per_track[tid]["categories"], platform.blocked_keywords,
                )
                per_track[tid]["tags"], per_track[tid]["categories"] = kept_tags, kept_cats
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
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        tags = self._normalize_tags(
            list(llm_result.get("tags", [])) + list(llm_result.get("new_tags", []))
        )
        categories: list[str] = []
        seen: set[str] = set()
        for raw in list(llm_result.get("categories", [])) + list(
            llm_result.get("new_categories", [])
        ):
            if not isinstance(raw, str):
                continue
            category = re.sub(r"\s+", " ", raw.strip())
            key = category.lower()
            if not category or key in seen:
                continue
            seen.add(key)
            categories.append(category)
        tags, categories = self._sanitize_categorization_labels(tags, categories)
        return tags, categories, [], []

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
        trained_model_hints: dict[str, list[str]] = {}
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
                trained_model_hints=trained_model_hints,
            ),
        )
        tags, categories, new_tags_added, new_categories_added = (
            self._integrate_llm_categorization(llm_result)
        )
        tags, categories = self._sanitize_categorization_labels(tags, categories)
        tags, categories = self._apply_editorial_rules(
            transcript, tags, categories, max_tags
        )
        tags, categories = self._fill_gaps_from_scores(
            tags, categories, zero_shot_scores, layer1_scores,
            max_tags=max_tags, max_categories=max_categories,
        )
        if not categories:
            categories = self._trained_model_fallback("category", transcript, max_categories)
        if not tags:
            tags = self._trained_model_fallback("tags", transcript, max_tags)
        tags, categories = self._apply_editorial_rules(
            transcript, tags, categories, max_tags
        )

        if platform.blocked_keywords:
            tags, categories, _ = await loop.run_in_executor(
                None, self._apply_blocked_keywords, transcript, tags, categories, platform.blocked_keywords,
            )
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
        self._log_auto_example(transcript, categories, tags)
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

    # ------------------------------------------------------------------
    # Label formatting / gap-filling (context-score-driven, no hardcoded categories)
    # ------------------------------------------------------------------

    def _sanitize_categorization_labels(
        self,
        tags: list[str],
        categories: list[str],
    ) -> tuple[list[str], list[str]]:
        """Tags must be #hashtags; taxonomy paths belong in discovery, not categorization output."""
        out_tags: list[str] = []
        seen_t: set[str] = set()
        out_cats: list[str] = []
        seen_c: set[str] = set()

        for raw in categories or []:
            cat = re.sub(r"\s+", " ", str(raw or "").strip())
            if not cat:
                continue
            if is_hierarchical_taxonomy_path(cat):
                cat = cat.split(" > ")[-1].strip()
                if not cat:
                    continue
            low = cat.lower()
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
                slug = _taxonomy_path_to_tag(str(raw))
                if slug and slug.lower() not in seen_t:
                    seen_t.add(slug.lower())
                    out_tags.append(slug)
                continue
            norm = self._normalize_tag(str(raw))
            if not norm or norm.lower() in seen_t:
                continue
            seen_t.add(norm.lower())
            out_tags.append(norm)

        return out_tags, out_cats

    def _fill_gaps_from_scores(
        self,
        tags: list[str],
        categories: list[str],
        zero_shot_scores: dict,
        keyword_scores: dict,
        *,
        max_tags: int,
        max_categories: int = 3,
    ) -> tuple[list[str], list[str]]:
        """Light guardrail after Qwen: fill gaps from the NLI/keyword scores computed
        for THIS transcript when Qwen returned nothing -- never overrides good picks,
        and never injects a category that wasn't scored against the actual content."""
        normalized_tags = self._normalize_tags(tags)
        normalized_categories = [c.strip() for c in (categories or []) if c and c.strip()]

        if not normalized_categories and zero_shot_scores:
            ranked = sorted(zero_shot_scores.items(), key=lambda x: x[1], reverse=True)
            for cat, score in ranked:
                if score < 0.45:
                    break
                if cat.lower() in _FORMAT_CATEGORIES:
                    continue
                normalized_categories.append(cat)
                if len(normalized_categories) >= max_categories:
                    break

        if not normalized_tags and keyword_scores:
            for tag, score in sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True):
                if score < 0.35:
                    break
                t = self._normalize_tag(tag)
                if t and t not in normalized_tags:
                    normalized_tags.append(t)
                if len(normalized_tags) >= max_tags:
                    break

        return normalized_tags[:max_tags], normalized_categories[:max_categories]

    def _normalize_tag(self, tag: str) -> str:
        if not tag:
            return ""
        clean = str(tag).strip().lower()
        clean = re.sub(r"\s+", "-", clean)
        clean = clean.lstrip("#")
        clean = re.sub(r"[^a-z0-9_\-]", "", clean)
        if not clean:
            return ""
        if len(clean) > 18 and "-" not in clean and "_" not in clean:
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

    def _rebalance_subject_over_format(
        self,
        transcript: str,
        tags: list[str],
        categories: list[str],
    ) -> tuple[list[str], list[str]]:
        """Keep model-selected subject labels ahead of generic recording formats."""
        del transcript
        normalized_tags = self._normalize_tags(tags)
        normalized_categories = [
            c.strip() for c in categories or [] if c and c.strip()
        ]
        has_subject_category = any(
            category.lower() not in _FORMAT_CATEGORIES
            for category in normalized_categories
        )
        has_subject_tag = any(tag not in self._FORMAT_TAGS for tag in normalized_tags)
        if has_subject_category:
            normalized_categories = [
                category
                for category in normalized_categories
                if category.lower() not in _FORMAT_CATEGORIES
            ]
        if has_subject_tag:
            normalized_tags = [tag for tag in normalized_tags if tag not in self._FORMAT_TAGS]
        return normalized_tags, normalized_categories

    def _apply_editorial_rules(
        self,
        transcript: str,
        tags: list[str],
        categories: list[str],
        max_tags: int,
    ) -> tuple[list[str], list[str]]:
        """Apply shared context filters to open-ended model labels."""
        text = (transcript or "").lower()
        clean_tags = self._normalize_tags(
            filter_freeform_tag_labels(text, [str(tag).lstrip("#") for tag in tags or []])
        )
        clean_categories = [c.strip() for c in categories or [] if c and c.strip()]

        if not assistive_tech_narrative(text):
            clean_categories = [
                c for c in clean_categories
                if c.lower()
                not in {
                    "accessibility",
                    "assistive technology",
                    "visual impairment",
                    "personal lived experience",
                }
            ]
        if not wildlife_media_narrative(text):
            clean_categories = [
                c for c in clean_categories
                if c.lower() not in {"wildlife", "photography", "animals", "nature"}
            ]

        clean_tags, clean_categories = self._rebalance_subject_over_format(
            transcript, clean_tags, clean_categories
        )
        return self._normalize_tags(clean_tags)[:max_tags], clean_categories[:3]

    def _apply_blocked_keywords(
        self, transcript: str, tags: list[str], categories: list[str], blocked_keywords: list[str],
    ) -> tuple[list[str], list[str], bool]:
        """Context-aware moderation: scores each blocked keyword against the actual
        transcript via zero-shot NLI (does this content genuinely discuss that topic?)
        instead of a blind substring match on tag/category text. Only keywords whose
        topic is actually present in the content get flagged and stripped."""
        blocked = [b.strip() for b in (blocked_keywords or []) if b and b.strip()]
        if not blocked or not transcript or not transcript.strip():
            return tags, categories, False
        scores = self._zero_shot_labels(transcript, blocked).get("scores", {})
        flagged = [kw for kw, score in scores.items() if score >= 0.5]
        if not flagged:
            return tags, categories, False
        self._log_keyword_examples(flagged, label="blocked")
        flagged_lower = [f.lower() for f in flagged]
        kept_tags = [t for t in tags if not any(bk in t.lower() for bk in flagged_lower)]
        kept_cats = [c for c in categories if not any(bk in c.lower() for bk in flagged_lower)]
        return kept_tags, kept_cats, True

    def _ensure_non_empty_tags(self, tags: list[str], categories: list[str], transcript: str, max_tags: int) -> list[str]:
        normalised = self._normalize_tags(tags)
        if normalised:
            return normalised[:max_tags]
        category_tags = [self._normalize_tag(c) for c in categories if c]
        category_tags = [t for t in category_tags if t]
        if category_tags:
            return self._normalize_tags(category_tags)[:max_tags]
        trained = self._trained_model_fallback("tags", transcript, max_tags)
        if trained:
            return self._normalize_tags(trained)[:max_tags]
        words = [w for w in re.findall(r"[a-zA-Z]{4,}", transcript.lower()) if w not in _STOPWORDS]
        if words:
            return self._normalize_tags([words[0]])[:max_tags]
        return []

    # ------------------------------------------------------------------

    def _extract_transcript_words(self, transcript: str) -> set[str]:
        words = set()
        for w in re.split(r"[\s\.,;:!?\-\"\'()]+", transcript):
            w = w.lower().strip()
            if len(w) > 3 and w not in _STOPWORDS:
                words.add(w)
        return words

    def _top_nli_categories(self, zero_shot_scores: dict, *, limit: int = 6) -> list[str]:
        ranked = sorted(zero_shot_scores.items(), key=lambda x: x[1], reverse=True)
        out: list[str] = []
        for cat, score in ranked:
            if score < 0.2:
                break
            if cat.lower() in _FORMAT_CATEGORIES and score < 0.5:
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

        for fmt in _FORMAT_CATEGORIES:
            for cat in list(scores):
                if cat.lower() == fmt and scores[cat] < 0.45:
                    scores[cat] *= 0.3

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ordered = [c for c, s in ranked if s >= 0.08]
        subject_scored = [c for c in ordered if c.lower() not in _FORMAT_CATEGORIES]
        format_scored = [c for c in ordered if c.lower() in _FORMAT_CATEGORIES]
        seen: set[str] = set()
        shortlist: list[str] = []

        def _append(candidates: list[str]) -> None:
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    shortlist.append(c)

        _append(subject_scored)
        _append([c for c in all_categories if c.lower() not in _FORMAT_CATEGORIES])
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

        ranked = sorted(zero_shot_scores.items(), key=lambda x: x[1], reverse=True)
        for cat, score in ranked[:8]:
            if score < 0.32:
                continue
            if cat.lower() in _FORMAT_CATEGORIES:
                continue
            if cat not in cats:
                cats.insert(0, cat)

        if not cats:
            cats = self._trained_model_fallback("category", transcript, max_categories)

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
        output = get_model_client().nli_sync(transcript[:1024], labels, hypothesis_template=self._ZS_TEMPLATE)
        return {"scores": dict(zip(output["labels"], output["scores"]))}

    def _get_sentiment(self, transcript: str) -> str:
        try:
            result = get_model_client().sentiment_sync(transcript[:512])
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
        all_tags: list[str],
        all_categories: list[str],
        max_tags: int,
    ) -> dict:
        l1  = layer1.get("scores", {})
        l2c = layer2_cat.get("scores", {})
        l2t = layer2_tag.get("scores", {})

        merged_tag_scores: dict[str, float] = {}
        for tag in all_tags:
            s1 = l1.get(tag, 0)
            s2 = l2t.get(tag, 0)
            score = (s1 * 0.4) + (s2 * 0.6) if s2 > 0 else s1 * 1.0
            merged_tag_scores[tag] = round(min(1.0, score), 4)

        ranked_tags = sorted(merged_tag_scores.items(), key=lambda x: x[1], reverse=True)

        tags = [t for t, s in ranked_tags if s >= self._TAG_THRESHOLD][:max_tags]

        cat_scores: dict[str, float] = {}
        for c in all_categories:
            s1 = l1.get(f"#{c}", 0)
            s2 = l2c.get(c, 0)
            score = (s1 * 0.4) + (s2 * 0.6)
            cat_scores[c] = round(min(1.0, score), 4)

        ranked_cats = sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)
        categories = [c for c, s in ranked_cats if s >= self._CAT_THRESHOLD][:3]
        if not categories and ranked_cats:
            categories = [c for c, _ in ranked_cats[:1]]

        logger.debug("[CATEGORIZER] top_tag_scores=%s", ranked_tags[:8])
        logger.debug("[CATEGORIZER] top_cat_scores=%s", sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)[:5])

        return {
            "tags": tags,
            "categories": categories,
            "confidence_scores": {**merged_tag_scores, **cat_scores},
        }
