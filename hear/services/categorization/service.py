import asyncio
import logging
import re
from collections import Counter, defaultdict
from collections.abc import Mapping

from hear.inference.client import LocalInferenceClient
from hear.services.llm import LLMService
from hear.services.pipeline.configuration import (
    CategoryLabels,
    PipelineCategoryCatalog,
    PipelineConfiguration,
    PipelineTaxonomy,
)
from hear.utils.content_context import (
    assistive_tech_narrative,
    filter_controlled_taxonomy_paths,
    filter_freeform_tag_labels,
    wildlife_media_narrative,
)

logger = logging.getLogger(__name__)
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "has",
    "was",
    "are",
    "his",
    "her",
    "they",
    "that",
    "this",
    "from",
    "have",
    "been",
    "said",
    "also",
    "only",
    "when",
    "into",
    "after",
    "their",
    "there",
    "were",
    "what",
    "which",
    "about",
    "will",
    "would",
    "could",
    "should",
    "over",
    "some",
    "all",
    "more",
    "than",
    "then",
    "just",
    "each",
    "even",
    "him",
    "had",
    "not",
    "but",
    "out",
    "who",
    "two",
    "time",
    "very",
    "our",
    "here",
    "where",
    "both",
    "other",
    "those",
    "these",
    "its",
    "year",
    "years",
}
_FORMAT_CATEGORIES = frozenset(
    {"podcast", "documentary", "entertainment", "lifestyle", "opinion", "media"}
)

class CategorizationService:
    def __init__(
        self,
        model_client: LocalInferenceClient | None = None,
        llm: LLMService | None = None,
        categories=None,
        taxonomy=None,
    ) -> None:
        self._model_client = model_client
        self._llm = llm
        configuration = PipelineConfiguration.empty()
        self._categories = categories or PipelineCategoryCatalog(configuration)
        self._taxonomy = taxonomy or PipelineTaxonomy(configuration)

    def _models(self) -> LocalInferenceClient:
        if self._model_client is None:
            raise RuntimeError("small_models_not_configured")
        return self._model_client

    def _llm_service(self) -> LLMService:
        if self._llm is None:
            return LLMService(enabled=False)
        return self._llm

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
            }
        data = self._categories.data
        catalog_cats, catalog_tags = self._expanded_catalog_labels(data)
        loop = asyncio.get_event_loop()
        active_tracks = {k: v for k, v in (per_track_transcripts or {}).items() if v and v.strip()}
        if len(active_tracks) > 1:
            return await self._categorize_multi_track(
                track_texts=active_tracks,
                data=data,
                max_tags=max_tags,
            )
        layer1 = await loop.run_in_executor(
            None, self._keyword_layer, transcript, segments or [], data.keyword_rules
        )
        catalog_cats = self._category_pool(transcript, catalog_cats)
        tag_pool = self._build_tag_pool(
            transcript, catalog_tags + list(custom_tags or []), layer1["scores"]
        )
        layer2_cat = await loop.run_in_executor(
            None, self._zero_shot_labels, transcript, catalog_cats
        )
        layer2_cat["scores"] = self._ground_category_scores(transcript, layer2_cat["scores"])
        context_cats = self._build_context_category_shortlist(
            transcript, catalog_cats, layer1["scores"], layer2_cat.get("scores", {})
        )
        nli_top = self._top_nli_categories(layer2_cat.get("scores", {}), limit=6)
        if self._llm_service().is_available:
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
        merged = self._merge(
            layer1,
            layer2_cat,
            layer2_tag,
            tag_labels,
            catalog_cats,
            max_tags,
        )
        merged["tags"] = self._normalize_tags(merged["tags"])[:max_tags]
        merged["tags"] = self._ensure_non_empty_tags(
            merged["tags"], merged["categories"], transcript, max_tags
        )
        merged["tags"], merged["categories"] = self._sanitize_categorization_labels(
            merged["tags"], merged["categories"]
        )
        merged["categories"] = self._finalize_categories(
            transcript, merged["categories"], layer2_cat.get("scores", {}), max_categories=3
        )
        merged["tags"] = self._subject_tags(
            transcript, merged["tags"], merged["categories"], merged["confidence_scores"]
        )
        merged["tags"] = self._ensure_non_empty_tags(
            merged["tags"], merged["categories"], transcript, max_tags
        )
        merged["tags"], merged["categories"] = self._apply_editorial_rules(
            transcript, merged["tags"], merged["categories"], max_tags
        )
        merged["tags"] = self._prune_related_tags(
            self._normalize_tags(merged["tags"]), merged["confidence_scores"]
        )[:max_tags]
        return {
            "tags": merged["tags"],
            "categories": merged["categories"],
            "confidence_scores": {
                label: merged["confidence_scores"].get(label, 0.0)
                for label in merged["tags"] + merged["categories"]
                if label in merged["confidence_scores"]
            },
            "sentiment": sentiment,
            "llm_used": False,
            "categorizer_mode": "nli",
        }

    async def _categorize_multi_track(
        self, track_texts: dict[str, str], data, max_tags: int
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
        llm_was_used = False
        per_track: dict[str, dict] = {}
        for track_id, t_text in track_texts.items():
            if not t_text or not t_text.strip():
                continue
            logger.info(
                "[CATEGORIZER] Multi-track: analysing track %s (%d words)",
                track_id[:16],
                len(t_text.split()),
            )
            layer1 = await loop.run_in_executor(
                None, self._keyword_layer, t_text, [], data.keyword_rules
            )
            tag_pool = self._build_tag_pool(t_text, catalog_tags, layer1["scores"])
            layer2_cat = await loop.run_in_executor(
                None, self._zero_shot_labels, t_text, self._category_pool(t_text, catalog_cats)
            )
            zs_scores = layer2_cat.get("scores", {})
            zs_scores = self._ground_category_scores(t_text, zs_scores)
            layer2_cat["scores"] = zs_scores
            context_cats = self._build_context_category_shortlist(
                t_text, catalog_cats, layer1["scores"], zs_scores
            )
            nli_top = self._top_nli_categories(zs_scores, limit=6)
            if self._llm_service().is_available:
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
                        loop=loop,
                    )
                    if qwen_track is None:
                        raise RuntimeError("qwen_primary returned no result")
                    t_tags = qwen_track["tags"]
                    t_cats = qwen_track["categories"]
                    t_sent = qwen_track.get("sentiment", "neutral")
                    per_track[track_id] = {
                        "tags": t_tags,
                        "categories": t_cats,
                        "sentiment": t_sent,
                    }
                    for tag in t_tags:
                        if tag not in all_tags:
                            all_tags.append(tag)
                            confidence_scores[tag] = 0.85
                    for cat in t_cats:
                        if cat not in all_categories:
                            all_categories.append(cat)
                    all_sentiments.append(t_sent)
                    llm_was_used = True
                    continue
                except Exception as exc:
                    logger.warning(
                        "[CATEGORIZER] Qwen failed for track %s (%s) — using NLI",
                        track_id[:16],
                        exc,
                    )
            layer2_tag = (
                await loop.run_in_executor(None, self._zero_shot_labels, t_text, tag_pool)
                if tag_pool else {"scores": {}}
            )
            nli_merged = self._merge(
                layer1, layer2_cat, layer2_tag, tag_pool, list(zs_scores), max_tags
            )
            t_sent = await loop.run_in_executor(None, self._get_sentiment, t_text)
            t_tags = self._normalize_tags(nli_merged.get("tags", []))
            t_cats = self._finalize_categories(
                t_text, nli_merged.get("categories", []), zs_scores, max_categories=3
            )
            t_tags, t_cats = self._sanitize_categorization_labels(t_tags, t_cats)
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
        all_tags = self._ensure_non_empty_tags(
            all_tags, all_categories, " ".join(track_texts.values()), max_tags
        )
        final_sentiment = (
            Counter(all_sentiments).most_common(1)[0][0] if all_sentiments else "neutral"
        )
        logger.info(
            "[CATEGORIZER] Multi-track merge: tags=%s categories=%s per_track_count=%d",
            all_tags[:max_tags],
            all_categories,
            len(per_track),
        )
        return {
            "tags": all_tags[:max_tags],
            "categories": all_categories,
            "confidence_scores": confidence_scores,
            "sentiment": final_sentiment,
            "llm_used": llm_was_used,
            "categorizer_mode": "qwen_primary" if llm_was_used else "nli",
            "per_track": per_track,
        }

    def _integrate_llm_categorization(self, llm_result: dict) -> tuple[list[str], list[str]]:
        tags = self._normalize_tags(list(llm_result.get("tags", [])))
        categories: list[str] = []
        seen: set[str] = set()
        for raw in list(llm_result.get("categories", [])):
            if not isinstance(raw, str):
                continue
            category = re.sub("\\s+", " ", raw.strip())
            key = category.lower()
            if not category or key in seen:
                continue
            seen.add(key)
            categories.append(category)
        tags, categories = self._sanitize_categorization_labels(tags, categories)
        return (tags, categories)

    def _expanded_catalog_labels(self, data) -> tuple[list[str], list[str]]:
        cats: list[str] = []
        tags: list[str] = []
        seen_c: set[str] = set()
        seen_t: set[str] = set()
        for c in self._categories.flat_catalog_categories():
            key = c.strip().lower()
            words = re.findall("[a-z]+", key)
            if (
                c.strip() and not c.startswith("#") and words
                and key not in {"none", "neutral", "local"}
                and len(words) <= 3 and len(set(words)) == len(words)
                and key not in seen_c
            ):
                seen_c.add(key)
                cats.append(c.strip())
        for t in data.tags:
            nt = self._normalize_tag(t)
            if nt and nt.lower() not in seen_t:
                seen_t.add(nt.lower())
                tags.append(nt)
        return (cats, tags)

    def _category_pool(self, transcript: str, labels: list[str]) -> list[str]:
        """Keep broad subjects and relevant specific labels within a bounded model call."""
        if len(labels) <= 512:
            return labels
        words = self._extract_transcript_words(transcript)
        broad = [label for label in labels if len(re.findall("[a-z]+", label.lower())) == 1]
        specific = [label for label in labels if label not in broad]
        specific.sort(key=lambda label: (
            -len(set(re.findall("[a-z]+", label.lower())) & words),
            len(re.findall("[a-z]+", label.lower())), len(label), label.lower(),
        ))
        relevant = [label for label in specific if set(re.findall("[a-z]+", label.lower())) & words]
        return list(dict.fromkeys(broad[:384] + relevant[:128]))[:512]

    def _ground_category_scores(self, transcript: str, scores: dict[str, float]) -> dict[str, float]:
        words = self._subject_terms(transcript)
        grounded = {}
        for label, score in scores.items():
            supported = bool(self._subject_terms(label) & words)
            if supported or score >= 0.85:
                grounded[label] = score
        return grounded

    @staticmethod
    def _subject_terms(text: str) -> set[str]:
        terms = set()
        for word in re.findall("[a-z]+", text.lower()):
            if word in _STOPWORDS:
                continue
            if len(word) > 6 and word.endswith("al"):
                word = word[:-2]
            elif len(word) > 4 and word.endswith("s"):
                word = word[:-1]
            terms.add(word)
        return terms

    def _subject_tags(self, transcript: str, tags: list[str], categories: list[str], scores: dict) -> list[str]:
        if not categories:
            return tags
        subjects = self._subject_terms(" ".join(categories))
        names = self._subject_terms(" ".join(re.findall(r"\b[A-Z][a-z]+\b", transcript)))
        # Avoid turning ordinary verbs in the report into its primary subjects.
        # Keep subject labels, named entities and strongly supported narrower tags.
        return [tag for tag in tags if self._subject_terms(tag) & (subjects | names) or scores.get(tag, 0) >= 0.9]

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
        loop,
        max_categories: int = 3,
    ) -> dict | None:
        taxonomy_paths = list(self._taxonomy.data.paths)
        llm_result = await loop.run_in_executor(
            None,
            lambda: self._llm_service().categorize(
                transcript,
                context_cats[:50],
                tag_pool[:120],
                layer1_scores,
                max_categories=max_categories,
                nli_top_categories=nli_top,
                taxonomy_paths=taxonomy_paths,
            ),
        )
        tags, categories = self._integrate_llm_categorization(llm_result)
        tags, categories = self._sanitize_categorization_labels(tags, categories)
        tags, categories = self._apply_editorial_rules(transcript, tags, categories, max_tags)
        tags, categories = self._fill_gaps_from_scores(
            tags,
            categories,
            zero_shot_scores,
            layer1_scores,
            max_tags=max_tags,
            max_categories=max_categories,
        )
        tags, categories = self._apply_editorial_rules(transcript, tags, categories, max_tags)
        tags = self._ensure_non_empty_tags(tags, categories, transcript, max_tags)
        tags = self._normalize_tags(tags)[:max_tags]
        categories = categories[:max_categories]
        confidence_scores = {t: 0.9 for t in tags}
        return {
            "tags": tags,
            "categories": categories,
            "confidence_scores": confidence_scores,
            "sentiment": llm_result.get("sentiment", "neutral"),
            "llm_used": True,
            "categorizer_mode": "qwen_primary",
        }

    def _sanitize_categorization_labels(
        self,
        tags_or_transcript: list[str] | str,
        categories_or_tags: list[str],
        categories: list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Tags must be #hashtags; taxonomy paths belong in discovery, not categorization output."""
        if categories is None:
            transcript = ""
            tags = tags_or_transcript
            categories = categories_or_tags
        else:
            transcript = str(tags_or_transcript)
            tags = categories_or_tags
            categories = filter_controlled_taxonomy_paths(transcript, categories)
        out_tags: list[str] = []
        seen_t: set[str] = set()
        out_cats: list[str] = []
        seen_c: set[str] = set()
        for raw in categories or []:
            cat = re.sub("\\s+", " ", str(raw or "").strip())
            if not cat:
                continue
            if CategoryLabels.is_hierarchical_taxonomy_path(cat):
                cat = cat.split(" > ")[-1].strip()
                if not cat:
                    continue
            low = cat.lower()
            if low in seen_c:
                continue
            seen_c.add(low)
            out_cats.append(cat)
            slug = (
                CategoryLabels._taxonomy_path_to_tag(raw)
                if CategoryLabels.is_hierarchical_taxonomy_path(str(raw))
                else ""
            )
            if slug and slug.lower() not in seen_t:
                seen_t.add(slug.lower())
                out_tags.append(slug)
        for raw in tags or []:
            if CategoryLabels.is_hierarchical_taxonomy_path(str(raw)):
                slug = CategoryLabels._taxonomy_path_to_tag(str(raw))
                if slug and slug.lower() not in seen_t:
                    seen_t.add(slug.lower())
                    out_tags.append(slug)
                continue
            norm = self._normalize_tag(str(raw))
            if not norm or norm.lower() in seen_t:
                continue
            seen_t.add(norm.lower())
            out_tags.append(norm)
        return (out_tags, out_cats)

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
        normalized_categories = [c.strip() for c in categories or [] if c and c.strip()]
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
        return (normalized_tags[:max_tags], normalized_categories[:max_categories])

    def _normalize_tag(self, tag: str) -> str:
        if not tag:
            return ""
        clean = str(tag).strip().lower()
        clean = re.sub("\\s+", "-", clean)
        clean = clean.lstrip("#")
        clean = re.sub("[^a-z0-9_\\-]", "", clean)
        if not clean:
            return ""
        if len(clean) > 18 and "-" not in clean and ("_" not in clean):
            return ""
        return f"#{clean}"

    @staticmethod
    def _tag_stems(tag: str) -> set[str]:
        stems = set()
        for token in tag.lstrip("#").lower().replace("_", "-").split("-"):
            token = token.strip()
            if len(token) >= 4:
                stems.add(token[:5])
        return stems

    @classmethod
    def _prune_related_tags(cls, tags: list[str], scores: dict[str, float]) -> list[str]:
        """Drop tags that only restate a stronger tag (politics / political-role / political-leaders)."""
        ordered = sorted(tags, key=lambda tag: -float(scores.get(tag, 0.0)))
        kept: list[str] = []
        kept_stems: list[set[str]] = []
        for tag in ordered:
            stems = cls._tag_stems(tag)
            if stems and any(stems <= other or other <= stems for other in kept_stems):
                continue
            kept.append(tag)
            kept_stems.append(stems)
        return [tag for tag in tags if tag in kept]

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
        self, transcript: str, tags: list[str], categories: list[str]
    ) -> tuple[list[str], list[str]]:
        """Keep model-selected subject labels ahead of generic recording formats."""
        del transcript
        normalized_tags = self._normalize_tags(tags)
        normalized_categories = [c.strip() for c in categories or [] if c and c.strip()]
        has_subject_category = any(
            category.lower() not in _FORMAT_CATEGORIES for category in normalized_categories
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
        return (normalized_tags, normalized_categories)

    def _apply_editorial_rules(
        self, transcript: str, tags: list[str], categories: list[str], max_tags: int
    ) -> tuple[list[str], list[str]]:
        """Apply shared context filters to open-ended model labels."""
        text = (transcript or "").lower()
        clean_tags = self._normalize_tags(
            filter_freeform_tag_labels(text, [str(tag).lstrip("#") for tag in tags or []])
        )
        clean_categories = [c.strip() for c in categories or [] if c and c.strip()]
        if not assistive_tech_narrative(text):
            clean_categories = [
                c
                for c in clean_categories
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
                c
                for c in clean_categories
                if c.lower() not in {"wildlife", "photography", "animals", "nature"}
            ]
        else:
            clean_categories = [c for c in clean_categories if c.lower() != "veterinary"]
            if any(term in text for term in ("award", "won", "photography competition")):
                clean_categories = [c for c in clean_categories if c.lower() != "podcast"]
                if not any(c.lower() == "wildlife" for c in clean_categories):
                    clean_categories.insert(0, "Wildlife")
                if not any(c.lower() in {"news", "photography"} for c in clean_categories):
                    clean_categories.append("News")
                clean_tags = self._normalize_tags([*clean_tags, "#wildlife"])
        if assistive_tech_narrative(text):
            clean_categories = [c for c in clean_categories if c.lower() != "technology"]
            clean_tags = [tag for tag in clean_tags if tag != "#technology"]
            if any(term in text for term in ("i was", "my ", "i'm", "i am")) and not any(
                c.lower() == "personal lived experience" for c in clean_categories
            ):
                clean_categories.insert(0, "Personal lived experience")
            clean_tags = self._normalize_tags(
                [
                    *clean_tags,
                    "#accessibility",
                    "#assistive-technology",
                    *(["#guidedogs"] if "guide dog" in text else []),
                ]
            )
        elif any(term in text for term in ("river", "charity", "council has allocated")):
            if not any(
                c.lower() in {"environment", "community", "charity", "news"}
                for c in clean_categories
            ):
                clean_categories.insert(0, "Environment")
            if not any(
                tag in clean_tags for tag in ("#environment", "#community", "#charity", "#water")
            ):
                clean_tags.append("#water" if "river" in text else "#community")
        clean_tags, clean_categories = self._rebalance_subject_over_format(
            transcript, clean_tags, clean_categories
        )
        return (self._normalize_tags(clean_tags)[:max_tags], clean_categories[:3])

    def _ensure_non_empty_tags(
        self, tags: list[str], categories: list[str], transcript: str, max_tags: int
    ) -> list[str]:
        normalised = self._normalize_tags(tags)
        if normalised:
            return normalised[:max_tags]
        category_tags = [self._normalize_tag(c) for c in categories if c]
        category_tags = [t for t in category_tags if t]
        if category_tags:
            return self._normalize_tags(category_tags)[:max_tags]
        words = [w for w in re.findall("[a-zA-Z]{4,}", transcript.lower()) if w not in _STOPWORDS]
        if words:
            return self._normalize_tags([words[0]])[:max_tags]
        return []

    def _extract_transcript_words(self, transcript: str) -> set[str]:
        words = set()
        for w in re.split("[\\s\\.,;:!?\\-\\\"\\'()]+", transcript):
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
            cat_words = set(re.findall("[a-z]+", cat.lower()))
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
        cats = [c.strip() for c in categories or [] if isinstance(c, str) and c.strip()]
        if not cats and zero_shot_scores:
            ranked = sorted(zero_shot_scores.items(), key=lambda x: x[1], reverse=True)
            cats = [c for c, s in ranked if s >= 0.28][:max_categories]
        if wildlife_media_narrative(transcript):
            cats = [c for c in cats if c.lower() not in {"veterinary", "podcast"}]
        ranked = sorted(zero_shot_scores.items(), key=lambda x: x[1], reverse=True)
        additions = []
        for cat, score in ranked[:8]:
            if score < 0.32:
                continue
            if cat.lower() in _FORMAT_CATEGORIES:
                continue
            if wildlife_media_narrative(transcript) and cat.lower() == "veterinary":
                continue
            if cat not in cats:
                additions.append(cat)
        candidates = list(dict.fromkeys(additions + cats))
        # Prefer one readable label for the same subject (Politics/Political,
        # Environment/Environmental), and omit redundant catalogue compounds.
        for category in list(candidates):
            terms = self._subject_terms(category)
            equivalents = [c for c in candidates if self._subject_terms(c) == terms]
            if len(equivalents) > 1:
                preferred = min(equivalents, key=lambda c: (len(c), c.lower()))
                candidates = [c for c in candidates if c == preferred or c not in equivalents]
        ranked_candidates = sorted(candidates, key=lambda cat: zero_shot_scores.get(cat, 0), reverse=True)
        return [
            category for category in ranked_candidates
            if not any(self._subject_terms(other) < self._subject_terms(category) for other in candidates)
        ][:max_categories]

    def _build_tag_pool(
        self, transcript: str, all_tags: list[str], keyword_scores: dict
    ) -> list[str]:
        tx_words = self._extract_transcript_words(transcript)
        relevant = []
        for tag in dict.fromkeys(all_tags + list(keyword_scores)):
            tag_words = re.findall("[a-z]+", tag.lower())
            if len(tag_words) > 4 or len(set(tag_words)) != len(tag_words):
                continue
            overlap = len(set(tag_words) & tx_words)
            if overlap or tag in keyword_scores:
                relevant.append((tag, keyword_scores.get(tag, 0), overlap, len(tag_words)))
        relevant.sort(key=lambda row: (-row[1], -row[2] / max(row[3], 1), row[3], row[0]))
        return [row[0] for row in relevant[:120]]

    def _keyword_layer(
        self, transcript: str, segments: list[dict], keyword_rules: Mapping[str, str]
    ) -> dict:
        text_lower = transcript.lower()
        scores = {}
        for pattern, tag in keyword_rules.items():
            bounded = "|".join(f"\\b{p.strip()}\\b" for p in pattern.lower().split("|"))
            matches = len(re.findall(bounded, text_lower))
            if matches > 0:
                scores[tag] = min(1.0, matches * 0.15 + 0.4)
        if segments:
            seg_counter: Counter[str] = Counter()
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

    _ZS_TEMPLATE = "This example is {}."

    def _zero_shot_labels(self, transcript: str, labels: list[str]) -> dict:
        if not labels:
            return {"scores": {}}
        # Sentence-sized evidence suits the pinned sentence-pair model. Sample the
        # whole recording instead of discarding every subject after character 1024.
        windows = [
            sentence[start:start + 384]
            for sentence in re.split(r"(?<=[.!?])\s+", transcript.strip())
            for start in range(0, len(sentence), 384)
            if sentence[start:start + 384].strip()
        ]
        if len(windows) > 6:
            windows = [windows[index * (len(windows) - 1) // 5] for index in range(6)]
        canonical = {label.lower().lstrip("#").replace("-", " "): label for label in labels}
        evidence: dict[str, list[float]] = defaultdict(list)
        for window in windows:
            output = self._models().nli_sync(
                window, list(canonical), hypothesis_template=self._ZS_TEMPLATE, multi_label=True
            )
            for label, score in zip(output["labels"], output["scores"], strict=False):
                if label in canonical:
                    evidence[canonical[label]].append(float(score))
        # A passing mention must not dominate the subject of a whole recording.
        scores = {label: sum(values) / len(values) for label, values in evidence.items()}
        return {"scores": scores}

    def _get_sentiment(self, transcript: str) -> str:
        try:
            result = self._models().sentiment_sync(transcript[:512])
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

    _TAG_THRESHOLD = 0.6
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
        l1 = layer1.get("scores", {})
        l2c = layer2_cat.get("scores", {})
        l2t = layer2_tag.get("scores", {})
        merged_tag_scores: dict[str, float] = {}
        for tag in dict.fromkeys(all_tags):
            s1 = l1.get(tag, 0)
            s2 = l2t.get(tag, 0)
            score = s1 * 0.4 + s2 * 0.6 if s1 > 0 and s2 > 0 else max(s1, s2)
            merged_tag_scores[tag] = round(min(1.0, score), 4)
        ranked_tags = sorted(merged_tag_scores.items(), key=lambda x: x[1], reverse=True)
        tags = [t for t, s in ranked_tags if s >= self._TAG_THRESHOLD][:max_tags]
        cat_scores: dict[str, float] = {}
        for c in all_categories:
            s1 = l1.get(self._normalize_tag(c), 0)
            s2 = l2c.get(c, 0)
            score = s1 * 0.4 + s2 * 0.6 if s1 else s2
            cat_scores[c] = round(min(1.0, score), 4)
        ranked_cats = sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)
        categories = [c for c, s in ranked_cats if s >= self._CAT_THRESHOLD][:3]
        logger.debug("[CATEGORIZER] top_tag_scores=%s", ranked_tags[:8])
        logger.debug(
            "[CATEGORIZER] top_cat_scores=%s",
            sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)[:5],
        )
        return {
            "tags": tags,
            "categories": categories,
            "confidence_scores": {**merged_tag_scores, **cat_scores},
        }
