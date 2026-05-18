from __future__ import annotations

import asyncio
import re

from app.config import settings
from app.core.category_loader import category_loader
from app.core.discovery_taxonomy import discovery_taxonomy_loader
from app.models.discovery import (
    ContentDiscoveryProfile,
    DiscoveryEntities,
    content_description_from_discovery,
    discovery_to_callback_dict,
)
from app.services.llm_service import llm_service


def _categorization_hint(categorization: dict | None) -> str:
    if not isinstance(categorization, dict):
        return ""
    parts: list[str] = []
    tags = categorization.get("tags") or []
    cats = categorization.get("categories") or []
    if isinstance(tags, list):
        parts.extend(str(t) for t in tags[:14])
    if isinstance(cats, list):
        parts.extend(str(c) for c in cats[:10])
    return ", ".join(parts)[:500]


class DiscoveryService:
    def merge_controlled_tags(
        self,
        profile: ContentDiscoveryProfile,
        llm_tags: list[str] | None,
        categorization: dict | None,
    ) -> list[str]:
        paths: list[str] = []
        seen: set[str] = set()

        def _add(path: str) -> None:
            canonical = discovery_taxonomy_loader.canonicalize_path(path)
            if not canonical:
                return
            low = canonical.lower()
            if low in seen:
                return
            seen.add(low)
            paths.append(canonical)

        for tag in llm_tags or []:
            _add(str(tag))

        topics: list[str] = []
        if profile.main_topic:
            topics.append(profile.main_topic)
        topics.extend(profile.secondary_topics or [])
        for matched in discovery_taxonomy_loader.match_paths_for_topics(topics):
            _add(matched)

        cat_data = category_loader.data
        cat_names = {c.lower(): c for c in cat_data.categories}
        for topic in topics:
            nt = re.sub(r"\s+", " ", topic.strip().lower())
            for key, canonical in cat_names.items():
                if key in nt or nt in key:
                    _add(canonical)

        if isinstance(categorization, dict):
            for c in categorization.get("categories") or []:
                if isinstance(c, str) and c.strip():
                    _add(c.strip())

        return paths[:15]

    def map_freeform_tags(
        self,
        profile: ContentDiscoveryProfile,
        categorization: dict | None,
    ) -> list[str]:
        free: list[str] = []
        seen: set[str] = set()
        for src in (profile.freeform_tags or [], profile.secondary_topics or []):
            for item in src:
                t = str(item).strip().lstrip("#")
                if not t:
                    continue
                key = t.lower()
                if key in seen:
                    continue
                seen.add(key)
                free.append(t)
        if isinstance(categorization, dict):
            for tag in categorization.get("tags") or []:
                if isinstance(tag, str):
                    t = tag.lstrip("#").strip()
                    if t and t.lower() not in seen:
                        seen.add(t.lower())
                        free.append(t)
        return free[:20]

    def _fallback_from_categorization(
        self,
        transcript: str,
        categorization: dict | None,
        *,
        content_id: str | None,
        track_name: str,
    ) -> ContentDiscoveryProfile | None:
        if not (transcript or "").strip():
            return None
        cats = []
        tags = []
        if isinstance(categorization, dict):
            cats = [str(c) for c in (categorization.get("categories") or []) if c]
            tags = [str(t) for t in (categorization.get("tags") or []) if t]
        main = cats[0] if cats else (tags[0].lstrip("#") if tags else "Spoken audio")
        snippet = (transcript or "").strip()[:400]
        profile = ContentDiscoveryProfile(
            content_id=content_id,
            title_suggestion=track_name[:200] if track_name else None,
            summary_short=snippet[:500] if snippet else None,
            one_line_description=snippet[:200] if snippet else None,
            primary_genre="Spoken audio",
            main_topic=main,
            secondary_topics=cats[1:6] + [t.lstrip("#") for t in tags[:5]],
            embedding_source_text=snippet[:800] if snippet else None,
        )
        profile.controlled_tags = self.merge_controlled_tags(profile, [], categorization)
        profile.freeform_tags = self.map_freeform_tags(profile, categorization)
        return profile

    def _parse_llm_dict(self, raw: dict, *, content_id: str | None) -> ContentDiscoveryProfile:
        entities_raw = raw.get("entities")
        if isinstance(entities_raw, list):
            entities = DiscoveryEntities(people=[str(x) for x in entities_raw if x][:20])
        elif isinstance(entities_raw, dict):
            entities = DiscoveryEntities(
                people=[str(x) for x in entities_raw.get("people", []) if x][:20],
                animals=[str(x) for x in entities_raw.get("animals", []) if x][:20],
                products=[str(x) for x in entities_raw.get("products", []) if x][:20],
                apps=[str(x) for x in entities_raw.get("apps", []) if x][:20],
                technologies=[str(x) for x in entities_raw.get("technologies", []) if x][:20],
            )
        else:
            entities = DiscoveryEntities()
        confidence: dict[str, float] = {}
        conf_src = raw.get("confidence_scores") if isinstance(raw.get("confidence_scores"), dict) else raw.get("confidence")
        if isinstance(conf_src, dict):
            for k, v in conf_src.items():
                try:
                    confidence[str(k)] = max(0.0, min(1.0, float(v)))
                except (TypeError, ValueError):
                    pass

        def _lst(key: str, cap: int) -> list[str]:
            val = raw.get(key)
            if not isinstance(val, list):
                return []
            return [str(x).strip() for x in val if x and str(x).strip()][:cap]

        max_phrases = max(1, settings.DISCOVERY_MAX_SEARCH_PHRASES)
        summary_short = str(raw.get("summary_short") or raw.get("short_summary") or "").strip()
        themes = _lst("themes", 12) or _lst("key_themes", 12)
        audience = _lst("audience_groups", 12) or _lst("audience_relevance", 12)
        profile = ContentDiscoveryProfile(
            content_id=content_id,
            title_suggestion=str(raw.get("title_suggestion") or raw.get("title") or "").strip()[:300] or None,
            summary_short=summary_short[:1200] or None,
            summary_long=str(raw.get("summary_long") or "").strip()[:2500] or None,
            one_line_description=str(raw.get("one_line_description") or "").strip()[:500] or None,
            short_summary=summary_short[:1200] or None,
            primary_genre=str(raw.get("primary_genre") or "").strip()[:200] or None,
            main_topic=str(raw.get("main_topic") or "").strip()[:200] or None,
            secondary_topics=_lst("secondary_topics", 20),
            speaker=str(raw.get("speaker") or "").strip()[:200] or None,
            audience_relevance=audience,
            tone=_lst("tone", 8),
            entities=entities,
            key_themes=themes,
            search_phrases=_lst("search_phrases", max_phrases),
            recommendation_labels=_lst("recommendation_labels", 10),
            sensitivity_flags=_lst("sensitivity_flags", 8),
            confidence=confidence,
            freeform_tags=_lst("freeform_tags", 20),
            embedding_source_text=str(raw.get("embedding_source_text") or summary_short or "").strip()[:2000] or None,
        )
        profile.controlled_tags = _lst("controlled_tags", 15)
        return profile

    async def build_profile(
        self,
        transcript: str,
        *,
        content_id: str | None = None,
        track_name: str = "",
        duration_seconds: float | None = None,
        source: str | None = None,
        speaker: str | None = None,
        categorization: dict | None = None,
        prior_description: str | None = None,
        partial_transcript: bool = False,
    ) -> ContentDiscoveryProfile | None:
        if not settings.DISCOVERY_METADATA_ENABLED:
            profile = self._fallback_from_categorization(
                transcript, categorization, content_id=content_id, track_name=track_name
            )
            if profile:
                profile.duration_seconds = duration_seconds
                profile.source = source
                if speaker and not profile.speaker:
                    profile.speaker = speaker
            return profile
        if not (transcript or "").strip():
            return None

        hint = _categorization_hint(categorization)
        taxonomy_paths = discovery_taxonomy_loader.data.paths
        if llm_service.is_available:
            try:
                raw = await asyncio.to_thread(
                    llm_service.build_discovery_profile,
                    transcript,
                    track_name=track_name,
                    duration_seconds=duration_seconds,
                    categorization_hint=hint,
                    prior_description=prior_description,
                    partial_transcript=partial_transcript,
                    max_search_phrases=settings.DISCOVERY_MAX_SEARCH_PHRASES,
                    taxonomy_paths=taxonomy_paths,
                )
                if isinstance(raw, dict) and raw:
                    profile = self._parse_llm_dict(raw, content_id=content_id)
                    profile.duration_seconds = duration_seconds
                    profile.source = source
                    if speaker and not profile.speaker:
                        profile.speaker = speaker
                    llm_controlled = list(profile.controlled_tags or [])
                    profile.controlled_tags = self.merge_controlled_tags(
                        profile, llm_controlled, categorization
                    )
                    profile.freeform_tags = self.map_freeform_tags(profile, categorization)
                    return profile
            except Exception as exc:
                print(f"[DISCOVERY] Qwen profile failed: {exc}")

        profile = self._fallback_from_categorization(
            transcript, categorization, content_id=content_id, track_name=track_name
        )
        if profile:
            profile.duration_seconds = duration_seconds
            profile.source = source
            if speaker and not profile.speaker:
                profile.speaker = speaker
        return profile


discovery_service = DiscoveryService()


def discovery_result_bundle(
    profile: ContentDiscoveryProfile | None,
    *,
    duration_seconds: float | None = None,
    source: str | None = None,
) -> tuple[dict | None, str | None]:
    if profile is None:
        return None, None
    return (
        discovery_to_callback_dict(
            profile,
            duration_seconds=duration_seconds,
            source=source,
        ),
        content_description_from_discovery(profile),
    )
