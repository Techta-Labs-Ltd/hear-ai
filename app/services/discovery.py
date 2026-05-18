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

_FILENAME_TITLE = re.compile(r"^\d{8}[\d_\-a-fA-Z]*$")
_SPEAKER_PATTERNS = (
    re.compile(
        r"\b(?:this is|i am|i'm|my name is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:with|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+in\s+[A-Z]",
        re.IGNORECASE,
    ),
)


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
    def _infer_speaker(self, transcript: str) -> str | None:
        for pattern in _SPEAKER_PATTERNS:
            match = pattern.search(transcript or "")
            if match:
                name = match.group(1).strip()
                if len(name) >= 3:
                    return name[:200]
        return None

    def _looks_like_filename_title(self, title: str, track_name: str) -> bool:
        t = (title or "").strip()
        n = (track_name or "").strip()
        if not t:
            return True
        if n and t == n:
            return bool(_FILENAME_TITLE.match(t.replace(" ", "")))
        return bool(_FILENAME_TITLE.match(t.replace(" ", "").replace(".", "")))

    def _is_weak_profile(
        self,
        profile: ContentDiscoveryProfile,
        transcript: str,
        track_name: str,
    ) -> bool:
        tx = (transcript or "").strip()
        ss = (profile.summary_short or "").strip()
        if not ss or not (profile.one_line_description or "").strip():
            return True
        if len(profile.search_phrases or []) < 2:
            return True
        if len(profile.key_themes or []) < 1:
            return True
        if len(profile.audience_relevance or []) < 1:
            return True
        if profile.entities.is_empty() and len(tx.split()) > 40:
            return True
        if not (profile.speaker or "").strip() and self._infer_speaker(tx):
            return True
        if tx and len(ss) > 40 and ss.lower()[:72] == tx.lower()[:72]:
            return True
        if self._looks_like_filename_title(profile.title_suggestion or "", track_name):
            return True
        return False

    def _enrich_profile(
        self,
        profile: ContentDiscoveryProfile,
        transcript: str,
        categorization: dict | None,
        track_name: str,
    ) -> ContentDiscoveryProfile:
        tx = (transcript or "").strip()
        if not profile.speaker:
            profile.speaker = self._infer_speaker(tx)
        if profile.speaker and profile.speaker not in profile.entities.people:
            profile.entities.people = [profile.speaker] + list(profile.entities.people)
        if self._looks_like_filename_title(profile.title_suggestion or "", track_name):
            topic = (profile.main_topic or profile.primary_genre or "Audio").strip()
            profile.title_suggestion = f"{topic}: {profile.speaker or 'spoken piece'}"[:300]
        if not profile.summary_long and profile.summary_short:
            profile.summary_long = profile.summary_short
        if not profile.embedding_source_text or profile.embedding_source_text.lower()[:60] == tx.lower()[:60]:
            parts = [profile.primary_genre, profile.main_topic]
            parts.extend(profile.secondary_topics or [])
            parts.extend(profile.key_themes or [])
            profile.embedding_source_text = ". ".join(p for p in parts if p)[:2000] or None
        if len(profile.search_phrases or []) < 3:
            phrases: list[str] = []
            seen: set[str] = set()
            for src in (
                profile.secondary_topics or [],
                profile.key_themes or [],
                [profile.main_topic] if profile.main_topic else [],
            ):
                for item in src:
                    p = str(item).strip()
                    if not p:
                        continue
                    low = p.lower()
                    if low in seen:
                        continue
                    seen.add(low)
                    phrases.append(p)
            if isinstance(categorization, dict):
                for tag in categorization.get("tags") or []:
                    t = str(tag).lstrip("#").strip()
                    if t and t.lower() not in seen:
                        seen.add(t.lower())
                        phrases.append(t)
            profile.search_phrases = (profile.search_phrases or []) + phrases
            profile.search_phrases = profile.search_phrases[: settings.DISCOVERY_MAX_SEARCH_PHRASES]
        if not profile.key_themes and profile.secondary_topics:
            profile.key_themes = profile.secondary_topics[:6]
        if not profile.audience_relevance and profile.main_topic:
            profile.audience_relevance = [f"Listeners interested in {profile.main_topic}"]
        if not profile.recommendation_labels and profile.primary_genre:
            profile.recommendation_labels = [
                f"For listeners interested in {profile.primary_genre}"
            ]
        if not profile.one_line_description and profile.summary_short:
            profile.one_line_description = profile.summary_short[:500]
        if not profile.summary_short or self._summary_reads_like_transcript(profile.summary_short, tx):
            profile.summary_short = self._compose_summary_blurb(profile, tx)
            profile.short_summary = profile.summary_short
        if profile.primary_genre and profile.primary_genre.strip().lower() in (
            "spoken audio",
            "podcast",
            "podcast episode",
        ):
            profile.primary_genre = profile.main_topic or profile.primary_genre
        return profile

    def _summary_reads_like_transcript(self, summary: str, transcript: str) -> bool:
        s = (summary or "").strip().lower()
        t = (transcript or "").strip().lower()
        if not s or not t or len(s) < 40:
            return False
        return s[:80] == t[:80]

    def _compose_summary_blurb(self, profile: ContentDiscoveryProfile, transcript: str) -> str:
        speaker = (profile.speaker or "").strip()
        topic = (profile.main_topic or "").strip()
        genre = (profile.primary_genre or "").strip()
        themes = profile.key_themes or []
        if speaker and topic:
            base = f"{speaker} discusses {topic}."
        elif topic:
            base = f"A conversation about {topic}."
        elif genre and genre.lower() not in ("spoken audio", "podcast"):
            base = f"A {genre.lower()} piece."
        else:
            base = "A spoken-word audio piece."
        if themes:
            base += f" Themes include {themes[0].lower()}"
            if len(themes) > 1:
                base += f" and {themes[1].lower()}"
            base += "."
        if not base.endswith(".") and profile.secondary_topics:
            base += f" Topics include {profile.secondary_topics[0]}."
        return base[:500]

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
        speaker = self._infer_speaker(transcript)
        profile = ContentDiscoveryProfile(
            content_id=content_id,
            title_suggestion=None,
            primary_genre=main,
            main_topic=main,
            secondary_topics=cats[1:6] + [t.lstrip("#") for t in tags[:5]],
            speaker=speaker,
        )
        profile = self._enrich_profile(profile, transcript, categorization, track_name)
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
            for attempt, strict in enumerate((False, True)):
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
                        strict=strict,
                    )
                    if not isinstance(raw, dict) or not raw:
                        continue
                    profile = self._parse_llm_dict(raw, content_id=content_id)
                    profile = self._enrich_profile(
                        profile, transcript, categorization, track_name
                    )
                    if self._is_weak_profile(profile, transcript, track_name) and attempt == 0:
                        print("[DISCOVERY] Qwen profile weak — retrying with strict prompt")
                        continue
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
                    print(f"[DISCOVERY] Qwen profile failed (attempt {attempt + 1}): {exc}")
                    if attempt == 1:
                        break
        elif not llm_service.is_available:
            print(
                "[DISCOVERY] Qwen not loaded — enable QWEN_LLM_ENABLED=true and GPU; "
                "using categorization fallback (limited metadata)"
            )

        profile = self._fallback_from_categorization(
            transcript, categorization, content_id=content_id, track_name=track_name
        )
        if profile:
            profile.duration_seconds = duration_seconds
            profile.source = source
            if speaker and not profile.speaker:
                profile.speaker = speaker
            profile = self._enrich_profile(profile, transcript, categorization, track_name)
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
