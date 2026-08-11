from __future__ import annotations

import asyncio
import re

from hear.config import settings
from hear.core.category_loader import category_loader
from hear.core.content_context import (
    assistive_tech_narrative,
    filter_controlled_taxonomy_paths,
    filter_freeform_tag_labels,
    is_assistive_taxonomy_path,
    tech_history_narrative,
)
from hear.core.discovery_taxonomy import _norm, discovery_taxonomy_loader
from hear.models.discovery import (
    ContentDiscoveryProfile,
    DiscoveryEntities,
    coerce_discovery_source,
    content_description_from_discovery,
    discovery_to_callback_dict,
)
from hear.services.llm import get_llm_service

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
_NON_PERSON_SPEAKER = re.compile(
    r"\b(?:glasses|hardware|wearable|device|technology|recruitment|production|podcast|"
    r"accessibility|assistive)\b",
    re.IGNORECASE,
)
_NON_PERSON_SPEAKER_NAMES = frozenset(
    {
        "minidisc",
        "mini disc",
        "walkman",
        "cassette",
        "betamax",
        "ipod",
        "napster",
        "sony",
        "philips",
        "compact disc",
        "smart glasses",
        "hardware",
        "technology",
        "documentary",
        "podcast",
        "speaker",
    }
)
_PERSON_NAME_SHAPE = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}$")
_SPEAKER_CAPTURE_STOP = frozenset(
    {"in", "from", "at", "and", "with", "of", "for", "on", "to", "by"}
)


def _norm_label(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


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
    def _trim_speaker_capture(self, name: str) -> str:
        parts: list[str] = []
        for word in (name or "").split():
            if word.lower() in _SPEAKER_CAPTURE_STOP:
                break
            if not re.match(r"^[A-Z][a-z]+$", word):
                break
            parts.append(word)
            if len(parts) >= 4:
                break
        return " ".join(parts)

    def _infer_speaker(self, transcript: str) -> str | None:
        for pattern in _SPEAKER_PATTERNS:
            match = pattern.search(transcript or "")
            if match:
                name = self._trim_speaker_capture(match.group(1).strip())
                if len(name) >= 3 and self._is_plausible_speaker(name, transcript, from_regex=True):
                    return name[:200]
        return None

    def _is_plausible_speaker(
        self,
        name: str,
        transcript: str,
        *,
        from_regex: bool = False,
    ) -> bool:
        s = (name or "").strip()
        if not s or len(s) < 3:
            return False
        if " > " in s:
            return False
        low = _norm_label(s)
        if low in _NON_PERSON_SPEAKER_NAMES:
            return False
        if low in discovery_taxonomy_loader.taxonomy_label_terms():
            return False
        if _NON_PERSON_SPEAKER.search(s):
            return False
        if from_regex:
            return bool(_PERSON_NAME_SHAPE.match(s))
        if not _PERSON_NAME_SHAPE.match(s):
            return False
        intro = re.compile(
            rf"\b(?:this is|i am|i'm|my name is|called|named)\s+{re.escape(s)}\b",
            re.IGNORECASE,
        )
        return bool(intro.search(transcript or ""))

    def _sanitize_speaker(
        self,
        speaker: str | None,
        transcript: str,
        *,
        trusted: bool = False,
    ) -> str | None:
        if trusted:
            s = (speaker or "").strip()
            return s[:200] if s else None
        s = (speaker or "").strip()
        if not s:
            return None
        if self._is_plausible_speaker(s, transcript):
            return s[:200]
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
        profile.speaker = self._sanitize_speaker(profile.speaker, tx)
        if not profile.speaker:
            profile.speaker = self._infer_speaker(tx)
        if profile.speaker and profile.speaker not in profile.entities.people:
            profile.entities.people = [profile.speaker] + list(profile.entities.people)
        profile.entities.people = [
            p
            for p in profile.entities.people
            if _norm_label(p) != _norm_label(profile.speaker or "")
            and self._is_plausible_speaker(p, tx)
        ]
        if profile.speaker and profile.speaker not in profile.entities.people:
            profile.entities.people = [profile.speaker] + profile.entities.people
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
        *,
        transcript: str = "",
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
            raw = str(tag).strip()
            if " > " in raw:
                _add(raw)

        topics: list[str] = []
        if profile.main_topic:
            topics.append(profile.main_topic)
        topics.extend(profile.secondary_topics or [])
        for matched in discovery_taxonomy_loader.match_paths_for_topics(topics):
            _add(matched)

        return filter_controlled_taxonomy_paths(transcript, paths)[:15]

    def map_freeform_tags(
        self,
        profile: ContentDiscoveryProfile,
        categorization: dict | None,
        *,
        transcript: str = "",
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
        return filter_freeform_tag_labels(transcript, free)[:20]

    def _finalize_discovery_profile(
        self,
        profile: ContentDiscoveryProfile,
        transcript: str,
    ) -> ContentDiscoveryProfile:
        tx = (transcript or "").strip()
        profile.speaker = self._sanitize_speaker(profile.speaker, tx)
        if profile.speaker and not self._is_plausible_speaker(profile.speaker, tx):
            profile.speaker = None
        genre = (profile.primary_genre or "").strip()
        if " > " in genre:
            if is_assistive_taxonomy_path(genre) and not assistive_tech_narrative(tx.lower()):
                profile.primary_genre = genre.split(" > ")[-1].strip()
            elif tech_history_narrative(tx.lower()):
                profile.primary_genre = "Technology history"
            else:
                profile.primary_genre = genre.split(" > ")[-1].strip()
        if tech_history_narrative(tx.lower()) and not assistive_tech_narrative(tx.lower()):
            if (profile.main_topic or "").strip().lower() in ("accessibility", "wildlife"):
                profile.main_topic = "Technology"
        profile.entities.people = [
            p for p in profile.entities.people if self._is_plausible_speaker(p, tx)
        ]
        profile.entities.products = [
            p
            for p in profile.entities.products
            if _norm_label(p) not in _NON_PERSON_SPEAKER_NAMES
        ]
        return profile

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
        profile.controlled_tags = self.merge_controlled_tags(
            profile, [], categorization, transcript=transcript
        )
        profile.freeform_tags = self.map_freeform_tags(
            profile, categorization, transcript=transcript
        )
        return self._finalize_discovery_profile(profile, transcript)

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
                profile.source = coerce_discovery_source(source) or None
                if speaker and not profile.speaker:
                    profile.speaker = self._sanitize_speaker(
                        speaker, transcript, trusted=True
                    )
            return profile
        if not (transcript or "").strip():
            return None

        hint = _categorization_hint(categorization)
        taxonomy_paths = discovery_taxonomy_loader.data.paths
        if get_llm_service().is_available:
            for attempt, strict in enumerate((False, True)):
                try:
                    raw = await asyncio.to_thread(
                        get_llm_service().build_discovery_profile,
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
                    profile.source = coerce_discovery_source(source) or None
                    if speaker and not profile.speaker:
                        profile.speaker = self._sanitize_speaker(
                            speaker, transcript, trusted=True
                        )
                    llm_controlled = list(profile.controlled_tags or [])
                    profile.controlled_tags = self.merge_controlled_tags(
                        profile, llm_controlled, categorization, transcript=transcript
                    )
                    profile.freeform_tags = self.map_freeform_tags(
                        profile, categorization, transcript=transcript
                    )
                    return self._finalize_discovery_profile(profile, transcript)
                except Exception as exc:
                    print(f"[DISCOVERY] Qwen profile failed (attempt {attempt + 1}): {exc}")
                    if attempt == 1:
                        break
        elif not get_llm_service().is_available:
            print(
                "[DISCOVERY] Qwen not loaded — enable QWEN_LLM_ENABLED=true and GPU; "
                "using categorization fallback (limited metadata)"
            )

        profile = self._fallback_from_categorization(
            transcript, categorization, content_id=content_id, track_name=track_name
        )
        if profile:
            profile.duration_seconds = duration_seconds
            profile.source = coerce_discovery_source(source) or None
            profile = self._enrich_profile(profile, transcript, categorization, track_name)
            if speaker and not profile.speaker:
                profile.speaker = self._sanitize_speaker(
                    speaker, transcript, trusted=True
                )
        return profile


_discovery_service = None


def get_discovery_service():
    global _discovery_service
    if _discovery_service is None:
        _discovery_service = DiscoveryService()
    return _discovery_service


def discovery_result_bundle(
    profile: ContentDiscoveryProfile | None,
    *,
    duration_seconds: float | None = None,
    source: str | None = None,
    published_at: str | None = None,
    trending_score: float | None = None,
) -> tuple[dict | None, str | None]:
    if profile is None:
        return None, None
    if profile is not None:
        if published_at and not profile.published_at:
            profile.published_at = published_at
        if trending_score is not None and profile.trending_score is None:
            profile.trending_score = trending_score
        if profile.published_at and not profile.latest_at:
            profile.latest_at = profile.published_at
    return (
        discovery_to_callback_dict(
            profile,
            duration_seconds=duration_seconds,
            source=source,
            published_at=published_at,
            trending_score=trending_score,
        ),
        content_description_from_discovery(profile),
    )
