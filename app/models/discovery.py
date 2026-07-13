from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, model_validator


def coerce_discovery_source(*candidates: Any) -> str:
    for val in candidates:
        if val is None:
            continue
        if isinstance(val, str):
            out = val.strip()
            if out:
                return out
        if isinstance(val, dict):
            for key in ("source", "name", "label", "type"):
                inner = val.get(key)
                if isinstance(inner, str) and inner.strip():
                    return inner.strip()
    return ""


class DiscoveryEntities(BaseModel):
    people: list[str] = Field(default_factory=list)
    animals: list[str] = Field(default_factory=list)
    products: list[str] = Field(default_factory=list)
    apps: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return not any(
            (
                self.people,
                self.animals,
                self.products,
                self.apps,
                self.technologies,
            )
        )

    def to_callback_dict(self) -> dict:
        return {
            "people": list(self.people),
            "animals": list(self.animals),
            "products": list(self.products),
            "apps": list(self.apps),
            "technologies": list(self.technologies),
        }


class ContentDiscoveryProfile(BaseModel):
    content_id: str | None = None
    title_suggestion: str | None = None
    summary_short: str | None = None
    summary_long: str | None = None
    one_line_description: str | None = None
    short_summary: str | None = None
    primary_genre: str | None = None
    main_topic: str | None = None
    secondary_topics: list[str] = Field(default_factory=list)
    speaker: str | None = None
    source: str | None = None
    duration_seconds: float | None = None
    audience_relevance: list[str] = Field(default_factory=list)
    tone: list[str] = Field(default_factory=list)
    entities: DiscoveryEntities = Field(default_factory=DiscoveryEntities)
    key_themes: list[str] = Field(default_factory=list)
    search_phrases: list[str] = Field(default_factory=list)
    recommendation_labels: list[str] = Field(default_factory=list)
    sensitivity_flags: list[str] = Field(default_factory=list)
    confidence: dict[str, float] = Field(default_factory=dict)
    controlled_tags: list[str] = Field(default_factory=list)
    freeform_tags: list[str] = Field(default_factory=list)
    embedding_source_text: str | None = None
    published_at: str | None = None
    latest_at: str | None = None
    trending_score: float | None = None

    @model_validator(mode="after")
    def sync_summary_aliases(self) -> ContentDiscoveryProfile:
        if self.short_summary and not self.summary_short:
            self.summary_short = self.short_summary
        elif self.summary_short and not self.short_summary:
            self.short_summary = self.summary_short
        return self


def content_description_from_discovery(profile: ContentDiscoveryProfile | dict | None) -> str | None:
    if profile is None:
        return None
    if isinstance(profile, dict):
        for key in ("one_line_description", "summary_short", "short_summary", "summary_long"):
            val = profile.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return None
    for val in (profile.one_line_description, profile.summary_short, profile.short_summary, profile.summary_long):
        if val and str(val).strip():
            return str(val).strip()
    return None


def flatten_entities(entities: DiscoveryEntities | dict | None) -> list[str]:
    if entities is None:
        return []
    if isinstance(entities, DiscoveryEntities):
        bucket = entities.to_callback_dict()
    elif isinstance(entities, dict):
        bucket = entities
    else:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for key in ("people", "animals", "products", "apps", "technologies"):
        for item in bucket.get(key) or []:
            s = str(item).strip()
            if not s:
                continue
            low = s.lower()
            if low in seen:
                continue
            seen.add(low)
            out.append(s)
    return out


def discovery_to_callback_dict(
    profile: ContentDiscoveryProfile | dict | None,
    *,
    duration_seconds: float | None = None,
    source: str | None = None,
    created_at: str | None = None,
    published_at: str | None = None,
    trending_score: float | None = None,
) -> dict | None:
    if profile is None:
        return None
    if isinstance(profile, dict):
        profile = ContentDiscoveryProfile.model_validate(profile)

    dur = duration_seconds if duration_seconds is not None else profile.duration_seconds
    ts = created_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    pub = (published_at or profile.published_at or "").strip() or None
    latest = (profile.latest_at or "").strip() or pub or ts
    trend_raw = trending_score if trending_score is not None else profile.trending_score
    try:
        trend = float(trend_raw) if trend_raw is not None else 0.0
    except (TypeError, ValueError):
        trend = 0.0
    summary_short = (profile.summary_short or profile.short_summary or "").strip()
    entities_obj = profile.entities.to_callback_dict()
    confidence = dict(profile.confidence or {})
    themes = list(profile.key_themes or [])
    audience = list(profile.audience_relevance or [])

    return {
        "content_id": profile.content_id or "",
        "title_suggestion": (profile.title_suggestion or "").strip(),
        "short_summary": summary_short,
        "summary_short": summary_short,
        "summary_long": (profile.summary_long or "").strip(),
        "one_line_description": (profile.one_line_description or "").strip(),
        "primary_genre": (profile.primary_genre or "").strip(),
        "main_topic": (profile.main_topic or "").strip(),
        "secondary_topics": list(profile.secondary_topics or []),
        "speaker": (profile.speaker or "").strip(),
        "source": coerce_discovery_source(source, profile.source),
        "duration_seconds": int(dur) if dur is not None else 0,
        "audience_relevance": audience,
        "tone": list(profile.tone or []),
        "entities": entities_obj,
        "entities_flat": flatten_entities(profile.entities),
        "key_themes": themes,
        "search_phrases": list(profile.search_phrases or []),
        "recommendation_labels": list(profile.recommendation_labels or []),
        "sensitivity_flags": list(profile.sensitivity_flags or []),
        "confidence": confidence,
        "controlled_tags": list(profile.controlled_tags or []),
        "freeform_tags": list(profile.freeform_tags or []),
        "embedding_source_text": (profile.embedding_source_text or "").strip(),
        "created_at": ts,
        "published_at": pub or "",
        "latest_at": latest,
        "trending_score": trend,
        "id": profile.content_id or "",
        "title": (profile.title_suggestion or "").strip(),
        "themes": themes,
        "audience_groups": audience,
        "confidence_scores": confidence,
    }
