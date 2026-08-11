"""Shared transcript context checks for categorization and discovery."""

from __future__ import annotations

_ASSISTIVE_TECH_TERMS = (
    "guide dog",
    "guide dogs",
    "visually impaired",
    "visual impairment",
    "sighted people",
    "sighted person",
    "meta ray",
    "ray-ban",
    "smart glasses",
    "hey meta",
    "orion",
    "assistive",
    "independence isn't",
    "independence is about",
    "blind",
    "partially sighted",
    "object recognition",
    "ai assistant",
)

_WILDLIFE_MEDIA_TERMS = (
    "wildlife",
    "wildlife photographer",
    "wildlife award",
    "wildlife trust",
    "photographer",
    "photo competition",
    "photography competition",
    "graphic designer",
    "won the",
    "has won",
    "award for",
    "country park",
    "rewilding",
    "great crested",
    "grebe",
    "great crested grebe",
    "bird",
    "birds",
    "chicks",
    "mating",
    "national park",
    "nature reserve",
    "sanctuary",
    "video titled",
)

_TECH_HISTORY_TERMS = (
    "minidisc",
    "mini-disc",
    "mini disc",
    "betamax",
    "walkman",
    "compact disc",
    "compact cassette",
    "cassette",
    "format war",
    "magneto-optical",
    "a-trac",
    "atrac",
    "netmd",
    "himd",
    "digital compact cassette",
    "philips",
    "matsushita",
    "rewritable",
    "mixtape",
    "audio format",
    "physical media",
    "ipod",
    "napster",
    "demucs",
)

_ASSISTIVE_TAXONOMY_PREFIXES = (
    "accessibility >",
)

_MISLEADING_FREEFORM_WHEN_NOT_WILDLIFE = frozenset(
    {"wildlife", "photography", "photographer", "nature", "animals", "grebe", "rewilding"}
)

_MISLEADING_FREEFORM_WHEN_NOT_ASSISTIVE = frozenset(
    {
        "accessibility",
        "assistive technology",
        "smart glasses",
        "smartglasses",
        "guide dogs",
        "guidedogs",
        "guidedog",
        "guide dog",
        "visual impairment",
        "visually impaired",
    }
)


def assistive_tech_narrative(text: str) -> bool:
    low = (text or "").lower()
    return sum(1 for term in _ASSISTIVE_TECH_TERMS if term in low) >= 2


def wildlife_media_narrative(text: str) -> bool:
    low = (text or "").lower()
    return sum(1 for term in _WILDLIFE_MEDIA_TERMS if term in low) >= 2


def tech_history_narrative(text: str) -> bool:
    low = (text or "").lower()
    return sum(1 for term in _TECH_HISTORY_TERMS if term in low) >= 2


def is_assistive_taxonomy_path(path: str) -> bool:
    low = (path or "").strip().lower()
    return any(low.startswith(p) for p in _ASSISTIVE_TAXONOMY_PREFIXES)


def filter_freeform_tag_labels(transcript: str, labels: list[str]) -> list[str]:
    low_tx = (transcript or "").lower()
    assistive = assistive_tech_narrative(low_tx)
    wildlife = wildlife_media_narrative(low_tx)
    out: list[str] = []
    seen: set[str] = set()
    for raw in labels or []:
        label = str(raw).strip().lstrip("#")
        if not label:
            continue
        key = label.lower()
        if key in seen:
            continue
        if not assistive and key in _MISLEADING_FREEFORM_WHEN_NOT_ASSISTIVE:
            continue
        if not wildlife and key in _MISLEADING_FREEFORM_WHEN_NOT_WILDLIFE:
            continue
        if tech_history_narrative(low_tx) and key in _MISLEADING_FREEFORM_WHEN_NOT_WILDLIFE:
            continue
        seen.add(key)
        out.append(label)
    return out


def filter_controlled_taxonomy_paths(transcript: str, paths: list[str]) -> list[str]:
    low_tx = (transcript or "").lower()
    assistive = assistive_tech_narrative(low_tx)
    out: list[str] = []
    seen: set[str] = set()
    for path in paths or []:
        cleaned = str(path).strip()
        if not cleaned or " > " not in cleaned:
            continue
        low = cleaned.lower()
        if low in seen:
            continue
        if is_assistive_taxonomy_path(cleaned) and not assistive:
            continue
        seen.add(low)
        out.append(cleaned)
    return out
