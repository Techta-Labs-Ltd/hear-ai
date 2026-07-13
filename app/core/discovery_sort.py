"""Sort helpers for discovery catalog items."""

from __future__ import annotations

from datetime import datetime
from typing import Any

VALID_DISCOVERY_SORTS = frozenset({"latest", "trending"})


def _parse_iso_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def discovery_latest_timestamp(item: dict) -> datetime:
    """Timestamp used for latest/recency sorting."""
    for key in ("latest_at", "published_at", "created_at"):
        parsed = _parse_iso_ts(item.get(key))
        if parsed is not None:
            return parsed
    return datetime.min.replace(tzinfo=None)


def discovery_trending_score(item: dict) -> float:
    raw = item.get("trending_score")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def sort_discovery_items(items: list[dict], sort: str = "latest") -> list[dict]:
    """Return a new list sorted by latest (recency) or trending (score then recency)."""
    mode = (sort or "latest").strip().lower()
    if mode not in VALID_DISCOVERY_SORTS:
        mode = "latest"
    if mode == "trending":
        return sorted(
            items,
            key=lambda row: (
                -discovery_trending_score(row),
                -discovery_latest_timestamp(row).timestamp(),
            ),
        )
    return sorted(
        items,
        key=lambda row: -discovery_latest_timestamp(row).timestamp(),
    )
