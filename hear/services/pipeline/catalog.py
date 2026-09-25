from __future__ import annotations

from dataclasses import dataclass

import httpx

from hear.core.category_loader import category_loader
from hear.core.discovery_taxonomy import discovery_taxonomy_loader
from hear.core.keyword_loader import harm_keyword_loader


@dataclass(frozen=True, slots=True)
class PipelineCatalogSnapshot:
    categories: tuple[str, ...]
    tags: tuple[str, ...]
    keyword_rules: dict[str, str]
    taxonomy_paths: tuple[str, ...]
    harm_keywords: tuple[str, ...]


class PipelineCatalogClient:
    def __init__(
        self,
        backend_url: str,
        service_key: str,
        *,
        timeout_seconds: float = 20.0,
    ) -> None:
        self._url = f"{backend_url.rstrip('/')}/internal/ai/runtime/catalog"
        self._headers = {"X-Service-Key": service_key}
        self._timeout = timeout_seconds

    def fetch(self) -> PipelineCatalogSnapshot:
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(self._url, headers=self._headers)
            response.raise_for_status()
            payload = response.json()
        snapshot = PipelineCatalogSnapshot(
            categories=tuple(str(item) for item in payload.get("categories") or []),
            tags=tuple(str(item) for item in payload.get("tags") or []),
            keyword_rules={
                str(key): str(value)
                for key, value in dict(payload.get("keyword_rules") or {}).items()
            },
            taxonomy_paths=tuple(
                str(item) for item in payload.get("taxonomy_paths") or []
            ),
            harm_keywords=tuple(
                str(item) for item in payload.get("harm_keywords") or []
            ),
        )
        self.apply(snapshot)
        return snapshot

    @staticmethod
    def apply(snapshot: PipelineCatalogSnapshot) -> None:
        category_loader.load_snapshot(
            list(snapshot.categories),
            list(snapshot.tags),
            dict(snapshot.keyword_rules),
        )
        discovery_taxonomy_loader.load_paths(list(snapshot.taxonomy_paths))
        harm_keyword_loader.load_keywords(list(snapshot.harm_keywords))
