import os
import re
import threading
from dataclasses import dataclass, field

from app.config import settings


@dataclass
class DiscoveryTaxonomyData:
    paths: list[str] = field(default_factory=list)
    path_lookup: dict[str, str] = field(default_factory=dict)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


class DiscoveryTaxonomyLoader:
    def __init__(self):
        self._data = DiscoveryTaxonomyData()
        self._lock = threading.Lock()

    def load(self, path: str | None = None):
        path = path or settings.DISCOVERY_TAXONOMY_FILE
        paths: list[str] = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                section = None
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    if line.startswith("[") and line.endswith("]"):
                        section = line[1:-1].upper()
                        continue
                    if section == "TAXONOMY":
                        paths.append(line)
        lookup = {_norm(p): p for p in paths}
        with self._lock:
            self._data = DiscoveryTaxonomyData(paths=paths, path_lookup=lookup)

    @property
    def data(self) -> DiscoveryTaxonomyData:
        with self._lock:
            return self._data

    def match_paths_for_topics(self, topics: list[str]) -> list[str]:
        if not topics:
            return []
        matched: list[str] = []
        seen: set[str] = set()
        for topic in topics:
            nt = _norm(topic)
            if not nt:
                continue
            for key, path in self.data.path_lookup.items():
                if key in seen:
                    continue
                if nt in key or key in nt or any(part in nt for part in key.split(" > ")):
                    matched.append(path)
                    seen.add(key)
        return matched[:12]

    def paths_for_prompt(self, max_items: int = 40) -> str:
        paths = self.data.paths[:max_items]
        if not paths:
            return "none"
        return "\n".join(f"- {p}" for p in paths)

    def canonicalize_path(self, path: str) -> str:
        cleaned = re.sub(r"\s+", " ", (path or "").strip())
        if not cleaned:
            return ""
        key = _norm(cleaned)
        hit = self.data.path_lookup.get(key)
        if hit:
            return hit
        for k, canonical in self.data.path_lookup.items():
            if key == k or key in k or k in key:
                return canonical
        return cleaned


discovery_taxonomy_loader = DiscoveryTaxonomyLoader()
