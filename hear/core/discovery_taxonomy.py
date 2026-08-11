import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from hear.models.database import SessionLocal, TaxonomyPath


@dataclass
class DiscoveryTaxonomyData:
    paths: list[str] = field(default_factory=list)
    path_lookup: dict[str, str] = field(default_factory=dict)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _hierarchical_segments(key: str) -> list[str]:
    return [_norm(part) for part in (key or "").split(" > ") if part.strip()]


def _topic_matches_taxonomy_path(topic: str, key: str) -> bool:
    """Avoid matching generic words (e.g. technology) inside longer path segments."""
    nt = _norm(topic)
    if not nt or len(nt) < 3:
        return False
    if nt == key:
        return True
    if " > " in key:
        segments = _hierarchical_segments(key)
        if nt in segments:
            return True
        if len(nt) >= 5 and any(nt == seg for seg in segments):
            return True
        if any(len(seg) >= 5 and seg in nt for seg in segments):
            return True
        return False
    if nt in key or key in nt:
        return True
    tokens = [t for t in re.findall(r"[a-z]{4,}", key)]
    return bool(tokens) and sum(1 for t in tokens if t in nt) >= min(2, len(tokens))


class DiscoveryTaxonomyLoader:
    def __init__(self):
        self._data = DiscoveryTaxonomyData()
        self._lock = threading.Lock()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def load(self, path: str | Path | None = None):
        if path is not None:
            paths = []
            section = ""
            source = Path(path)
            if source.is_file():
                for raw_line in source.read_text(encoding="utf-8").splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    if line.startswith("[") and line.endswith("]"):
                        section = line[1:-1].upper()
                    elif section == "TAXONOMY":
                        paths.append(line)
        else:
            db = SessionLocal()
            try:
                paths = [row.path for row in db.query(TaxonomyPath).order_by(TaxonomyPath.path).all()]
            finally:
                db.close()
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
                if _topic_matches_taxonomy_path(nt, key):
                    matched.append(path)
                    seen.add(key)
        return matched[:12]

    def canonicalize_path(self, path: str) -> str:
        cleaned = re.sub(r"\s+", " ", (path or "").strip())
        if not cleaned:
            return ""
        key = _norm(cleaned)
        hit = self.data.path_lookup.get(key)
        if hit:
            return hit
        for k, canonical in self.data.path_lookup.items():
            if _topic_matches_taxonomy_path(key, k):
                return canonical
        return cleaned

    def taxonomy_label_terms(self) -> frozenset[str]:
        """Normalized segment and full-path labels — not valid human speaker names."""
        terms: set[str] = set()
        for path in self.data.paths:
            terms.add(_norm(path))
            for segment in path.split(" > "):
                seg = segment.strip()
                if seg:
                    terms.add(_norm(seg))
        return frozenset(terms)


discovery_taxonomy_loader = DiscoveryTaxonomyLoader()
