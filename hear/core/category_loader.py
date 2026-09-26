import re
import threading
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CategoryData:
    categories: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    keyword_rules: dict[str, str] = field(default_factory=dict)
    all_labels: list[str] = field(default_factory=list)


class CategoryLabels:
    @staticmethod
    def is_hierarchical_taxonomy_path(label: str) -> bool:
        return " > " in (label or "")

    @staticmethod
    def _taxonomy_path_to_tag(path: str) -> str:
        parts = [p.strip().lower() for p in (path or "").split(">") if p.strip()]
        if not parts:
            return ""
        slug_parts = []
        for part in parts:
            slug = re.sub("[^a-z0-9]+", "-", part).strip("-")
            if slug:
                slug_parts.append(slug)
        slug = "-".join(slug_parts)
        slug = re.sub("-+", "-", slug)
        return f"#{slug}" if slug else ""


class CategoryLoader:
    def __init__(self):
        self._data = CategoryData()
        self._lock = threading.RLock()
        self._loaded = False
        self._file_path: Path | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def load(self, path: str | Path | None = None) -> None:
        if path is None:
            raise RuntimeError("pipeline_catalog_not_loaded")
        self._load_file(Path(path))

    def load_snapshot(
        self,
        categories: list[str],
        tags: list[str],
        keyword_rules: dict[str, str],
    ) -> None:
        normalized_categories = [str(item).strip() for item in categories if str(item).strip()]
        normalized_tags = [str(item).strip() for item in tags if str(item).strip()]
        normalized_rules = {
            str(pattern).strip(): str(tag).strip()
            for pattern, tag in keyword_rules.items()
            if str(pattern).strip() and str(tag).strip()
        }
        with self._lock:
            self._file_path = None
            self._data = CategoryData(
                categories=normalized_categories,
                tags=normalized_tags,
                keyword_rules=normalized_rules,
                all_labels=normalized_categories + normalized_tags,
            )
            self._loaded = True

    def _load_file(self, path: Path) -> None:
        """Load an explicit catalog file for migration tools and isolated tests.

        Production calls omit ``path`` and use PostgreSQL as the source of truth.
        """
        categories: list[str] = []
        tags: list[str] = []
        keyword_rules: dict[str, str] = {}
        section = ""
        if path.is_file():
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("[") and line.endswith("]"):
                    section = line[1:-1].upper()
                elif section == "CATEGORIES":
                    categories.append(line)
                elif section == "TAGS" and line.startswith("#"):
                    tags.append(line)
                elif section == "KEYWORDS" and "=" in line:
                    pattern, tag = line.rsplit("=", 1)
                    keyword_rules[pattern.strip()] = tag.strip()
        with self._lock:
            self._file_path = path
            self._data = CategoryData(
                categories=categories,
                tags=tags,
                keyword_rules=keyword_rules,
                all_labels=categories + tags,
            )
            self._loaded = True

    def _save_file(self) -> None:
        if self._file_path is None:
            return
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "[CATEGORIES]",
            *self._data.categories,
            "",
            "[TAGS]",
            *self._data.tags,
            "",
            "[KEYWORDS]",
        ]
        lines.extend((f"{pattern} = {tag}" for pattern, tag in self._data.keyword_rules.items()))
        self._file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @property
    def data(self) -> CategoryData:
        if not self._loaded:
            raise RuntimeError("pipeline_catalog_not_loaded")
        with self._lock:
            return self._data

    def flat_catalog_categories(self) -> list[str]:
        if not self._loaded:
            raise RuntimeError("pipeline_catalog_not_loaded")
        with self._lock:
            return [
                c
                for c in self._data.categories
                if c.strip() and (not CategoryLabels.is_hierarchical_taxonomy_path(c))
            ]


category_loader = CategoryLoader()
