import os
import threading
from dataclasses import dataclass, field

from app.config import settings


@dataclass
class CategoryData:
    categories: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    keyword_rules: dict[str, str] = field(default_factory=dict)
    all_labels: list[str] = field(default_factory=list)


class CategoryLoader:
    def __init__(self):
        self._data = CategoryData()
        self._lock = threading.Lock()

    def load(self, path: str = None):
        path = path or settings.CATEGORIES_FILE
        if not os.path.exists(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        section = None
        categories = []
        tags = []
        keyword_rules = {}

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].upper()
                continue
            if section == "CATEGORIES":
                categories.append(line)
            elif section == "TAGS":
                tags.append(line)
            elif section == "KEYWORDS":
                if "=" in line:
                    pattern, tag = line.rsplit("=", 1)
                    keyword_rules[pattern.strip()] = tag.strip()

        with self._lock:
            self._data = CategoryData(
                categories=categories,
                tags=tags,
                keyword_rules=keyword_rules,
                all_labels=categories + tags,
            )

    @property
    def data(self) -> CategoryData:
        with self._lock:
            return self._data

    def add_tag(self, tag: str):
        if not tag.startswith("#"):
            tag = f"#{tag}"
        with self._lock:
            if tag not in self._data.tags:
                self._data.tags.append(tag)
                self._data.all_labels.append(tag)
                self._save()

    def add_category(self, category: str):
        with self._lock:
            if category not in self._data.categories:
                self._data.categories.append(category)
                self._data.all_labels.append(category)
                self._save()

    def _save(self):
        path = settings.CATEGORIES_FILE
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("[CATEGORIES]\n")
            for c in self._data.categories:
                f.write(f"{c}\n")
            f.write("\n[TAGS]\n")
            for t in self._data.tags:
                f.write(f"{t}\n")
            f.write("\n[KEYWORDS]\n")
            for pattern, tag in self._data.keyword_rules.items():
                f.write(f"{pattern} = {tag}\n")


category_loader = CategoryLoader()
