import os
import re
import threading
from dataclasses import dataclass, field

from app.config import settings


@dataclass
class CategoryData:
    categories: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    keyword_rules: dict[str, str] = field(default_factory=dict)
    all_labels: list[str] = field(default_factory=list)


def is_hierarchical_taxonomy_path(label: str) -> bool:
    return " > " in (label or "")


def _taxonomy_path_to_tag(path: str) -> str:
    parts = [p.strip().lower() for p in (path or "").split(">") if p.strip()]
    if not parts:
        return ""
    slug_parts = []
    for part in parts:
        slug = re.sub(r"[^a-z0-9]+", "-", part).strip("-")
        if slug:
            slug_parts.append(slug)
    slug = "-".join(slug_parts)
    slug = re.sub(r"-+", "-", slug)
    return f"#{slug}" if slug else ""


class CategoryLoader:
    def __init__(self):
        self._data = CategoryData()
        self._lock = threading.RLock()
        self._file_path = settings.CATEGORIES_FILE
        self._loaded = False

    def load(self, path: str = None):
        path = path or settings.CATEGORIES_FILE
        self._file_path = path
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
                if line.startswith("#"):
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
            self._loaded = True

    @property
    def data(self) -> CategoryData:
        if not self._loaded:
            self.load()
        with self._lock:
            return self._data

    def flat_catalog_categories(self) -> list[str]:
        """Editorial categories for NLI/Qwen — excludes discovery hierarchy paths."""
        if not self._loaded:
            self.load()
        with self._lock:
            return [
                c for c in self._data.categories
                if c.strip() and not is_hierarchical_taxonomy_path(c)
            ]

    def prune_hierarchical_categories(self) -> int:
        """Remove taxonomy paths mistakenly stored as flat categories."""
        if not self._loaded:
            self.load()
        with self._lock:
            before = len(self._data.categories)
            self._data.categories = [
                c for c in self._data.categories if not is_hierarchical_taxonomy_path(c)
            ]
            removed = before - len(self._data.categories)
            if removed:
                self._data.all_labels = self._data.categories + self._data.tags
                self._save()
            return removed

    def import_discovery_taxonomy(self, taxonomy_paths: list[str]) -> tuple[list[str], list[str]]:
        """Merge discovery taxonomy into tags + leaf labels only (not full paths as categories)."""
        if not self._loaded:
            self.load()
        added_tags: list[str] = []
        added_cats: list[str] = []
        for path in taxonomy_paths or []:
            cleaned = re.sub(r"\s+", " ", (path or "").strip())
            if not cleaned:
                continue
            with self._lock:
                cat_names = {c.lower() for c in self._data.categories}
            leaf = cleaned.split(" > ")[-1].strip() if " > " in cleaned else cleaned
            if leaf and leaf.lower() not in cat_names:
                self.add_category(leaf)
                added_cats.append(leaf)
                cat_names.add(leaf.lower())
            tax_tag = _taxonomy_path_to_tag(cleaned)
            if tax_tag:
                with self._lock:
                    tag_names = {t.lower() for t in self._data.tags}
                if tax_tag.lower() not in tag_names:
                    self.add_tag(tax_tag)
                    added_tags.append(tax_tag)
        return added_tags, added_cats

    def ensure_labels(
        self,
        tags: list[str] | None = None,
        categories: list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Add any missing tags/categories to the catalog and save categories.txt."""
        if not self._loaded:
            self.load()
        new_tags: list[str] = []
        new_cats: list[str] = []
        for raw in tags or []:
            tag = raw if str(raw).startswith("#") else f"#{raw}"
            tag = f"#{str(tag).lstrip('#').strip().lower().replace(' ', '-')}"
            tag = re.sub(r"[^#a-z0-9\-]", "", tag)
            if not tag or tag == "#":
                continue
            with self._lock:
                existed = tag.lower() in {t.lower() for t in self._data.tags}
            if not existed:
                self.add_tag(tag)
                new_tags.append(tag)
        for raw in categories or []:
            cat = re.sub(r"\s+", " ", str(raw or "").strip())
            if not cat:
                continue
            if is_hierarchical_taxonomy_path(cat):
                cat = cat.split(" > ")[-1].strip()
            if not cat:
                continue
            with self._lock:
                existed = cat.lower() in {c.lower() for c in self._data.categories}
            if not existed:
                self.add_category(cat)
                new_cats.append(cat)
        return new_tags, new_cats

    def add_tag(self, tag: str):
        if not self._loaded:
            self.load()
        if not tag.startswith("#"):
            tag = f"#{tag}"
        tag = re.sub(r"[^#a-z0-9\-]", "", tag.lower())
        if tag == "#":
            return
        with self._lock:
            existing = {t.lower() for t in self._data.tags}
            if tag.lower() not in existing:
                self._data.tags.append(tag)
                self._data.all_labels.append(tag)
                self._save()

    def add_category(self, category: str):
        if not self._loaded:
            self.load()
        category = re.sub(r"\s+", " ", (category or "").strip())
        if not category:
            return
        with self._lock:
            existing = {c.lower() for c in self._data.categories}
            if category.lower() not in existing:
                self._data.categories.append(category)
                self._data.all_labels.append(category)
                self._save()

    def _save(self):
        path = self._file_path or settings.CATEGORIES_FILE
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with self._lock:
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
