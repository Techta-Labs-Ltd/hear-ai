import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

from hear.models.database import CategoryLabel, KeywordRule, SessionLocal, TagLabel


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
        self._loaded = False
        self._file_path: Path | None = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def load(self, path: str | Path | None = None):
        if path is not None:
            self._load_file(Path(path))
            return
        db = SessionLocal()
        try:
            categories = [row.name for row in db.query(CategoryLabel).order_by(CategoryLabel.name).all()]
            tags = [row.name for row in db.query(TagLabel).order_by(TagLabel.name).all()]
            keyword_rules = {row.pattern: row.tag for row in db.query(KeywordRule).all()}
        finally:
            db.close()
        with self._lock:
            self._data = CategoryData(
                categories=categories,
                tags=tags,
                keyword_rules=keyword_rules,
                all_labels=categories + tags,
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
        lines = ["[CATEGORIES]", *self._data.categories, "", "[TAGS]", *self._data.tags, "", "[KEYWORDS]"]
        lines.extend(f"{pattern} = {tag}" for pattern, tag in self._data.keyword_rules.items())
        self._file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

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
        """Add any missing tags/categories to the catalog."""
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
            if tag.lower() in existing:
                return
            if self._file_path is not None:
                self._data.tags.append(tag)
                self._data.all_labels.append(tag)
                self._save_file()
                return
        db = SessionLocal()
        try:
            if not db.query(TagLabel).filter(TagLabel.name == tag).first():
                db.add(TagLabel(name=tag))
                db.commit()
        finally:
            db.close()
        with self._lock:
            self._data.tags.append(tag)
            self._data.all_labels.append(tag)

    def add_category(self, category: str):
        if not self._loaded:
            self.load()
        category = re.sub(r"\s+", " ", (category or "").strip())
        if not category:
            return
        with self._lock:
            existing = {c.lower() for c in self._data.categories}
            if category.lower() in existing:
                return
            if self._file_path is not None:
                self._data.categories.append(category)
                self._data.all_labels.append(category)
                self._save_file()
                return
        db = SessionLocal()
        try:
            if not db.query(CategoryLabel).filter(CategoryLabel.name == category).first():
                db.add(CategoryLabel(name=category))
                db.commit()
        finally:
            db.close()
        with self._lock:
            self._data.categories.append(category)
            self._data.all_labels.append(category)


category_loader = CategoryLoader()
