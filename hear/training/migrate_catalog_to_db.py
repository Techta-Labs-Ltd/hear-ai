"""One-time migration: import the legacy categories.txt / discovery_taxonomy.txt /
harm_keywords.txt files into Postgres (category_labels, tag_labels, keyword_rules,
taxonomy_paths, harm_keywords). Run once by hand:
    uv run python -m hear.training.migrate_catalog_to_db
"""
import json
import os

from hear.models.database import (
    CategoryLabel,
    HarmKeyword,
    KeywordRule,
    SessionLocal,
    TagLabel,
    TaxonomyPath,
    init_db,
)


def _parse_categories_txt(path: str) -> tuple[list[str], list[str], dict[str, str]]:
    categories: list[str] = []
    tags: list[str] = []
    keyword_rules: dict[str, str] = {}
    if not os.path.exists(path):
        return categories, tags, keyword_rules

    section = None
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
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
    return categories, tags, keyword_rules


def _parse_sectioned_txt(path: str, section_name: str) -> list[str]:
    items: list[str] = []
    if not os.path.exists(path):
        return items
    section = None
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].upper()
                continue
            if section == section_name:
                items.append(line)
    return items


def _parse_harm_keywords_txt(path: str) -> tuple[list[str], list[str]]:
    harm = [line.lower() for line in _parse_sectioned_txt(path, "HARM_KEYWORDS")]
    platform = [line.lower() for line in _parse_sectioned_txt(path, "PLATFORM_KEYWORDS")]
    return harm, platform


def migrate() -> dict:
    txt_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    categories_file = os.path.join(txt_dir, "categories.txt")
    taxonomy_file = os.path.join(txt_dir, "discovery_taxonomy.txt")
    harm_file = os.path.join(txt_dir, "harm_keywords.txt")

    categories, tags, keyword_rules = _parse_categories_txt(categories_file)
    taxonomy_paths = _parse_sectioned_txt(taxonomy_file, "TAXONOMY")
    harm_keywords, platform_keywords = _parse_harm_keywords_txt(harm_file)

    inserted = {
        "categories": 0, "tags": 0, "keyword_rules": 0, "taxonomy_paths": 0,
        "harm_keywords": 0, "platform_keywords": 0,
    }
    db = SessionLocal()
    try:
        existing_cats = {row.name for row in db.query(CategoryLabel.name).all()}
        for c in categories:
            if c not in existing_cats:
                db.add(CategoryLabel(name=c))
                existing_cats.add(c)
                inserted["categories"] += 1

        existing_tags = {row.name for row in db.query(TagLabel.name).all()}
        for t in tags:
            if t not in existing_tags:
                db.add(TagLabel(name=t))
                existing_tags.add(t)
                inserted["tags"] += 1

        existing_patterns = {row.pattern for row in db.query(KeywordRule.pattern).all()}
        for pattern, tag in keyword_rules.items():
            if pattern not in existing_patterns:
                db.add(KeywordRule(pattern=pattern, tag=tag))
                existing_patterns.add(pattern)
                inserted["keyword_rules"] += 1

        existing_paths = {row.path for row in db.query(TaxonomyPath.path).all()}
        for p in taxonomy_paths:
            if p not in existing_paths:
                db.add(TaxonomyPath(path=p))
                existing_paths.add(p)
                inserted["taxonomy_paths"] += 1

        existing_harm = {(row.keyword, row.kind) for row in db.query(HarmKeyword.keyword, HarmKeyword.kind).all()}
        for kw in harm_keywords:
            if (kw, "harm") not in existing_harm:
                db.add(HarmKeyword(keyword=kw, kind="harm"))
                existing_harm.add((kw, "harm"))
                inserted["harm_keywords"] += 1
        for kw in platform_keywords:
            if (kw, "platform") not in existing_harm:
                db.add(HarmKeyword(keyword=kw, kind="platform"))
                existing_harm.add((kw, "platform"))
                inserted["platform_keywords"] += 1

        db.commit()
    finally:
        db.close()

    return {
        "parsed": {
            "categories": len(categories),
            "tags": len(tags),
            "keyword_rules": len(keyword_rules),
            "taxonomy_paths": len(taxonomy_paths),
            "harm_keywords": len(harm_keywords),
            "platform_keywords": len(platform_keywords),
        },
        "inserted": inserted,
    }


if __name__ == "__main__":
    init_db()
    print(json.dumps(migrate(), indent=2))
