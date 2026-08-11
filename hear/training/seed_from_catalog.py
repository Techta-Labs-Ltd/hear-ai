"""Generates weak/synthetic CategoryTrainingExample rows (source='catalog_seed')
from the DB-backed catalog (category_labels, tag_labels, keyword_rules), so the
Ray Train classifiers have a starting point instead of waiting for real usage
data to accumulate from zero. Run once by hand, and again whenever the catalog
gets new labels that aren't yet covered by real training data:
    uv run python -m hear.training.seed_from_catalog
"""
import json

from hear.models.database import (
    CategoryLabel,
    CategoryTrainingExample,
    KeywordRule,
    SessionLocal,
    TagLabel,
)


def seed() -> dict:
    db = SessionLocal()
    inserted = {"keyword_tag_examples": 0, "category_name_examples": 0, "tag_name_examples": 0}
    try:
        existing_seed_texts = {
            row.text for row in db.query(CategoryTrainingExample.text)
            .filter(CategoryTrainingExample.source == "catalog_seed").all()
        }

        # Each keyword rule is already a (text -> tag) weak-label pair once split on "|".
        tag_covered: set[str] = set()
        for rule in db.query(KeywordRule).all():
            for keyword in rule.pattern.split("|"):
                keyword = keyword.strip()
                if not keyword or keyword in existing_seed_texts:
                    continue
                db.add(CategoryTrainingExample(
                    source="catalog_seed",
                    event_type="seed_keyword",
                    text=keyword,
                    category=None,
                    tags=[rule.tag],
                    label=None,
                    raw_payload=None,
                ))
                existing_seed_texts.add(keyword)
                inserted["keyword_tag_examples"] += 1
            tag_covered.add(rule.tag)

        # Categories/tags with no keyword-rule coverage get a thin seed from their own
        # name, so the classifier's output space includes every catalog label.
        for row in db.query(CategoryLabel).all():
            if row.name in existing_seed_texts:
                continue
            db.add(CategoryTrainingExample(
                source="catalog_seed",
                event_type="seed_label",
                text=row.name,
                category=row.name,
                tags=None,
                label=None,
                raw_payload=None,
            ))
            existing_seed_texts.add(row.name)
            inserted["category_name_examples"] += 1

        for row in db.query(TagLabel).all():
            if row.name in tag_covered or row.name in existing_seed_texts:
                continue
            db.add(CategoryTrainingExample(
                source="catalog_seed",
                event_type="seed_label",
                text=row.name.lstrip("#"),
                category=None,
                tags=[row.name],
                label=None,
                raw_payload=None,
            ))
            existing_seed_texts.add(row.name)
            inserted["tag_name_examples"] += 1

        db.commit()
    finally:
        db.close()
    return inserted


if __name__ == "__main__":
    print(json.dumps(seed(), indent=2))
