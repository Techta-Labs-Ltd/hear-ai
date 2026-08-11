import asyncio

import pytest

from hear.resolver.nlp.temporal import detect_temporal
from hear.resolver.resolution.resolvers import (
    resolve_category, resolve_creator, resolve_location, resolve_org, resolve_tag,
)
from hear.resolver.semantic.gpu import SemanticMatcher

INDEX = {
    "version": 1,
    "categories.json": [
        {"id": "football", "canonical": "football", "normalized": "football",
         "phrases": ["football", "soccer"], "synonyms": ["soccer"], "stems": {}},
        {"id": "sport", "canonical": "sport", "normalized": "sport",
         "phrases": ["sport"], "synonyms": [], "stems": {}},
    ],
    "creators.json": [
        {"name": "David Beard", "normalized": "david beard", "aliases": ["dave", "beard"], "entity_type": "creator"},
    ],
    "organisations.json": [
        {"name": "BBC News", "normalized": "bbc news", "aliases": ["bbc"], "entity_type": "org"},
    ],
    "tags.json": [
        {"name": "business", "normalized": "business", "slug": "business"},
    ],
    "locations.json": [
        {"city": "London", "country_code": "gb", "normalized": "london", "lat": "51.5", "lng": "-0.1", "post_code": "GB-LND"},
        {"city": "Manchester", "country_code": "gb", "normalized": "manchester", "lat": "53.4", "lng": "-2.2", "post_code": "GB-MAN"},
        {"city": "London", "country_code": "ca", "normalized": "london", "lat": "42.9", "lng": "-81.2", "post_code": "CA-ON"},
    ],
}


def test_category_exact():
    rec, conf, _ = resolve_category("football", INDEX)
    assert rec and rec["canonical"] == "football" and conf == 100.0


def test_category_synonym():
    rec, conf, _ = resolve_category("soccer", INDEX)
    assert rec and rec["canonical"] == "football"


def test_category_exact_beats_other_synonym():
    rec, _, _ = resolve_category("sport", INDEX)
    assert rec and rec["canonical"] == "sport"


def test_category_typo_fuzzy():
    rec, conf, _ = resolve_category("footbal", INDEX)
    assert rec and rec["canonical"] == "football" and conf >= 80


def test_category_garbage_rejected():
    rec, _, _ = resolve_category("zzqxplm", INDEX)
    assert rec is None


def test_creator_alias():
    rec, conf, _ = resolve_creator("dave", INDEX)
    assert rec and rec["name"] == "David Beard" and conf == 100.0


def test_org_alias():
    rec, _, _ = resolve_org("bbc", INDEX)
    assert rec and rec["name"] == "BBC News"


def test_tag_exact():
    rec, conf, _ = resolve_tag("business", INDEX)
    assert rec and rec["slug"] == "business" and conf == 100.0


def test_location_exact_with_coords():
    rec, conf, _ = resolve_location("london", INDEX, "gb")
    assert rec and rec["city"] == "London" and rec["country_code"] == "gb"
    assert rec["lat"] == "51.5" and rec["lng"] == "-0.1" and rec["post_code"] == "GB-LND"


def test_location_country_scoped():
    rec, _, _ = resolve_location("london", INDEX, "ca")
    assert rec and rec["country_code"] == "ca"


def test_location_typo():
    rec, conf, _ = resolve_location("manchestr", INDEX, "gb")
    assert rec and rec["city"] == "Manchester" and conf >= 85


def test_location_short_foreign_rejected():
    rec, _, _ = resolve_location("roma", INDEX, "gb")
    assert rec is None


@pytest.mark.parametrize("text,expected", [
    ("play the latest sport", "recency"),
    ("news today", "date"),
    ("news from yesterday", "date"),
    ("play jazz in london", None),
])
def test_temporal(text, expected):
    t = detect_temporal(text)
    assert (t or {}).get("type") == expected if expected else t is None


def test_tiers_disabled_are_noops():
    sm = SemanticMatcher()
    rec, score, cands = asyncio.run(sm.match("soccer", "category"))
    assert rec is None and score == 0.0 and cands == []
