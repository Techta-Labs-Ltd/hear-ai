from hear.core.discovery_taxonomy import DiscoveryTaxonomyLoader
from hear.core.content_context import (
    assistive_tech_narrative,
    filter_controlled_taxonomy_paths,
    filter_freeform_tag_labels,
    tech_history_narrative,
)
from hear.services.categorization.service import CategorizationService


MINIDISC_SNIPPET = (
    "Many of you will have used Minidisc back in the 1990s. "
    "This is a documentary all about Minidisc. Sony and Philips format war. "
    "The story of Minidisc and the iPod."
)


def test_technology_topic_does_not_match_assistive_taxonomy_path():
    loader = DiscoveryTaxonomyLoader()
    loader.load("data/discovery_taxonomy.txt")
    paths = loader.match_paths_for_topics(["Technology", "Main topic Technology"])
    assert not any("Smart glasses" in p for p in paths)
    assert loader.canonicalize_path("Technology") == "Technology"


def test_minidisc_story_filters_wrong_tags():
    assert tech_history_narrative(MINIDISC_SNIPPET)
    assert not assistive_tech_narrative(MINIDISC_SNIPPET)
    free = filter_freeform_tag_labels(
        MINIDISC_SNIPPET,
        ["wildlife", "photography", "accessibility", "History", "music"],
    )
    assert "wildlife" not in free
    assert "photography" not in free
    assert "accessibility" not in free
    assert "History" in free
    controlled = filter_controlled_taxonomy_paths(
        MINIDISC_SNIPPET,
        [
            "Accessibility > Visual impairment > Assistive technology > Smart glasses",
            "Product experience > Informal review",
        ],
    )
    assert controlled == ["Product experience > Informal review"]


def test_categorizer_strips_wildlife_from_minidisc_editorial():
    svc = CategorizationService()
    tags, cats = svc._apply_editorial_rules(
        MINIDISC_SNIPPET,
        ["#wildlife", "#photography", "#accessibility", "#music"],
        ["Wildlife", "Accessibility", "Technology"],
        8,
    )
    assert "Wildlife" not in cats
    assert "#wildlife" not in tags
    assert "#accessibility" not in tags
    assert "Technology" in cats


TREE_PLANTING_NEWS = (
    "More than ten thousand trees were planted across North Northamptonshire. "
    "The council thanked schools, volunteers, community groups, and local businesses "
    "for supporting the annual tree planting programme."
)


def test_tree_planting_news_rejects_unrelated_assistive_labels():
    svc = CategorizationService()
    tags, categories = svc._apply_editorial_rules(
        TREE_PLANTING_NEWS,
        ["#guidedogs", "#community", "#communitydevelopment"],
        ["Personal lived experience", "Environment", "News"],
        5,
    )
    assert "#guidedogs" not in tags
    assert "Personal lived experience" not in categories
    assert tags == ["#community"]
    assert categories == ["Environment", "News"]


def test_editorial_rules_do_not_inject_hardcoded_subject_labels():
    svc = CategorizationService()
    tags, categories = svc._apply_editorial_rules(
        TREE_PLANTING_NEWS, [], [], 5
    )
    assert tags == []
    assert categories == []


def test_long_spoken_search_tag_requires_word_separators():
    svc = CategorizationService()
    assert svc._normalize_tag("communitydevelopment") == ""
    assert svc._normalize_tag("north northamptonshire") == "#north-northamptonshire"
