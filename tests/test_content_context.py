from app.core.discovery_taxonomy import DiscoveryTaxonomyLoader
from app.core.content_context import (
    assistive_tech_narrative,
    filter_controlled_taxonomy_paths,
    filter_freeform_tag_labels,
    tech_history_narrative,
)
from app.services.categorizer import CategorizationService


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
