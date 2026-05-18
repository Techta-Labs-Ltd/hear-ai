import pytest

from app.core.discovery_taxonomy import DiscoveryTaxonomyLoader
from app.models.discovery import (
    ContentDiscoveryProfile,
    DiscoveryEntities,
    content_description_from_discovery,
    discovery_to_callback_dict,
    flatten_entities,
)
from app.services.discovery import DiscoveryService


@pytest.fixture
def taxonomy_loader(tmp_path):
    path = tmp_path / "taxonomy.txt"
    path.write_text(
        "[TAXONOMY]\n"
        "Accessibility > Visual impairment\n"
        "Accessibility > Guide dogs\n"
        "AI and smart glasses\n",
        encoding="utf-8",
    )
    loader = DiscoveryTaxonomyLoader()
    loader.load(str(path))
    return loader


def test_taxonomy_matches_topics(taxonomy_loader):
    paths = taxonomy_loader.match_paths_for_topics(
        ["guide dogs for blind users", "smart glasses demo"]
    )
    assert "Accessibility > Guide dogs" in paths
    assert "AI and smart glasses" in paths


def test_content_description_prefers_one_line():
    profile = ContentDiscoveryProfile(
        one_line_description="A walkthrough of assistive tech.",
        summary_short="Longer summary here.",
    )
    assert content_description_from_discovery(profile) == "A walkthrough of assistive tech."


def test_discovery_callback_matches_backend_schema():
    profile = ContentDiscoveryProfile(
        content_id="audio_000123",
        title_suggestion="Smart glasses walk",
        main_topic="Assistive technology",
        summary_short="Short blurb",
        summary_long="Longer text",
        one_line_description="One line",
        primary_genre="Personal lived experience",
        controlled_tags=["Accessibility > Guide dogs"],
        freeform_tags=["Wearables"],
        entities=DiscoveryEntities(people=["Alex"], animals=["Rocco"]),
        key_themes=["Independence"],
        audience_relevance=["Blind listeners"],
        search_phrases=["guide dog walking"],
        confidence={"main_topic": 0.92},
        speaker="Alex",
    )
    data = discovery_to_callback_dict(profile, duration_seconds=312, source="upload")
    assert data["id"] == "audio_000123"
    assert data["title"] == "Smart glasses walk"
    assert data["duration_seconds"] == 312
    assert data["source"] == "upload"
    assert data["speaker"] == "Alex"
    assert data["themes"] == ["Independence"]
    assert data["audience_groups"] == ["Blind listeners"]
    assert "Alex" in data["entities"] and "Rocco" in data["entities"]
    assert data["confidence_scores"]["main_topic"] == 0.92
    assert data["main_topic"] == "Assistive technology"
    assert data["one_line_description"] == "One line"


def test_flatten_entities():
    ents = DiscoveryEntities(people=["A"], products=["Meta Ray-Ban"])
    assert flatten_entities(ents) == ["A", "Meta Ray-Ban"]


def test_qwen_controlled_tags_preserved_and_enriched(taxonomy_loader, monkeypatch):
    svc = DiscoveryService()
    monkeypatch.setattr(
        "app.services.discovery.discovery_taxonomy_loader",
        taxonomy_loader,
    )
    profile = ContentDiscoveryProfile(
        main_topic="guide dogs",
        secondary_topics=["independence"],
    )
    cat = {"categories": ["Personal lived experience"], "tags": ["#Blindness"]}
    qwen_tags = ["Accessibility > Guide dogs", "Human connection"]
    tags = svc.merge_controlled_tags(profile, qwen_tags, cat)
    assert tags[0] == "Accessibility > Guide dogs"
    assert "Human connection" in tags
    assert "Personal lived experience" in tags


def test_canonicalize_path(taxonomy_loader):
    assert taxonomy_loader.canonicalize_path("accessibility > guide dogs") == "Accessibility > Guide dogs"


def test_fallback_profile_when_llm_disabled(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "DISCOVERY_METADATA_ENABLED", False)
    svc = DiscoveryService()
    profile = svc._fallback_from_categorization(
        "I use smart glasses every day for navigation.",
        {"categories": ["Technology"], "tags": ["#Wearables"]},
        content_id="t1",
        track_name="My track",
    )
    assert profile is not None
    assert profile.main_topic == "Technology"
    assert profile.content_id == "t1"
    assert profile.embedding_source_text
