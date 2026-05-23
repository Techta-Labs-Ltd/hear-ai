import pytest

from app.core.discovery_taxonomy import DiscoveryTaxonomyLoader
from app.models.discovery import (
    ContentDiscoveryProfile,
    DiscoveryEntities,
    content_description_from_discovery,
    coerce_discovery_source,
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


def test_coerce_discovery_source_ignores_categorization_dict():
    cat = {"categories": ["Music"], "tags": ["#rock"]}
    assert coerce_discovery_source(cat) == ""
    profile = ContentDiscoveryProfile(content_id="x")
    data = discovery_to_callback_dict(profile, source=cat)
    assert data["source"] == ""
    assert data["speaker"] == "Alex"
    assert data["key_themes"] == ["Independence"]
    assert data["themes"] == ["Independence"]
    assert data["audience_relevance"] == ["Blind listeners"]
    assert data["audience_groups"] == ["Blind listeners"]
    assert data["entities"]["people"] == ["Alex"]
    assert "Rocco" in data["entities"]["animals"]
    assert "Alex" in data["entities_flat"]
    assert data["confidence"]["main_topic"] == 0.92
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


def test_extract_json_nested_object():
    from app.services.llm_service import LLMService

    raw = (
        'Here is the result: {"title_suggestion":"Guide dogs story","speaker":"Paul",'
        '"entities":{"people":["Paul","Annie"],"animals":["Rocco"]},"search_phrases":["a","b"]} end'
    )
    parsed = LLMService._extract_json(raw)
    assert parsed["speaker"] == "Paul"
    assert parsed["entities"]["animals"] == ["Rocco"]
    assert len(parsed["search_phrases"]) == 2


def test_infer_speaker_and_weak_profile():
    svc = DiscoveryService()
    tx = "Hello everyone, this is Denise Wallace in Glasgow. I'm really pleased..."
    assert svc._infer_speaker(tx) == "Denise Wallace"
    profile = ContentDiscoveryProfile(
        title_suggestion="20260514092830_372d5b77",
        summary_short=tx[:400],
        one_line_description=tx[:200],
        main_topic="Gaming",
        search_phrases=[],
        key_themes=[],
    )
    assert svc._is_weak_profile(profile, tx, "20260514092830_372d5b77")


def test_enrich_profile_fills_search_phrases():
    svc = DiscoveryService()
    profile = ContentDiscoveryProfile(
        main_topic="Gaming",
        secondary_topics=["Truman Adventure Games", "interviews"],
        key_themes=[],
        search_phrases=[],
    )
    enriched = svc._enrich_profile(
        profile,
        "Hello everyone, this is Denise Wallace in Glasgow.",
        {"tags": ["#gaming"], "categories": ["Entertainment"]},
        "20260514092830_372d5b77",
    )
    assert enriched.speaker == "Denise Wallace"
    assert len(enriched.search_phrases) >= 2
    assert not enriched.title_suggestion.startswith("20260514")


def test_context_category_shortlist_ranks_music_from_transcript():
    from app.services.categorizer import CategorizationService

    svc = CategorizationService()
    tx = (
        "We discuss a remix of Joan of Arc by Orchestral Manoeuvres in the Dark, "
        "the song lyrics, and how the melody feels."
    )
    zs = {"Music": 0.82, "Podcast": 0.41, "Entertainment": 0.55, "Gaming": 0.1}
    kw = {"#music": 0.7, "#podcast": 0.3}
    shortlist = svc._build_context_category_shortlist(
        tx, list(zs.keys()) + ["News", "Sports"], kw, zs
    )
    assert shortlist[0] == "Music"
    assert "Podcast" not in shortlist[:2]


def test_finalize_categories_uses_nli_when_llm_empty():
    from app.services.categorizer import CategorizationService

    svc = CategorizationService()
    tx = "A remix of Joan of Arc by Orchestral Manoeuvres in the Dark."
    zs = {"Music": 0.8, "Podcast": 0.4, "Entertainment": 0.5}
    cats = svc._finalize_categories(tx, [], zs, max_categories=2)
    assert "Music" in cats
    assert "Podcast" not in cats


def test_categorizer_prefers_music_over_podcast():
    from app.services.categorizer import CategorizationService

    svc = CategorizationService()
    tx = (
        "This episode is a heartfelt discussion about a remix of Joan of Arc by "
        "Orchestral Manoeuvres in the Dark. We talk about love, loss, and how the song feels."
    )
    tags, cats = svc._rebalance_subject_over_format(
        tx,
        ["#podcast", "#gaming"],
        ["Podcast", "Entertainment"],
    )
    assert "Music" in cats
    assert "#music" in tags
    assert "Podcast" not in cats
    assert "#podcast" not in tags


def test_sanitize_speaker_rejects_taxonomy_device_terms():
    svc = DiscoveryService()
    tx = "I was recording your production for me."
    assert svc._sanitize_speaker("Smart glasses", tx) is None
    assert svc._sanitize_speaker("Hardware", tx) is None
    assert svc._infer_speaker(tx) is None


def test_sanitize_speaker_keeps_introduced_name():
    svc = DiscoveryService()
    tx = "Hello everyone, this is Denise Wallace in Glasgow."
    assert svc._sanitize_speaker("Denise Wallace", tx) == "Denise Wallace"
    assert svc._sanitize_speaker("Denise Wallace", tx, trusted=True) == "Denise Wallace"


def test_enrich_profile_clears_hallucinated_llm_speaker():
    svc = DiscoveryService()
    profile = ContentDiscoveryProfile(
        speaker="Smart glasses",
        main_topic="Recording feedback",
        entities=DiscoveryEntities(people=["Smart glasses", "Hardware"]),
        summary_short="User feedback on recording.",
        one_line_description="Feedback on production recording.",
        key_themes=["Recording quality"],
        audience_relevance=["Producers"],
        search_phrases=["recording quality", "production feedback", "hardware review"],
    )
    enriched = svc._enrich_profile(
        profile,
        "I was recording your production for me.",
        {"categories": ["Technology"], "tags": ["#Hardware"]},
        "track.wav",
    )
    assert enriched.speaker is None
    assert "Smart glasses" not in enriched.entities.people


def test_taxonomy_label_terms_includes_leaf_segments(taxonomy_loader):
    terms = taxonomy_loader.taxonomy_label_terms()
    assert "guide dogs" in terms
    assert "ai and smart glasses" in terms
    assert "accessibility > visual impairment" in terms


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
