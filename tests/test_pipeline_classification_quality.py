import pytest

from hear.services.categorization.discovery import DiscoveryService
from hear.services.categorization.service import CategorizationService
from hear.services.pipeline.configuration import PipelineCategoryCatalog, PipelineConfiguration


class PoliticalModels:
    def nli_sync(self, text, labels, *, hypothesis_template, multi_label):
        assert multi_label is True
        assert all(label == label.lower() and not label.startswith("#") for label in labels)
        assert len(labels) <= 512
        return {
            "labels": labels,
            "scores": [0.9 if label == "politics" else 0.01 for label in labels],
        }

    def sentiment_sync(self, text):
        return {"label": "positive", "labels": ["positive"], "scores": [0.8]}


@pytest.mark.anyio
async def test_large_polluted_catalog_does_not_force_first_category_or_echo_all_scores():
    configuration = PipelineConfiguration(
        version=1,
        categories=("Audio-Described", "Politics", "Community Local Community Local Council"),
        tags=tuple(f"#council-tag-{index}" for index in range(24000)),
        keyword_rules=(),
        harm_keywords=(),
        taxonomy_paths=(),
    )
    service = CategorizationService(
        PoliticalModels(), categories=PipelineCategoryCatalog(configuration)
    )
    result = await service.categorize("The council elected its leaders and debated their policies.")
    assert result["categories"] == ["Politics"]
    assert result["tags"] == ["#politics"]
    assert result["sentiment"] == "positive"
    assert len(result["confidence_scores"]) <= 2
    assert "Audio-Described" not in result["confidence_scores"]


def test_no_supported_category_is_better_than_an_arbitrary_low_confidence_label():
    result = CategorizationService()._merge(
        {"scores": {}},
        {"scores": {"Audio-Described": 0.0001, "Politics": 0.0002}},
        {"scores": {}},
        [],
        ["Audio-Described", "Politics"],
        5,
    )
    assert result["categories"] == []


def test_category_ranking_keeps_highest_confidence_first():
    service = CategorizationService()
    assert service._finalize_categories(
        "The council debates political policy.",
        [],
        {"Politics": 0.9, "News": 0.5},
        max_categories=2,
    ) == ["Politics", "News"]


def test_category_output_deduplicates_same_subject_and_catalog_compounds():
    service = CategorizationService()
    scores = {"Political": 0.9, "Politics": 0.8, "Community Local Politics": 0.7}
    assert service._finalize_categories("Local political policy", list(scores), scores) == [
        "Politics"
    ]


def test_tags_describe_subjects_instead_of_incidental_verbs():
    service = CategorizationService()
    assert service._subject_tags(
        "Havering councillors challenge and question political leadership.",
        ["#challenge", "#question", "#political", "#havering"],
        ["Politics"],
        {"#challenge": 0.8, "#question": 0.8, "#political": 0.8, "#havering": 0.7},
    ) == ["#political", "#havering"]


def test_later_subjects_are_analyzed_without_promoting_a_single_passing_mention():
    calls = []

    class Models:
        def nli_sync(self, text, labels, **kwargs):
            calls.append(text)
            return {"labels": labels, "scores": [0.9 if "election" in text else 0.05]}

    service = CategorizationService(Models())
    transcript = "Opening context " * 150 + ". The local election elected a new council."
    output = service._zero_shot_labels(transcript, ["Politics"])
    assert any("election" in text for text in calls)
    assert len(calls) <= 6
    assert output["scores"]["Politics"] < 0.35


def test_discovery_without_llm_describes_the_recording_instead_of_repeating_category():
    transcript = (
        "Havering Residents Association has elected new council leaders. "
        "Councillor Gillian Ford will scrutinise the new administration."
    )
    profile = DiscoveryService()._fallback_from_categorization(
        transcript,
        {"categories": ["Politics"], "tags": ["#politics"]},
        content_id="track",
        track_name="Bad Quality Tracks 0406 - Track 2.mp3",
    )
    assert profile is not None
    assert "Havering" in profile.title_suggestion
    assert "Gillian Ford" in profile.summary_short
    assert "conversation about Politics" not in profile.summary_short
    assert profile.speaker is None


def test_related_tags_collapse_to_the_strongest_one():
    from hear.services.categorization.service import CategorizationService

    tags = [
        "#political",
        "#political-role",
        "#leadership",
        "#political-leaders",
        "#council-leadership",
    ]
    scores = {
        "#political": 0.80,
        "#political-role": 0.79,
        "#leadership": 0.70,
        "#political-leaders": 0.66,
        "#council-leadership": 0.64,
    }
    assert CategorizationService._prune_related_tags(tags, scores) == ["#political", "#leadership"]
