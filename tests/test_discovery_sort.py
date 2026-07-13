from app.core.discovery_sort import sort_discovery_items
from app.models.discovery import discovery_to_callback_dict, ContentDiscoveryProfile


def test_discovery_callback_includes_latest_and_trending_fields():
    profile = ContentDiscoveryProfile(
        content_id="t1",
        title_suggestion="Title",
        summary_short="Summary",
        one_line_description="One line",
    )
    data = discovery_to_callback_dict(
        profile,
        published_at="2026-05-01T10:00:00Z",
        trending_score=42.5,
    )
    assert data["published_at"] == "2026-05-01T10:00:00Z"
    assert data["latest_at"] == "2026-05-01T10:00:00Z"
    assert data["trending_score"] == 42.5


def test_sort_latest_by_latest_at():
    items = [
        {"latest_at": "2026-05-01T10:00:00Z", "trending_score": 100},
        {"latest_at": "2026-05-20T10:00:00Z", "trending_score": 1},
        {"latest_at": "2026-05-10T10:00:00Z", "trending_score": 50},
    ]
    sorted_items = sort_discovery_items(items, "latest")
    assert [i["latest_at"] for i in sorted_items] == [
        "2026-05-20T10:00:00Z",
        "2026-05-10T10:00:00Z",
        "2026-05-01T10:00:00Z",
    ]


def test_sort_trending_by_score_then_latest():
    items = [
        {"latest_at": "2026-05-01T10:00:00Z", "trending_score": 10},
        {"latest_at": "2026-05-20T10:00:00Z", "trending_score": 50},
        {"latest_at": "2026-05-15T10:00:00Z", "trending_score": 50},
    ]
    sorted_items = sort_discovery_items(items, "trending")
    assert sorted_items[0]["trending_score"] == 50
    assert sorted_items[0]["latest_at"] == "2026-05-20T10:00:00Z"
    assert sorted_items[1]["latest_at"] == "2026-05-15T10:00:00Z"
    assert sorted_items[2]["trending_score"] == 10
