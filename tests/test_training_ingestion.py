from types import SimpleNamespace

from hear.training import categorizer_train

import pytest

from hear.services.transport import operations as operations_module
from hear.services.transport.operations import Operations


class FakeSession:
    def __init__(self):
        self.added = []
        self.committed = False
        self.closed = False

    def add(self, value):
        self.added.append(value)

    def commit(self):
        self.committed = True
        for value in self.added:
            if getattr(value, "id", None) is None:
                value.id = "example-1"

    def close(self):
        self.closed = True
def test_ray_training_defers_before_minimum_dataset(monkeypatch):
    monkeypatch.setattr(
        categorizer_train,
        "load_training_examples",
        lambda target: (["sample"] * 49, ["label"] * 49),
    )

    result = categorizer_train.ray_train_categorizer("category")

    assert result == {
        "target": "category",
        "error": "only 49 examples (need >= 50)",
    }




@pytest.mark.anyio
async def test_platform_settings_grpc_update_feeds_trainable_labels(monkeypatch):
    session = FakeSession()
    blocked_sync = []
    auto_tag_sync = []
    catalog_tags = []

    scheduled = []
    monkeypatch.setattr(Operations, "_schedule_training", lambda self, values: scheduled.append(values))
    monkeypatch.setattr(operations_module, "SessionLocal", lambda: session)
    monkeypatch.setattr(
        operations_module,
        "harm_keyword_loader",
        SimpleNamespace(sync_platform_keywords=lambda values: blocked_sync.extend(values)),
    )
    monkeypatch.setattr(
        operations_module,
        "auto_tag_keyword_loader",
        SimpleNamespace(sync=lambda values: auto_tag_sync.extend(values)),
    )
    monkeypatch.setattr(
        operations_module,
        "category_loader",
        SimpleNamespace(add_tag=lambda value: catalog_tags.append(value)),
    )

    service = object.__new__(Operations)
    result = await service.update_platform_settings(
        "Spam, Fraud",
        "News, Breaking",
    )

    assert result == {
        "status": "accepted",
        "blocked_keywords_count": 2,
        "auto_tag_keywords_count": 2,
    }
    assert blocked_sync == ["spam", "fraud"]
    assert auto_tag_sync == ["news", "breaking"]
    assert catalog_tags == ["news", "breaking"]
    assert session.committed and session.closed

    auto_examples = [row for row in session.added if row.event_type == "auto_tag_keyword"]
    blocked_examples = [row for row in session.added if row.event_type == "blocked_keyword"]
    assert [row.tags for row in auto_examples] == [["#news"], ["#breaking"]]
    assert scheduled == [{"harm", "tags"}]
    assert [row.label for row in blocked_examples] == ["harmful", "harmful"]


@pytest.mark.anyio
async def test_category_event_grpc_ingestion_persists_training_example(monkeypatch):
    session = FakeSession()
    categories = []
    tags = []
    scheduled = []
    monkeypatch.setattr(Operations, "_schedule_training", lambda self, values: scheduled.append(values))

    monkeypatch.setattr(operations_module, "SessionLocal", lambda: session)
    monkeypatch.setattr(
        operations_module,
        "category_loader",
        SimpleNamespace(
            add_category=lambda value: categories.append(value),
            add_tag=lambda value: tags.append(value),
        ),
    )

    service = object.__new__(Operations)
    result = await service.ingest_category_event(
        {
            "event_type": "track_tagged",
            "text": "A report about renewable energy",
            "category": "Environment",
            "tags": ["#renewable-energy"],
            "label": "verified",
            "source_id": "track-1",
        }
    )

    assert result == {"status": "accepted", "example_id": "example-1"}
    assert categories == ["Environment"]
    assert tags == ["#renewable-energy"]
    row = session.added[0]
    assert row.source == "grpc"
    assert row.category == "Environment"
    assert row.tags == ["#renewable-energy"]
    assert scheduled == [{"category", "tags"}]
    assert row.raw_payload["source_id"] == "track-1"
