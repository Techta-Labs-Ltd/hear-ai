from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hear.services.transport import operations


@pytest.mark.anyio
async def test_platform_settings_updates_keywords_and_tags(monkeypatch):
    blocked = Mock()
    auto_tags = Mock()
    add_tag = Mock()
    monkeypatch.setattr(
        operations, "harm_keyword_loader", SimpleNamespace(sync_platform_keywords=blocked)
    )
    monkeypatch.setattr(operations, "auto_tag_keyword_loader", SimpleNamespace(sync=auto_tags))
    monkeypatch.setattr(operations, "category_loader", SimpleNamespace(add_tag=add_tag))
    session = Mock(side_effect=AssertionError("Unexpected database session"))
    monkeypatch.setattr(operations.DatabaseRuntime, "SessionLocal", session)
    service = object.__new__(operations.Operations)
    result = await service.update_platform_settings(" Spam, Scam, ", " News, Sports, ")
    blocked.assert_called_once_with(["spam", "scam"])
    auto_tags.assert_called_once_with(["news", "sports"])
    assert [call.args[0] for call in add_tag.call_args_list] == ["news", "sports"]
    session.assert_not_called()
    assert result == {
        "status": "accepted",
        "blocked_keywords_count": 2,
        "auto_tag_keywords_count": 2,
    }
