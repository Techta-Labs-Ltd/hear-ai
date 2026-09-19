from types import SimpleNamespace

import pytest

from hear.services.moderation import service as moderation_module
from hear.services.moderation.service import ModerationService


@pytest.mark.anyio
async def test_platform_keyword_without_harmful_context_is_not_flagged(monkeypatch):
    monkeypatch.setattr(moderation_module, "predict_harm", lambda _text: 0.05)
    monkeypatch.setattr(
        moderation_module,
        "harm_keyword_loader",
        SimpleNamespace(
            all_keywords=["shoot"],
            harm_keywords=[],
            sync_platform_keywords=lambda _values: None,
        ),
    )
    monkeypatch.setattr(
        moderation_module,
        "get_model_client",
        lambda: SimpleNamespace(
            moderate_sync=lambda _text: {
                "labels": ["toxic", "threat"],
                "scores": [0.04, 0.02],
            }
        ),
    )

    result = await ModerationService().moderate(
        "We will shoot the music video tomorrow",
        blocked_keywords=["shoot"],
    )

    assert result["flagged"] is False
    assert result["intent"] == "safe"
    assert result["blocked_words_found"] == ["shoot"]


