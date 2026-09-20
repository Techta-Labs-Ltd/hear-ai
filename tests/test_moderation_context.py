from types import SimpleNamespace

import pytest

from hear.services.moderation import service as moderation_module
from hear.services.moderation.service import ModerationService


@pytest.mark.anyio
async def test_non_harmful_content_is_not_flagged(monkeypatch):
    monkeypatch.setattr(
        moderation_module,
        "harm_keyword_loader",
        SimpleNamespace(harm_keywords=[]),
    )
    monkeypatch.setattr(
        moderation_module.ModelClientRegistry,
        "get_model_client",
        lambda: SimpleNamespace(
            moderate_sync=lambda _text: {"labels": ["toxic", "threat"], "scores": [0.04, 0.02]}
        ),
    )
    result = await ModerationService().moderate("We will shoot the music video tomorrow")
    assert result["flagged"] is False
    assert result["intent"] == "safe"
    assert result["blocked_words_found"] == []
