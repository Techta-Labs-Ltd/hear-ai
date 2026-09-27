import pytest

from hear.services.moderation.service import ModerationService


class Models:
    @staticmethod
    def moderate_sync(_text):
        return {"labels": ["toxic", "threat"], "scores": [0.04, 0.02]}

    @staticmethod
    def nli_sync(_text, _labels):
        return {"labels": [], "scores": []}


@pytest.mark.anyio
async def test_non_harmful_content_is_not_flagged():
    result = await ModerationService(Models(), harm_keywords=()).moderate(
        "We will shoot the music video tomorrow"
    )
    assert result["flagged"] is False
    assert result["intent"] == "safe"
    assert result["blocked_words_found"] == []
