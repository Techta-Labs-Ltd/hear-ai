import pytest

from hear.services.moderation.service import ModerationService


class Models:
    @staticmethod
    def moderate_batch_sync(texts):
        return [{"labels": ["toxic", "threat"], "scores": [0.04, 0.02]} for _ in texts]

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


class WindowedModels:
    """toxic-bert stand-in: a threat score only for sentences that contain the marker."""

    def __init__(self, marker="MARKER"):
        self.marker = marker
        self.windows = []

    def moderate_batch_sync(self, texts):
        self.windows.extend(texts)
        return [
            {"labels": ["toxic", "threat"], "scores": [0.02, 0.97 if self.marker in text else 0.01]}
            for text in texts
        ]

    @staticmethod
    def nli_sync(_text, _labels):
        return {"labels": [], "scores": []}


class Llm:
    def __init__(self, verdict, available=True):
        self.verdict = verdict
        self.is_available = available
        self.calls = []

    def moderate(self, transcript, **kwargs):
        self.calls.append(kwargs)
        return dict(self.verdict)


SAFE_VERDICT = {
    "flagged": False,
    "severity": "none",
    "intent": "safe",
    "reason": "",
    "flagged_categories": [],
    "blocked_words_found": [],
}
FILLER = "The council met on Tuesday to discuss the new library opening hours. " * 80
SCAM_NEWS = (
    "Here is a warning from the police. Police warn of a phone scam targeting pensioners. "
    "Callers pretend to be from the bank. Never share your PIN."
)


@pytest.mark.anyio
async def test_keyword_hit_in_news_is_judged_in_context_and_cleared():
    llm = Llm(SAFE_VERDICT)
    result = await ModerationService(
        WindowedModels(), llm, harm_keywords=("spam", "scam", "fraud")
    ).moderate(FILLER + SCAM_NEWS + FILLER)
    assert result["flagged"] is False
    assert result["blocked_words_found"] == []
    assert result["keywords_reviewed"] == ["scam"]
    [call] = llm.calls
    assert call["keyword_hits"] == ["scam"]
    # The LLM sees the sentences around the hit, from the middle of a long transcript.
    assert any(
        "Police warn of a phone scam" in passage and "Never share your PIN." in passage
        for passage in call["passages"]
    )


@pytest.mark.anyio
async def test_keyword_hit_the_llm_confirms_is_labelled_blocked_keyword():
    llm = Llm(
        {
            "flagged": True,
            "severity": "high",
            "intent": "harmful",
            "reason": "Promotes a fake investment.",
            "flagged_categories": ["Fraud"],
        }
    )
    result = await ModerationService(WindowedModels(), llm, harm_keywords=("scam",)).moderate(
        "Send me your savings, this is no scam, you will double it."
    )
    assert result["flagged"] is True
    assert result["flagged_categories"] == ["Blocked keyword", "Fraud"]
    assert result["reason"].startswith("Blocked keyword: scam.")
    assert result["blocked_words_found"] == ["scam"]
    assert "Threats / Violence" not in result["flagged_categories"]


@pytest.mark.anyio
async def test_keyword_hit_without_the_llm_goes_to_review_with_its_context():
    result = await ModerationService(
        WindowedModels(), Llm(SAFE_VERDICT, available=False), harm_keywords=("scam",)
    ).moderate(SCAM_NEWS)
    assert result["flagged"] is True
    assert result["severity"] == "medium"
    assert result["flagged_categories"] == ["Blocked keyword"]
    assert result["reason"].startswith("Blocked keyword: scam.")
    assert "Police warn of a phone scam targeting pensioners." in result["reason"]
    assert result["blocked_words_found"] == ["scam"]


@pytest.mark.anyio
async def test_the_whole_transcript_is_scored_not_just_the_opening():
    models = WindowedModels()
    result = await ModerationService(models, harm_keywords=()).moderate(
        FILLER + "This is the MARKER sentence. " + FILLER
    )
    assert result["flagged"] is True
    assert len(models.windows) > 100
    assert "This is the MARKER sentence." in models.windows


@pytest.mark.anyio
async def test_llm_verdict_stands_even_when_toxicity_is_high():
    llm = Llm(SAFE_VERDICT)
    result = await ModerationService(WindowedModels(), llm, harm_keywords=()).moderate(
        FILLER + "The court heard how the MARKER attack unfolded. " + FILLER
    )
    assert result["flagged"] is False
    [call] = llm.calls
    assert call["is_borderline"] is False
    # The LLM sees the toxic sentence with two sentences either side, not a fixed window.
    [passage] = call["passages"]
    assert passage.count("The council met") == 4 and "MARKER attack" in passage


def test_long_transcripts_reach_the_llm_as_opening_plus_passages():
    from hear.services.llm import LLMService

    body = LLMService._moderation_body(FILLER * 3, ["Police warn of a phone scam."])
    assert body.startswith("Transcript opening")
    assert "[1] Police warn of a phone scam." in body
    assert len(body) < 1500
    assert LLMService._moderation_body("Short.", ["x"]) == "Transcript:\nShort."
