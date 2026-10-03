import pytest

SmallModelsEngine = pytest.importorskip("hear.inference.small_models", exc_type=ImportError).SmallModelsEngine


def test_sentiment_response_is_readable_by_categorizer():
    engine = SmallModelsEngine.__new__(SmallModelsEngine)
    engine._sentiment = lambda *args, **kwargs: [{"label": "negative", "score": 0.9}]
    result = engine._infer("sentiment", "This is disappointing.", None, None)
    assert result["label"] == "negative"
    assert result["labels"] == ["negative"]


def test_toxicity_returns_all_labels_for_severity_decision():
    def classify(text, *, truncation, top_k):
        assert top_k is None
        return [{"label": "insult", "score": 0.95}, {"label": "threat", "score": 0.9}]

    engine = SmallModelsEngine.__new__(SmallModelsEngine)
    engine._toxic = classify
    result = engine._infer("toxic_bert", "A direct threat.", None, None)
    assert result["labels"] == ["insult", "threat"]
    assert result["scores"] == [0.95, 0.9]
