import asyncio
import logging
import re

from hear.inference.client import LocalInferenceClient
from hear.services.llm import LLMService

logger = logging.getLogger(__name__)
SEVERITY_NONE = "none"
SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"
SEVERITY_CRITICAL = "critical"
TOXIC_CATEGORIES = {
    "toxic": "General toxicity",
    "severe_toxic": "Severe toxicity",
    "obscene": "Obscene language",
    "threat": "Threats of violence",
    "insult": "Personal attacks or insults",
    "identity_hate": "Identity-based hate speech",
}
INTENT_LABELS = [
    "safe, harmless content",
    "direct personal threat or call to violence",
    "hate speech or discrimination against a group",
    "explicit sexual content",
]
HARMFUL_INTENT_LABELS = {
    "direct personal threat or call to violence",
    "hate speech or discrimination against a group",
    "explicit sexual content",
}
_SAFE_THRESHOLD = 0.3
_HIGH_THRESHOLD = 0.8
BLOCKED_KEYWORD_CATEGORY = "Blocked keyword"
# toxic-bert scores each sentence on its own: packed with neutral text, a threat that
# scores 0.88 alone dropped to 0.38. Unpunctuated runs are cut to this length.
_MAX_SENTENCE_CHARS = 300
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
# Sentences either side of a keyword hit or toxic sentence that the LLM sees, and its budget.
_CONTEXT_SENTENCES = 2
_MAX_TOXIC_PASSAGES = 4
_MAX_PASSAGE_CHARS = 6000


class ModerationService:
    def __init__(
        self,
        model_client: LocalInferenceClient,
        llm: LLMService | None = None,
        harm_keywords=None,
    ) -> None:
        self._model_client = model_client
        self._llm = llm or LLMService(enabled=False)
        self._harm_keywords = tuple(
            getattr(harm_keywords, "harm_keywords", harm_keywords) or ()
        )

    def _models(self) -> LocalInferenceClient:
        return self._model_client

    def _llm_service(self) -> LLMService:
        return self._llm

    async def moderate(self, text: str) -> dict:
        if not text or not text.strip():
            return self._safe()
        loop = asyncio.get_event_loop()
        text_lower = text.lower()
        keyword_hits = [kw for kw in self._harm_keywords if self._contains_keyword(text_lower, kw)]
        sentences = self._sentences(text)
        local_result = await loop.run_in_executor(None, self._classify_local, sentences)
        scores: dict[str, float] = local_result.get("scores", {})
        max_score: float = local_result.get("max_score", 0.0)
        if not keyword_hits and max_score < _SAFE_THRESHOLD:
            return self._safe()
        # A watch-list keyword is a reason to look closer, not a verdict: "police warn of
        # a phone scam" is news. The LLM judges the passages around each hit.
        passages = self._passages(sentences, keyword_hits, local_result)
        if self._llm_service().is_available:
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: self._llm_service().moderate(
                        text,
                        detoxify_scores=scores,
                        keyword_hits=keyword_hits,
                        passages=passages,
                        is_borderline=max_score < _HIGH_THRESHOLD,
                    ),
                )
                return self._with_keywords(result, keyword_hits)
            except Exception as exc:
                logger.warning(
                    "[MODERATION] Qwen review failed (score %.2f, keywords %s) (%s) — using local fallback",
                    max_score,
                    keyword_hits,
                    exc,
                )
        if keyword_hits:
            return self._unverified_keywords(keyword_hits, passages, local_result)
        if max_score < _HIGH_THRESHOLD:
            intent_result = await loop.run_in_executor(
                None, self._classify_intent, " ".join(passages) or text
            )
            severity = self._compute_severity(local_result, intent_result)
            flagged = severity in (SEVERITY_HIGH, SEVERITY_CRITICAL)
            intent = intent_result.get("intent", "safe")
            reason = self._build_reason(local_result, intent_result, [], intent, severity)
            return {
                "flagged": flagged,
                "severity": severity,
                "intent": intent,
                "reason": reason,
                "flagged_categories": self._get_flagged_categories(local_result),
                "blocked_words_found": [],
            }
        return {
            "flagged": True,
            "severity": self._score_to_severity(max_score, local_result),
            "intent": "harmful",
            "reason": f"High toxicity detected (score {max_score:.2f})",
            "flagged_categories": self._get_flagged_categories(local_result),
            "blocked_words_found": [],
        }

    @staticmethod
    def _safe() -> dict:
        return {
            "flagged": False,
            "severity": SEVERITY_NONE,
            "intent": "safe",
            "reason": "",
            "flagged_categories": [],
            "blocked_words_found": [],
        }

    @staticmethod
    def _with_keywords(result: dict, keyword_hits: list[str]) -> dict:
        """Name the keywords on an LLM verdict; they count as found only if it flags."""
        result = dict(result)
        if keyword_hits and result.get("flagged"):
            categories = [
                c for c in result.get("flagged_categories") or [] if c != BLOCKED_KEYWORD_CATEGORY
            ]
            result["flagged_categories"] = [BLOCKED_KEYWORD_CATEGORY, *categories]
            reason = str(result.get("reason") or "").strip()
            result["reason"] = f"Blocked keyword: {', '.join(keyword_hits[:5])}." + (
                f" {reason}" if reason else ""
            )
            result["blocked_words_found"] = list(keyword_hits)
        else:
            result["blocked_words_found"] = []
        result["keywords_reviewed"] = list(keyword_hits)
        return result

    def _unverified_keywords(
        self, keyword_hits: list[str], passages: list[str], local_result: dict
    ) -> dict:
        """Without the LLM a keyword hit cannot be cleared, so it goes to human review."""
        max_score = local_result.get("max_score", 0.0)
        severity = (
            self._score_to_severity(max_score, local_result)
            if max_score >= _HIGH_THRESHOLD
            else SEVERITY_MEDIUM
        )
        reason = (
            f"Blocked keyword: {', '.join(keyword_hits[:5])}. "
            "Context not checked: the language model is unavailable."
        )
        excerpt = self._excerpt(passages, keyword_hits)
        if excerpt:
            reason += f' Context: "{excerpt}"'
        return {
            "flagged": True,
            "severity": severity,
            "intent": "questionable",
            "reason": reason,
            "flagged_categories": [
                BLOCKED_KEYWORD_CATEGORY,
                *self._get_flagged_categories(local_result),
            ],
            "blocked_words_found": list(keyword_hits),
            "keywords_reviewed": list(keyword_hits),
        }

    @staticmethod
    def _excerpt(passages: list[str], keyword_hits: list[str], radius: int = 100) -> str:
        """About two hundred characters of context centred on the first keyword hit."""
        for passage in passages:
            lowered = passage.lower()
            found = [lowered.find(kw.lower()) for kw in keyword_hits if kw.lower() in lowered]
            if found:
                start = max(0, min(found) - radius)
                return passage[start : min(found) + radius].strip()
        return ""

    @staticmethod
    def _sentences(text: str) -> list[str]:
        """Sentences, with unpunctuated runs cut at word boundaries to bounded pieces."""
        pieces: list[str] = []
        for sentence in _SENTENCE_BOUNDARY.split(text.strip()):
            words = sentence.split()
            current: list[str] = []
            length = 0
            for word in words:
                if current and length + 1 + len(word) > _MAX_SENTENCE_CHARS:
                    pieces.append(" ".join(current))
                    current, length = [], 0
                current.append(word[:_MAX_SENTENCE_CHARS])
                length += len(current[-1]) + (1 if length else 0)
            if current:
                pieces.append(" ".join(current))
        return pieces

    def _passages(
        self, sentences: list[str], keyword_hits: list[str], local_result: dict
    ) -> list[str]:
        """What the LLM judges: the sentences around each keyword hit and toxic sentence."""
        centres = [
            index
            for index, sentence in enumerate(sentences)
            if any(self._contains_keyword(sentence.lower(), kw) for kw in keyword_hits)
        ]
        ranked = sorted(local_result.get("sentences", []), key=lambda item: item[0], reverse=True)
        centres.extend(
            index for score, index in ranked[:_MAX_TOXIC_PASSAGES] if score >= _SAFE_THRESHOLD
        )
        spans: list[list[int]] = []
        used = 0
        for centre in centres:
            low = max(0, centre - _CONTEXT_SENTENCES)
            high = min(len(sentences), centre + _CONTEXT_SENTENCES + 1)
            overlapping = next((s for s in spans if low <= s[1] and s[0] <= high), None)
            if overlapping is not None:
                overlapping[0], overlapping[1] = min(low, overlapping[0]), max(high, overlapping[1])
                continue
            length = sum(len(sentence) + 1 for sentence in sentences[low:high])
            if spans and used + length > _MAX_PASSAGE_CHARS:
                break
            spans.append([low, high])
            used += length
        return [
            " ".join(sentences[low:high])[:_MAX_PASSAGE_CHARS] for low, high in sorted(spans)
        ]

    def _classify_local(self, sentences: list[str]) -> dict:
        """Score every sentence, so neither its position nor the text around it hides it."""
        scores: dict[str, float] = {}
        ranked: list[tuple[float, int]] = []
        for index, result in enumerate(self._models().moderate_batch_sync(sentences)):
            sentence_scores = self._label_scores(result)
            for label, score in sentence_scores.items():
                scores[label] = max(scores.get(label, 0.0), score)
            ranked.append((max(sentence_scores.values(), default=0.0), index))
        high_scores = {k: v for k, v in scores.items() if v >= 0.5}
        flagged = any(
            (
                v >= 0.5
                for k, v in scores.items()
                if k in ("severe_toxic", "threat", "identity_hate")
            )
        )
        return {
            "flagged": flagged,
            "max_score": max(scores.values()) if scores else 0,
            "high_scores": high_scores,
            "scores": scores,
            "sentences": ranked,
        }

    @staticmethod
    def _label_scores(results) -> dict[str, float]:
        scores: dict[str, float] = {}
        if isinstance(results, dict):
            labels = results.get("labels", [])
            scores_list = results.get("scores", [])
            for label, score in zip(labels, scores_list, strict=False):
                scores[label.lower()] = round(score, 4)
        elif isinstance(results, list):
            if results and isinstance(results[0], list):
                results = results[0]
            for item in results:
                if isinstance(item, dict):
                    label = item.get("label", "").lower()
                    scores[label] = round(item.get("score", 0), 4)
        return scores

    def _score_to_severity(self, max_score: float, local_result: dict) -> str:
        local_flagged = local_result.get("flagged", False)
        if max_score >= 0.95 or (local_flagged and max_score >= 0.85):
            return SEVERITY_CRITICAL
        if max_score >= 0.8:
            return SEVERITY_HIGH
        if max_score >= 0.6:
            return SEVERITY_MEDIUM
        return SEVERITY_LOW

    def _classify_intent(self, text: str) -> dict:
        try:
            output = self._models().nli_sync(text[:1024], INTENT_LABELS)
            label_scores = dict(zip(output["labels"], output["scores"], strict=False))
            harmful_score = max(
                (label_scores.get(lbl, 0) for lbl in HARMFUL_INTENT_LABELS), default=0
            )
            safe_score = label_scores.get("safe, harmless content", 0)
            if harmful_score >= 0.55:
                top_harmful = max(HARMFUL_INTENT_LABELS, key=lambda label: label_scores.get(label, 0))
                return {
                    "intent": "harmful",
                    "reason": f"NLI classified as: {top_harmful} ({harmful_score:.2f})",
                    "scores": label_scores,
                }
            if harmful_score >= 0.35 and safe_score < 0.5:
                return {
                    "intent": "questionable",
                    "reason": f"Potentially harmful content ({harmful_score:.2f})",
                    "scores": label_scores,
                }
            return {"intent": "safe", "reason": "", "scores": label_scores}
        except Exception:
            return {"intent": "safe", "reason": "", "scores": {}}

    def _contains_keyword(self, text_lower: str, keyword: str) -> bool:
        kw = keyword.lower().strip()
        if not kw:
            return False
        pattern = f"(?<!\\w){re.escape(kw)}(?!\\w)"
        return re.search(pattern, text_lower) is not None

    def _compute_severity(self, local_result: dict, intent_result: dict) -> str:
        intent = intent_result.get("intent", "safe")
        max_toxic = local_result.get("max_score", 0)
        local_flagged = local_result.get("flagged", False)
        if intent == "harmful" and local_flagged:
            return SEVERITY_CRITICAL
        if intent == "harmful":
            return SEVERITY_HIGH
        if local_flagged and intent == "questionable":
            return SEVERITY_HIGH
        if local_flagged and intent == "safe":
            return SEVERITY_MEDIUM
        if intent == "questionable":
            return SEVERITY_MEDIUM
        if max_toxic > 0.6:
            return SEVERITY_LOW
        return SEVERITY_NONE

    def _get_flagged_categories(self, local_result: dict) -> list[str]:
        categories = []
        for label, _score in local_result.get("high_scores", {}).items():
            if label in TOXIC_CATEGORIES:
                categories.append(TOXIC_CATEGORIES[label])
        return categories

    def _build_reason(
        self,
        local_result: dict,
        intent_result: dict,
        keyword_hits: list[str],
        intent: str,
        severity: str,
    ) -> str:
        intent_reason = intent_result.get("reason", "")
        if intent_reason:
            return intent_reason
        parts = []
        high = local_result.get("high_scores", {})
        if high:
            labels = [TOXIC_CATEGORIES.get(k, k) for k in high]
            parts.append(f"Detected: {', '.join(labels)}")
        if keyword_hits:
            parts.append(f"Monitored words found: {', '.join(keyword_hits)}")
        if not parts:
            if severity == SEVERITY_NONE:
                return "Content appears safe"
            return "Content flagged by automated analysis"
        return ". ".join(parts)
