import asyncio
import logging
import re

from hear.core.keyword_loader import harm_keyword_loader
from hear.services.llm import LLMServiceProvider
from hear.services.model_client import ModelClientRegistry

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


class ModerationService:
    async def moderate(self, text: str) -> dict:
        if not text or not text.strip():
            return {
                "flagged": False,
                "severity": SEVERITY_NONE,
                "intent": "safe",
                "reason": "",
                "flagged_categories": [],
                "blocked_words_found": [],
            }
        text_lower = text.lower()
        built_in_keywords = harm_keyword_loader.harm_keywords
        loop = asyncio.get_event_loop()
        built_in_hits = [kw for kw in built_in_keywords if self._contains_keyword(text_lower, kw)]
        if built_in_hits:
            return {
                "flagged": True,
                "severity": SEVERITY_CRITICAL,
                "intent": "harmful",
                "reason": f"Contains flagged harmful language: {', '.join(built_in_hits[:5])}",
                "flagged_categories": ["Threats / Violence"],
                "blocked_words_found": built_in_hits,
            }
        keyword_hits = []
        local_result = await loop.run_in_executor(None, self._classify_local, text)
        scores: dict[str, float] = local_result.get("scores", {})
        max_score: float = local_result.get("max_score", 0.0)
        if max_score < _SAFE_THRESHOLD:
            return {
                "flagged": False,
                "severity": SEVERITY_NONE,
                "intent": "safe",
                "reason": "",
                "flagged_categories": [],
                "blocked_words_found": keyword_hits,
            }
        if max_score < _HIGH_THRESHOLD:
            if LLMServiceProvider.get_llm_service().is_available:
                try:
                    result = await loop.run_in_executor(
                        None,
                        lambda: LLMServiceProvider.get_llm_service().moderate(
                            text,
                            detoxify_scores=scores,
                            harm_keywords=list(built_in_keywords),
                            is_borderline=True,
                        ),
                    )
                    return result
                except Exception as exc:
                    logger.warning(
                        "[MODERATION] Qwen failed on borderline (%.2f) (%s) — using NLI fallback",
                        max_score,
                        exc,
                    )
            intent_result = await loop.run_in_executor(None, self._classify_intent, text)
            severity = self._compute_severity(local_result, intent_result)
            flagged = severity in (SEVERITY_HIGH, SEVERITY_CRITICAL)
            intent = intent_result.get("intent", "safe")
            reason = self._build_reason(local_result, intent_result, keyword_hits, intent, severity)
            return {
                "flagged": flagged,
                "severity": severity,
                "intent": intent,
                "reason": reason,
                "flagged_categories": self._get_flagged_categories(local_result),
                "blocked_words_found": keyword_hits,
            }
        severity = self._score_to_severity(max_score, local_result)
        if LLMServiceProvider.get_llm_service().is_available:
            try:
                result = await loop.run_in_executor(
                    None,
                    lambda: LLMServiceProvider.get_llm_service().moderate(
                        text,
                        detoxify_scores=scores,
                        harm_keywords=list(built_in_keywords),
                        is_borderline=False,
                    ),
                )
                result["severity"] = severity
                result["flagged"] = True
                return result
            except Exception as exc:
                logger.warning(
                    "[MODERATION] Qwen failed on high-score (%.2f) (%s) — using toxic-bert result",
                    max_score,
                    exc,
                )
        return {
            "flagged": True,
            "severity": severity,
            "intent": "harmful",
            "reason": f"High toxicity detected (score {max_score:.2f})",
            "flagged_categories": self._get_flagged_categories(local_result),
            "blocked_words_found": keyword_hits,
        }

    def _classify_local(self, text: str) -> dict:
        results = ModelClientRegistry.get_model_client().moderate_sync(text[:512])
        scores = {}
        if isinstance(results, dict):
            labels = results.get("labels", [])
            scores_list = results.get("scores", [])
            for label, score in zip(labels, scores_list):
                scores[label.lower()] = round(score, 4)
        elif isinstance(results, list):
            if results and isinstance(results[0], list):
                results = results[0]
            for item in results:
                if isinstance(item, dict):
                    label = item.get("label", "").lower()
                    scores[label] = round(item.get("score", 0), 4)
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
        }

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
            output = ModelClientRegistry.get_model_client().nli_sync(text[:1024], INTENT_LABELS)
            label_scores = dict(zip(output["labels"], output["scores"]))
            harmful_score = max(
                (label_scores.get(lbl, 0) for lbl in HARMFUL_INTENT_LABELS), default=0
            )
            safe_score = label_scores.get("safe, harmless content", 0)
            if harmful_score >= 0.55:
                top_harmful = max(HARMFUL_INTENT_LABELS, key=lambda l: label_scores.get(l, 0))
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
        for label, score in local_result.get("high_scores", {}).items():
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
