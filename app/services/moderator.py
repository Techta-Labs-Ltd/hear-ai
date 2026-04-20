import asyncio

import torch
from transformers import pipeline as hf_pipeline

from app.core.keyword_loader import harm_keyword_loader

MODERATION_MODEL = "unitary/toxic-bert"
INTENT_MODEL = "cross-encoder/nli-distilroberta-base"

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

HARM_KEYWORDS: list[str] = [
    "i will kill you", "i'm going to kill you", "i am going to kill you",
    "gonna kill you", "gon kill you", "imma kill you", "i'll kill you",
    "i will murder you", "i will end you", "i will hurt you",
    "i am going to beat", "i'm going to beat", "gonna beat her up", "gonna beat him up",
    "beat her up", "beat him up", "beat you up", "beat them up",
    "beat her badly", "beat him badly", "beat her very badly", "beat him very badly",
    "i'll beat you", "i will beat you", "imma beat you", "smash her face",
    "smash his face", "punch her", "punch him", "kick her head", "kick his head",
    "you're dead", "you are dead", "you're dead to me",
    "i'll shoot you", "i will shoot you", "i'll stab you", "i will stab you",
    "i'll blow your head off", "i'll put a bullet in you",
    "i want you dead", "i hope you die",
    "spin the block", "up the score", "catch a body", "caught a body",
    "drop a body", "wet him up", "wet her up", "splash him", "splash her",
    "poke him", "poke her", "shank him", "shank her", "shank them",
    "ride out on", "back out the strap", "back out the tool",
    "link man with the strap", "up the stick", "do him dirty",
    "score on sight", "put him in the dirt",
    "slide on", "leave him leaking", "leave her leaking", "leave man leaking",
    "stomp him out", "stomp her out", "rush him", "rush her",
    "spin his block", "spin her block", "skeng", "no lacking",
    "cap him", "clap him", "blick him", "smoke him", "smoke her", "smoke them",
    "catch him slipping", "catch her slipping", "run up on him", "run up on her",
    "squeeze the trigger", "pull the trigger on",
    "put him six feet", "put her six feet", "send him to god",
    "drop him", "drop her", "body bag him", "body bag her",
    "get smoked", "get merked", "get rocked", "get domed",
    "put hands on", "beat him down", "beat her down",
    "bust at", "bust shots at", "send shots", "let it bang",
    "bust a cap", "dump on", "chop him down", "chop her down",
    "air him out", "air her out", "hit the lick", "rob him", "rob her",
    "with my strap", "strapped up", "glizzy", "draco",
    "mac-10", "uzi", "ar-15", "banana clip", "extended clip", "drum mag",
    "glock with a switch", "auto switch",
    "kill all", "death to", "gas the", "exterminate the",
    "white power", "white supremacy", "nigger", "nigga die",
    "kike", "spic", "chink", "sand nigger", "raghead",
    "bomb the mosque", "bomb the church", "blow up the school",
    "jihad against", "holy war against",
    "child porn", "kiddie porn", "cp link", "rape a child",
    "molest a child", "underaged naked",
    "kill myself", "killing myself", "want to die", "going to end it",
    "slit my wrists", "hang myself", "suicide pact",
    "overdose on purpose", "want to overdose",
    "allahu akbar bomb", "suicide bomb", "blow myself up",
    "school shooting", "mass shooting", "shoot up the", "bomb threat",
    "pipe bomb", "nail bomb",
]


class ModerationService:
    def __init__(self):
        self._classifier = None
        self._intent = None
        self._device = 0 if torch.cuda.is_available() else -1

    def load(self):
        self._classifier = hf_pipeline(
            "text-classification",
            model=MODERATION_MODEL,
            device=self._device,
            top_k=None,
        )
        self._intent = hf_pipeline(
            "zero-shot-classification",
            model=INTENT_MODEL,
            device=self._device,
            multi_label=True,
        )

    @property
    def is_loaded(self) -> bool:
        return self._classifier is not None

    async def moderate(self, text: str, blocked_keywords: list[str] = None) -> dict:
        if not text or not text.strip():
            return {
                "flagged": False,
                "severity": SEVERITY_NONE,
                "intent": "safe",
                "reason": "",
                "flagged_categories": [],
                "blocked_words_found": [],
            }

        # Sync platform keywords from backend settings into the loader
        if blocked_keywords:
            harm_keyword_loader.sync_platform_keywords(blocked_keywords)

        text_lower = text.lower()
        all_keywords = harm_keyword_loader.all_keywords

        built_in_hits = [kw for kw in all_keywords if kw in text_lower]
        if built_in_hits:
            return {
                "flagged": True,
                "severity": SEVERITY_CRITICAL,
                "intent": "harmful",
                "reason": f"Contains flagged harmful language: {', '.join(built_in_hits[:5])}",
                "flagged_categories": ["Threats / Violence"],
                "blocked_words_found": built_in_hits,
            }

        keyword_hits = self._check_keywords(text, blocked_keywords or [])

        loop = asyncio.get_event_loop()
        local_result, intent_result = await asyncio.gather(
            loop.run_in_executor(None, self._classify_local, text),
            loop.run_in_executor(None, self._classify_intent, text),
        )

        severity = self._compute_severity(local_result, intent_result)
        flagged = severity in (SEVERITY_HIGH, SEVERITY_CRITICAL)
        intent = intent_result.get("intent", "safe")
        reason = self._build_reason(local_result, intent_result, keyword_hits, intent, severity)
        flagged_categories = self._get_flagged_categories(local_result)

        if flagged and intent == "harmful":
            nli_scores = intent_result.get("scores", {})
            harmful_score = max(
                (nli_scores.get(lbl, 0) for lbl in HARMFUL_INTENT_LABELS),
                default=0,
            )
            if harmful_score >= 0.70:
                self._learn_phrases(text)

        return {
            "flagged": flagged,
            "severity": severity,
            "intent": intent,
            "reason": reason,
            "flagged_categories": flagged_categories,
            "blocked_words_found": keyword_hits,
        }

    def _classify_local(self, text: str) -> dict:
        results = self._classifier(text[:512])
        if isinstance(results, list) and results and isinstance(results[0], list):
            results = results[0]

        scores = {}
        for item in results:
            label = item["label"].lower()
            scores[label] = round(item["score"], 4)

        high_scores = {k: v for k, v in scores.items() if v >= 0.5}
        flagged = any(
            v >= 0.5 for k, v in scores.items()
            if k in ("severe_toxic", "threat", "identity_hate")
        )

        return {
            "flagged": flagged,
            "max_score": max(scores.values()) if scores else 0,
            "high_scores": high_scores,
            "scores": scores,
        }

    def _learn_phrases(self, text: str):
        import re
        existing = set(harm_keyword_loader.all_keywords)
        sentences = re.split(r"[.!?\n]+", text.lower())
        for sentence in sentences:
            phrase = sentence.strip().strip("\"'").strip()
            if not phrase:
                continue
            word_count = len(phrase.split())
            if 2 <= word_count <= 10 and phrase not in existing:
                harm_keyword_loader.add_harm_keyword(phrase)
                print(f"[MODERATION] Learned new harm phrase: {phrase!r}")

    def _classify_intent(self, text: str) -> dict:
        try:
            output = self._intent(text[:1024], INTENT_LABELS, multi_label=True)
            label_scores = dict(zip(output["labels"], output["scores"]))

            harmful_score = max(
                (label_scores.get(lbl, 0) for lbl in HARMFUL_INTENT_LABELS),
                default=0,
            )
            safe_score = label_scores.get("safe, harmless content", 0)

            if harmful_score >= 0.55:
                top_harmful = max(
                    HARMFUL_INTENT_LABELS,
                    key=lambda l: label_scores.get(l, 0),
                )
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
            return {
                "intent": "safe",
                "reason": "",
                "scores": label_scores,
            }
        except Exception:
            return {"intent": "safe", "reason": "", "scores": {}}

    def _check_keywords(self, text: str, blocked_keywords: list[str]) -> list[str]:
        if not blocked_keywords:
            return []
        text_lower = text.lower()
        return [kw for kw in blocked_keywords if kw.lower() in text_lower]

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

    def _build_reason(self, local_result: dict, intent_result: dict, keyword_hits: list[str], intent: str, severity: str) -> str:
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
