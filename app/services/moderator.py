import asyncio
import json

import httpx
import torch
from transformers import pipeline as hf_pipeline

from app.config import settings

MODERATION_MODEL = "unitary/toxic-bert"

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

HARM_KEYWORDS: list[str] = [
    # Direct threats
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

    # UK drill / road slang threats
    "spin the block", "up the score", "catch a body", "caught a body",
    "drop a body", "wet him up", "wet her up", "splash him", "splash her",
    "poke him", "poke her", "shank him", "shank her", "shank them",
    "ride out on", "back out the strap", "back out the tool",
    "link man with the strap", "up the stick", "do him dirty",
    "score on sight", "on sight", "put him in the dirt",
    "slide on", "leave him leaking", "leave her leaking", "leave man leaking",
    "stomp him out", "stomp her out", "rush him", "rush her",
    "spin his block", "spin her block", "skeng", "no lacking",

    # US slang / trap / gang threats
    "cap him", "clap him", "blick him", "smoke him", "smoke her", "smoke them",
    "catch him slipping", "catch her slipping", "run up on him", "run up on her",
    "squeeze the trigger", "pull the trigger on",
    "put him six feet", "put her six feet", "send him to god",
    "drop him", "drop her", "body bag him", "body bag her",
    "get smoked", "get merked", "get rocked", "get domed",
    "put hands on", "beat him down", "beat her down", "stomp him out",
    "bust at", "bust shots at", "send shots", "let it bang",
    "bust a cap", "dump on", "chop him down", "chop her down",
    "air him out", "air her out", "hit the lick", "rob him", "rob her",

    # Weapons
    "with my strap", "strapped up", "glizzy", "chopper", "draco",
    "beam on it", "pole", "banger", "burner", "gat", "mac-10",
    "uzi", "ak", "ar-15", "banana clip", "extended clip", "drum mag",
    "switch", "auto switch", "glock with a switch",

    # Hate speech / slurs
    "kill all", "death to", "gas the", "exterminate the",
    "white power", "white supremacy", "n*gger", "nigger", "nigga die",
    "kike", "spic", "chink", "sand nigger", "raghead",
    "bomb the mosque", "bomb the church", "blow up the school",
    "jihad against", "holy war against",

    # Sexual exploitation / CSAM
    "child porn", "kiddie porn", "cp link", "rape a child",
    "molest a child", "underaged naked", "loli",

    # Self-harm
    "kill myself", "killing myself", "want to die", "going to end it",
    "slit my wrists", "hang myself", "suicide pact",
    "overdose on purpose", "want to overdose",

    # Extremism
    "allahu akbar bomb", "suicide bomb", "blow myself up",
    "school shooting", "mass shooting", "shoot up the", "bomb threat",
    "pipe bomb", "ied", "nail bomb",
]


class ModerationService:
    def __init__(self):
        self._classifier = None
        self._device = 0 if torch.cuda.is_available() else -1

    def load(self):
        self._classifier = hf_pipeline(
            "text-classification",
            model=MODERATION_MODEL,
            device=self._device,
            top_k=None,
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

        text_lower = text.lower()

        # Hard-match built-in harm phrases — short-circuit immediately
        built_in_hits = [kw for kw in HARM_KEYWORDS if kw in text_lower]
        if built_in_hits:
            return {
                "flagged": True,
                "severity": SEVERITY_CRITICAL,
                "intent": "harmful",
                "reason": f"Contains flagged harmful language: {', '.join(built_in_hits[:5])}",
                "flagged_categories": ["Threats / Violence"],
                "blocked_words_found": built_in_hits,
            }

        loop = asyncio.get_event_loop()

        local_task = loop.run_in_executor(None, self._classify_local, text)
        context_task = self._classify_with_context(text, blocked_keywords or [])
        keyword_hits = self._check_keywords(text, blocked_keywords or [])

        local_result, context_result = await asyncio.gather(local_task, context_task)

        severity = self._compute_severity(local_result, context_result)
        flagged = severity in (SEVERITY_HIGH, SEVERITY_CRITICAL)
        intent = context_result.get("intent", "unknown")
        reason = self._build_reason(local_result, context_result, keyword_hits, intent, severity)
        flagged_categories = self._get_flagged_categories(local_result, context_result)

        return {
            "flagged": flagged,
            "severity": severity,
            "intent": intent,
            "reason": reason,
            "flagged_categories": flagged_categories,
            "blocked_words_found": context_result.get("blocked_words", keyword_hits),
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

        # Threats and severe toxicity > 0.5 must be hard flagged
        return {
            "flagged": any(v >= 0.5 for k, v in scores.items() if k in ("severe_toxic", "threat", "identity_hate")),
            "max_score": max(scores.values()) if scores else 0,
            "high_scores": high_scores,
            "scores": scores,
        }

    async def _classify_with_context(self, text: str, blocked_keywords: list[str]) -> dict:
        if not settings.OPENAI_API_KEY:
            keyword_hits = self._check_keywords(text, blocked_keywords)
            if keyword_hits:
                return {
                    "intent": "cautionary",
                    "reason": f"Contains monitored words: {', '.join(keyword_hits)}",
                    "blocked_words": keyword_hits,
                    "openai_used": False,
                }
            return {"intent": "safe", "reason": "", "blocked_words": [], "openai_used": False}

        blocked_section = ""
        if blocked_keywords:
            blocked_section = f"\nMonitored keywords: [{', '.join(blocked_keywords)}]\n"

        prompt = (
            "You are a content moderation system for an audio recording platform. "
            "Analyze this transcript and determine if the SPEAKER is producing harmful content.\n\n"
            "IMPORTANT RULES:\n"
            "- A reporter discussing crime, robbery, scams = NOT harmful (reporting)\n"
            "- Someone warning others about scams = NOT harmful (educational)\n"
            "- A podcast about war or politics = NOT harmful (discussion)\n"
            "- Someone directly threatening violence = harmful\n"
            "- Someone promoting hate or discrimination = harmful\n"
            "- Someone producing explicit sexual content targeting minors = harmful\n"
            "- Profanity alone is NOT harmful unless directed at someone as abuse\n\n"
            "Determine the INTENT:\n"
            "- 'safe': normal discussion, reporting, educating, informing\n"
            "- 'cautionary': sensitive topics discussed neutrally, mild language\n"
            "- 'questionable': borderline, may benefit from human review\n"
            "- 'harmful': speaker is threatening, promoting violence/hate, or producing genuinely harmful content\n\n"
            f"{blocked_section}"
            "If monitored keywords appear, only flag them if used in a genuinely harmful way.\n\n"
            f"Transcript:\n{text[:3000]}\n\n"
            "Return ONLY valid JSON:\n"
            '{"intent": "safe|cautionary|questionable|harmful", '
            '"reason": "one clear sentence explaining your decision", '
            '"blocked_words_in_harmful_context": ["only words used harmfully, empty if none"]}'
        )

        try:
            async with httpx.AsyncClient(timeout=45) as client:
                response = await client.post(
                    f"{settings.OPENAI_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": settings.OPENAI_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "max_tokens": 200,
                    },
                )
                response.raise_for_status()
                data = response.json()

            content = data["choices"][0]["message"]["content"].strip()
            content = content.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(content)

            return {
                "intent": parsed.get("intent", "safe"),
                "reason": parsed.get("reason", ""),
                "blocked_words": parsed.get("blocked_words_in_harmful_context", []),
                "openai_used": True,
            }
        except Exception:
            keyword_hits = self._check_keywords(text, blocked_keywords)
            if keyword_hits:
                return {
                    "intent": "cautionary",
                    "reason": f"Contains monitored words: {', '.join(keyword_hits)}. Context analysis unavailable.",
                    "blocked_words": keyword_hits,
                    "openai_used": False,
                }
            return {"intent": "safe", "reason": "", "blocked_words": [], "openai_used": False}

    def _check_keywords(self, text: str, blocked_keywords: list[str]) -> list[str]:
        if not blocked_keywords:
            return []
        text_lower = text.lower()
        return [kw for kw in blocked_keywords if kw.lower() in text_lower]

    def _compute_severity(self, local_result: dict, context_result: dict) -> str:
        intent = context_result.get("intent", "safe")
        openai_used = context_result.get("openai_used", True)
        max_toxic = local_result.get("max_score", 0)
        local_flagged = local_result.get("flagged", False)

        # Both agree it's harmful
        if intent == "harmful" and local_flagged:
            return SEVERITY_CRITICAL

        # OpenAI alone says harmful (context-aware)
        if intent == "harmful":
            return SEVERITY_HIGH

        if local_flagged:
            if not openai_used:
                # No OpenAI to cross-check — trust toxic-bert directly
                return SEVERITY_HIGH
            # OpenAI is available and disagrees — only flag if uncertain
            if intent == "questionable":
                return SEVERITY_HIGH
            # OpenAI says safe/cautionary — trust it, downgrade to non-flagged
            return SEVERITY_MEDIUM

        if intent == "questionable":
            return SEVERITY_MEDIUM

        if intent == "cautionary" or max_toxic > 0.6:
            return SEVERITY_LOW

        return SEVERITY_NONE

    def _get_flagged_categories(self, local_result: dict, context_result: dict) -> list[str]:
        categories = []
        for label, score in local_result.get("high_scores", {}).items():
            if label in TOXIC_CATEGORIES:
                categories.append(TOXIC_CATEGORIES[label])
        return categories

    def _build_reason(self, local_result: dict, context_result: dict, keyword_hits: list[str], intent: str, severity: str) -> str:
        context_reason = context_result.get("reason", "")
        if context_reason:
            return context_reason

        parts = []
        high = local_result.get("high_scores", {})
        if high:
            labels = [TOXIC_CATEGORIES.get(k, k) for k in high.keys()]
            parts.append(f"Detected: {', '.join(labels)}")

        if keyword_hits:
            parts.append(f"Monitored words found: {', '.join(keyword_hits)}")

        if not parts:
            if severity == SEVERITY_NONE:
                return "Content appears safe"
            return "Content flagged by automated analysis"

        return ". ".join(parts)
