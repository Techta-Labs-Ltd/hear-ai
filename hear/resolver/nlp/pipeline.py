import logging
from typing import Any

import spacy
from spacy.lang.en import English

logger = logging.getLogger(__name__)

FILLER_WORDS = frozenset({
    "play", "listen", "hear", "watch", "find", "show", "give", "get", "open",
    "start", "tell", "me", "please", "the", "a", "an", "to", "for", "some",
    "can", "you", "i", "want", "like", "need", "looking", "something",
    "about", "with", "my", "now", "just", "let", "id", "us", "we", "that",
    "what", "who", "where", "when", "how", "this", "it", "on",
    "latest", "newest", "recent", "today", "yesterday", "tomorrow", "anything",
})

ACTION_WORDS = frozenset({
    "play", "listen", "hear", "start", "find", "show", "give", "get", "open",
    "browse", "whats", "what's", "tell", "search", "look",
})

CREATOR_PREPS = frozenset({"from", "by"})
LOCATION_PREPS = frozenset({"in", "near", "around", "at"})


class NLPPipeline:
    def __init__(self) -> None:
        self._nlp = None
        self._enabled = False
        self._alias_to_creator: dict[str, str] = {}
        self._alias_to_org: dict[str, str] = {}
        self._city_names: set[str] = set()
        try:
            self._nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            logger.info("spacy_en_core_web_sm_loaded")
            self._enabled = True
        except Exception:
            logger.warning("spacy_model_load_failed, using fallback tokenizer")
            self._nlp = English()

    def load_aliases(self, creators: list[dict], orgs: list[dict]) -> None:
        self._alias_to_creator.clear()
        self._alias_to_org.clear()
        for c in creators:
            name = c.get("name", "")
            if name:
                self._alias_to_creator[c.get("normalized", name.lower())] = name
            for alias in c.get("aliases", []):
                if alias:
                    self._alias_to_creator[alias.lower()] = name
        for o in orgs:
            name = o.get("name", "")
            if name:
                self._alias_to_org[o.get("normalized", name.lower())] = name
            for alias in o.get("aliases", []):
                if alias:
                    self._alias_to_org[alias.lower()] = name

    def load_cities(self, locations: list[dict]) -> None:
        self._city_names.clear()
        for loc in locations:
            city = loc.get("normalized", "") or loc.get("city", "")
            if city:
                self._city_names.add(city.lower())

    def detect_entity(self, token: str) -> tuple[str | None, str]:
        t = token.lower().strip()
        if t in self._alias_to_creator:
            return self._alias_to_creator[t], "creator"
        if t in self._alias_to_org:
            return self._alias_to_org[t], "org"
        if t in self._city_names:
            return t, "location"
        return None, ""

    def process(self, utterance: str) -> dict[str, Any]:
        utterance_clean = utterance.lower().strip()
        tokens = [t for t in utterance_clean.split() if t not in FILLER_WORDS]
        if not tokens:
            return {"category_tokens": [], "creator_token": None, "org_token": None, "location_token": None, "action": "general"}

        action = "general"
        for t in tokens:
            if t in ACTION_WORDS:
                action = "play" if t in ("play", "listen", "hear", "start") else t

        if self._enabled and self._nlp is not None:
            return self._spacy_process(tokens, action)

        return self._simple_process(tokens, action)

    def _spacy_process(self, tokens: list[str], action: str) -> dict[str, Any]:
        text = " ".join(tokens)
        doc = self._nlp(text)

        creator_parts: list[str] = []
        org_parts: list[str] = []
        location_parts: list[str] = []
        category_parts: list[str] = []
        seen_prep = None
        buffer: list[str] = []

        for token in doc:
            word = token.text.lower()
            if word in CREATOR_PREPS:
                if buffer:
                    if seen_prep == "creator":
                        creator_parts.extend(buffer)
                    else:
                        category_parts.extend(buffer)
                if creator_parts:
                    seen_prep = "done"
                else:
                    seen_prep = "creator"
                buffer = []
            elif word in LOCATION_PREPS:
                if buffer:
                    if seen_prep == "creator":
                        creator_parts.extend(buffer)
                    else:
                        category_parts.extend(buffer)
                seen_prep = "location"
                buffer = []
            elif word in self._alias_to_creator and seen_prep != "location" and seen_prep != "done":
                creator_parts.append(word)
            elif hasattr(token, "ent_type_") and token.ent_type_ == "LOC" and seen_prep != "creator" and seen_prep != "done":
                location_parts.append(word)
            elif seen_prep == "done":
                pass
            else:
                buffer.append(word)

        if buffer:
            if seen_prep == "creator":
                creator_parts.extend(buffer)
            elif seen_prep == "location":
                location_parts.extend(buffer)
            elif seen_prep == "done":
                pass
            else:
                category_parts.extend(buffer)

        return self._build_result(category_parts, creator_parts, org_parts, location_parts, action)

    def _simple_process(self, tokens: list[str], action: str) -> dict[str, Any]:
        creator_parts: list[str] = []
        location_parts: list[str] = []
        category_parts: list[str] = []
        seen_prep = None
        buffer: list[str] = []

        for word in tokens:
            w = word.strip()
            entity, etype = self.detect_entity(w)
            if etype == "creator" and seen_prep != "location" and seen_prep != "done":
                creator_parts.append(w)
                continue
            if etype == "org" and seen_prep != "location" and seen_prep != "done":
                creator_parts.append(w)
                continue
            if etype == "location" and seen_prep != "creator" and seen_prep != "done":
                location_parts.append(w)
                continue
            if w in CREATOR_PREPS:
                if buffer:
                    if seen_prep == "creator":
                        creator_parts.extend(buffer)
                    else:
                        category_parts.extend(buffer)
                if creator_parts:
                    seen_prep = "done"
                else:
                    seen_prep = "creator"
                buffer = []
            elif w in LOCATION_PREPS:
                if buffer:
                    if seen_prep == "creator":
                        creator_parts.extend(buffer)
                    else:
                        category_parts.extend(buffer)
                seen_prep = "location"
                buffer = []
            elif seen_prep == "done":
                pass
            else:
                buffer.append(w)

        if buffer:
            if seen_prep == "creator":
                creator_parts.extend(buffer)
            elif seen_prep == "location":
                location_parts.extend(buffer)
            elif seen_prep == "done":
                pass
            else:
                category_parts.extend(buffer)

        return self._build_result(category_parts, creator_parts, [], location_parts, action)

    def _build_result(self, cat_parts: list[str], creator_parts: list[str], org_parts: list[str], location_parts: list[str], action: str) -> dict[str, Any]:
        cat_token = " ".join(cat_parts) if cat_parts else ""
        creator_token = " ".join(creator_parts) if creator_parts else None
        location_token = " ".join(location_parts) if location_parts else None
        org_token = " ".join(org_parts) if org_parts else None

        if not creator_token and not org_token and cat_token in FILLER_WORDS:
            cat_token = ""

        return {
            "category_tokens": cat_parts,
            "creator_token": creator_token,
            "org_token": org_token,
            "location_token": location_token,
            "action": action,
        }
