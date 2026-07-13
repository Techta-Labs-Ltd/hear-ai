import json
import logging
import re
import threading

from app.config import settings as app_settings
from app.services.triton_client import get_triton_client

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


class LLMService:
    def __init__(self):
        self._lock = threading.Lock()

    @property
    def is_available(self) -> bool:
        return bool(app_settings.QWEN_LLM_ENABLED)

    def load(self):
        pass

    def unload(self):
        pass

    def ensure_loaded(self):
        pass

    def _generate(self, messages: list[dict], max_new_tokens: int = 256) -> str:
        with self._lock:
            return get_triton_client().llm_generate_sync(messages, max_tokens=max_new_tokens)

    @staticmethod
    def _extract_json(text: str) -> dict:
        if not text:
            return {}
        stripped = text.strip()
        if stripped.startswith("{"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass
        start = stripped.find("{")
        if start < 0:
            return {}
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(stripped)):
            ch = stripped[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(stripped[start : i + 1])
                    except json.JSONDecodeError:
                        break
        return {}

    def moderate(
        self,
        transcript: str,
        *,
        detoxify_scores: dict[str, float] | None = None,
        harm_keywords: list[str] | None = None,
        is_borderline: bool = False,
    ) -> dict:
        if not self.is_available:
            raise RuntimeError("LLM not available")

        found_kw = [kw for kw in (harm_keywords or []) if kw in transcript.lower()][:10]

        context_parts: list[str] = []
        if detoxify_scores:
            top = sorted(detoxify_scores.items(), key=lambda x: x[1], reverse=True)[:6]
            score_str = ", ".join(f"{k}={v:.2f}" for k, v in top if v > 0.05)
            if score_str:
                context_parts.append(f"Toxicity model pre-scores: {score_str}")
        if found_kw:
            context_parts.append(f"Flagged keywords found: {', '.join(found_kw)}")
        if is_borderline:
            context_parts.append(
                "The toxicity model is UNCERTAIN. Only flag if you are confident this is harmful."
            )

        context = ("\n" + "\n".join(context_parts)) if context_parts else ""

        user_content = (
            f"Transcript:\n{transcript[:2000]}"
            f"{context}\n\n"
            "Classify this content. Return ONLY this JSON (no markdown, no extra text):\n"
            '{"flagged":false,"severity":"none","intent":"safe","reason":"","flagged_categories":[]}\n\n'
            "severity: none | low | medium | high | critical\n"
            "intent: safe | questionable | harmful\n\n"
            "FLAG as harmful ONLY for:\n"
            "  - Direct threats of violence against a specific person\n"
            "  - Hate speech targeting a group (race, religion, gender, sexuality)\n"
            "  - Sexual content involving minors\n"
            "  - Explicit incitement to terrorism or mass violence\n\n"
            "DO NOT FLAG:\n"
            "  - Sports commentary, match results, player analysis\n"
            "  - Cooking shows, recipes, food content\n"
            "  - News reporting, journalism, current affairs\n"
            "  - Music lyrics about general themes (love, life, community)\n"
            "  - Podcasts, interviews, general conversation\n"
            "  - Fiction and storytelling\n"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise content safety classifier for an audio podcast platform. "
                    "Return ONLY valid JSON with no explanation or markdown."
                ),
            },
            {"role": "user", "content": user_content},
        ]

        raw = self._generate(messages, max_new_tokens=140)
        parsed = self._extract_json(raw)

        severity = parsed.get("severity", "none")
        if severity not in ("none", "low", "medium", "high", "critical"):
            severity = "none"
        intent = parsed.get("intent", "safe")
        if intent not in ("safe", "questionable", "harmful"):
            intent = "safe"
        flagged = bool(parsed.get("flagged", False))
        if intent == "harmful" and severity in ("high", "critical"):
            flagged = True
        if intent == "safe":
            flagged = False
            severity = "none"

        logger.info(
            "[LLM/MODERATE] flagged=%s severity=%s intent=%s borderline=%s",
            flagged, severity, intent, is_borderline,
        )

        return {
            "flagged": flagged,
            "severity": severity,
            "intent": intent,
            "reason": parsed.get("reason", ""),
            "flagged_categories": parsed.get("flagged_categories", []),
            "blocked_words_found": found_kw,
        }

    def categorize(
        self,
        transcript: str,
        categories: list[str],
        tags: list[str],
        keyword_hits: dict[str, float] | None = None,
        max_categories: int = 2,
        nli_top_categories: list[str] | None = None,
        taxonomy_paths: list[str] | None = None,
    ) -> dict:
        if not self.is_available:
            raise RuntimeError("LLM not available")

        kw_hint = ""
        if keyword_hits:
            top = sorted(keyword_hits.items(), key=lambda x: x[1], reverse=True)[:8]
            kw_hint = f"\nKeyword analysis pre-detected: {', '.join(t for t, _ in top)}"
        nli_hint = ""
        if nli_top_categories:
            nli_hint = (
                f"\nTranscript classifier top subjects (strong signal): "
                f"{', '.join(nli_top_categories[:6])}"
            )

        cat_str = ", ".join(categories[:50])
        tag_str = ", ".join(tags[:120])
        tax_block = ""
        if taxonomy_paths:
            tax_block = (
                "\nEditorial taxonomy paths (use for subject context; map to tags/categories):\n"
                + "\n".join(f"- {p}" for p in taxonomy_paths[:25])
                + "\n"
            )

        user_content = (
            f"Suggested categories (use when they fit; otherwise new_categories): {cat_str}\n"
            f"Suggested tags (use when they fit; otherwise new_tags): {tag_str}\n"
            f"{tax_block}{kw_hint}{nli_hint}\n\n"
            f"Transcript:\n{transcript[:4000]}\n\n"
            "You are the PRIMARY classifier. Read the FULL transcript and decide what this audio is ABOUT.\n"
            "Keyword/NLI/taxonomy hints are advisory — override them when they misread the story.\n"
            "Return ONLY this JSON (no markdown, no extra text):\n"
            '{"tags":["#accessibility"],"categories":["Personal lived experience"],'
            '"sentiment":"neutral","new_tags":["#guidedogs"],"new_categories":[]}\n\n'
            "Rules:\n"
            "- tags: up to 5 hashtag tags with # prefix (e.g. #environment, #community) — "
            "NEVER return taxonomy paths with ' > ' as tags\n"
            f"- categories: up to {max_categories} flat editorial category names (catalog or new) — "
            "NEVER return hierarchical taxonomy paths with ' > ' as categories\n"
            "- new_tags: REQUIRED when the best subject tags are missing from the suggested list "
            "(create concise # tags, e.g. #assistive-technology, #guide-dogs)\n"
            "- new_categories: REQUIRED when no suggested category fits (e.g. Personal lived experience)\n"
            "- Put your best tags in BOTH tags and new_tags if needed — unknown labels are saved to the catalog\n"
            "- sentiment: positive | negative | neutral\n"
            "- Classify the MAIN SUBJECT (music, sport, news, wildlife, politics) — NOT audio format alone\n"
            "- Do NOT use Podcast/Documentary as the only category when the subject is music, news, wildlife, etc.\n"
            "- News stories about wildlife photographers, awards, or nature documentaries: use News, Wildlife, "
            "Photography, Environment — NOT Veterinary, Dentistry, or clinical medicine unless the piece is "
            "about animal healthcare\n"
            "- First-person stories by blind or visually impaired speakers about guide dogs, smart glasses "
            "(e.g. Meta Ray-Ban), AI assistants (e.g. Orion), independence, and family: use Personal lived "
            "experience, Accessibility, Human connection; tags like #Accessibility #GuideDogs "
            "#AssistiveTechnology #SmartGlasses — NOT Wildlife, Nature, #wildlife, or Technology as the main "
            "subject (trees/lakes mentioned in passing are NOT a wildlife documentary)\n"
            "- Technology history / audio format documentaries (Minidisc, Betamax, cassette, format wars, Sony, Philips): "
            "use Technology, Entertainment, or Documentary; tags like #technology #history #audio — "
            "NOT #wildlife #photography #accessibility and NEVER taxonomy paths about smart glasses\n"
            "- Sports commentary → Sports; recipe → Food; obituary → Obituaries + News\n"
            "- Prefer specific subjects over vague buckets (Technology, Business) unless truly central\n"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert audio content categorizer for a Talking Newspaper and podcast platform. "
                    "Return ONLY valid JSON. Your categories must match what a human editor would assign."
                ),
            },
            {"role": "user", "content": user_content},
        ]

        raw = self._generate(messages, max_new_tokens=280)
        parsed = self._extract_json(raw)

        valid_tags = {t.lower() for t in tags}
        valid_cats = {c.lower() for c in categories}

        def _norm_tag(raw: str) -> str:
            t = str(raw or "").strip().lower().lstrip("#")
            t = re.sub(r"\s+", "-", t)
            t = re.sub(r"[^a-z0-9\-]", "", t)
            return f"#{t}" if t else ""

        tags_out: list[str] = []
        new_tags: list[str] = []
        seen_tags: set[str] = set()
        for raw in list(parsed.get("tags", [])) + list(parsed.get("new_tags", [])):
            if not isinstance(raw, str):
                continue
            norm = _norm_tag(raw)
            if not norm or norm in seen_tags:
                continue
            seen_tags.add(norm)
            if norm.lower() in valid_tags:
                tags_out.append(norm)
            else:
                new_tags.append(norm)
            if len(tags_out) + len(new_tags) >= 5:
                break

        cats_out: list[str] = []
        new_cats: list[str] = []
        seen_cats: set[str] = set()
        for raw in list(parsed.get("categories", [])) + list(parsed.get("new_categories", [])):
            if not isinstance(raw, str):
                continue
            cat = re.sub(r"\s+", " ", raw.strip())
            if not cat:
                continue
            key = cat.lower()
            if key in seen_cats:
                continue
            seen_cats.add(key)
            if key in valid_cats:
                for c in categories:
                    if c.lower() == key:
                        cats_out.append(c)
                        break
            else:
                new_cats.append(cat)
            if len(cats_out) + len(new_cats) >= max_categories:
                break
        sentiment = parsed.get("sentiment", "neutral")
        if sentiment not in ("positive", "negative", "neutral"):
            sentiment = "neutral"

        logger.info("[LLM/CATEGORIZE] tags=%s categories=%s new_tags=%s new_categories=%s", tags_out, cats_out, new_tags, new_cats)

        return {"tags": tags_out, "categories": cats_out, "sentiment": sentiment, "new_tags": new_tags, "new_categories": new_cats}

    def build_discovery_profile(
        self,
        transcript: str,
        *,
        track_name: str = "",
        duration_seconds: float | None = None,
        categorization_hint: str = "",
        prior_description: str | None = None,
        partial_transcript: bool = False,
        max_search_phrases: int = 12,
        taxonomy_paths: list[str] | None = None,
        strict: bool = False,
    ) -> dict:
        if not self.is_available:
            raise RuntimeError("LLM not available")
        body = (transcript or "").strip()[:8000]
        title = (track_name or "").strip()[:200]
        hint = (categorization_hint or "").strip()[:600]
        prior = (prior_description or "").strip()[:500]
        dur = f"{duration_seconds:.0f}s" if duration_seconds else "unknown"
        partial_note = (
            "The transcript may be partial or from metadata only; infer carefully."
            if partial_transcript
            else ""
        )
        tax_block = "none"
        if taxonomy_paths:
            tax_block = "\n".join(f"- {p}" for p in taxonomy_paths[:40])
        strict_note = ""
        if strict:
            strict_note = (
                "CRITICAL: Fill EVERY field below. Never paste the transcript opening as summary_short or "
                "one_line_description. Invent a discovery title — never use the track filename.\n"
            )
        user_content = (
            f"Track filename (do NOT use as title_suggestion): {title or 'unknown'}\n"
            f"Duration: {dur}\n"
            f"Existing tags/categories hint: {hint or 'none'}\n"
            f"Prior description (regenerate fully, do not copy blindly): {prior or 'none'}\n"
            f"{partial_note}\n"
            f"{strict_note}\n"
            f"Transcript:\n{body or '(no transcript)'}\n\n"
            "Build a rich discovery profile for spoken-word audio search and recommendations.\n"
            "Do NOT reduce this to a single generic tag like Technology unless technology is truly the main subject.\n"
            "Prioritize the SUBJECT (music, sport, accessibility, news) and emotional centre — not the word podcast.\n"
            "primary_genre = subject genre (e.g. Music discussion, Personal lived experience) — never just Podcast.\n"
            "Write summaries in third person about the content — never quote the opening line of the transcript.\n"
            f"Return ONLY valid JSON (no markdown) with at least 5 search_phrases and 3 key_themes:\n"
            '{"title_suggestion":"","summary_short":"","summary_long":"","one_line_description":"",'
            '"speaker":"","primary_genre":"","main_topic":"","secondary_topics":[],'
            '"audience_relevance":[],"tone":[],'
            '"key_themes":[],"controlled_tags":[],"'
            '"entities":{"people":[],"animals":[],"products":[],"apps":[],"technologies":[]},'
            '"search_phrases":[],"recommendation_labels":[],"sensitivity_flags":[],'
            '"confidence":{"primary_genre":0.9,"main_topic":0.9},'
            '"embedding_source_text":"","freeform_tags":[]}\n'
            "title_suggestion = human discovery title (not the filename). "
            "speaker = human narrator's personal name ONLY when they introduce themselves in the transcript "
            "(e.g. Denise Wallace). Use empty string if no person is named. "
            "NEVER put devices, products (Minidisc, Walkman, iPod), brands, formats, apps, topics, or taxonomy paths in speaker. "
            "controlled_tags = ONLY hierarchical taxonomy paths from the vocabulary that clearly match the story — "
            "do NOT tag accessibility/smart glasses unless the piece is about blind users, guide dogs, or assistive tech. "
            "Do NOT tag wildlife/photography unless the story is about animals, nature media, or photo awards. "
            "key_themes = insight-level themes (e.g. independence is about choice). "
            "audience_relevance = who would find this relevant. "
            "controlled_tags = hierarchical paths with ' > ' (pick from vocabulary below when possible). "
            "search_phrases = natural-language queries listeners might use. "
            "recommendation_labels = 'For listeners interested in ...' style lines. "
            "entities = names of people, animals, products, apps, technologies mentioned.\n"
            f"Reference taxonomy vocabulary:\n{tax_block}\n"
            "summary_short = 2-3 sentence engine blurb; summary_long = warm human paragraph; "
            "one_line_description = single catalogue line; "
            "embedding_source_text = dense keyword-rich line for vector search (no full transcript)."
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert audio discovery metadata analyst for a podcast and Talking Newspaper platform. "
                    "Return ONLY one complete JSON object. No markdown."
                ),
            },
            {"role": "user", "content": user_content},
        ]
        max_tokens = max(400, int(app_settings.DISCOVERY_MAX_NEW_TOKENS))
        raw = self._generate(messages, max_new_tokens=max_tokens)
        parsed = self._extract_json(raw)
        if not isinstance(parsed, dict) or not parsed:
            return {}
        entities_raw = parsed.get("entities")
        if not isinstance(entities_raw, dict):
            entities_raw = {}
        parsed["entities"] = {
            "people": [str(x) for x in entities_raw.get("people", []) if x][:20],
            "animals": [str(x) for x in entities_raw.get("animals", []) if x][:20],
            "products": [str(x) for x in entities_raw.get("products", []) if x][:20],
            "apps": [str(x) for x in entities_raw.get("apps", []) if x][:20],
            "technologies": [str(x) for x in entities_raw.get("technologies", []) if x][:20],
        }
        ss = str(parsed.get("summary_short") or parsed.get("short_summary") or "").strip()
        if ss:
            parsed["summary_short"] = ss
            parsed["short_summary"] = ss
        if not str(parsed.get("embedding_source_text") or "").strip():
            parsed["embedding_source_text"] = ss
        return parsed

    def describe_audio_content(
        self,
        transcript: str,
        *,
        track_name: str = "",
        context_hint: str = "",
    ) -> str | None:
        try:
            parsed = self.build_discovery_profile(
                transcript,
                track_name=track_name,
                categorization_hint=context_hint,
                max_search_phrases=6,
            )
        except RuntimeError:
            return None
        for key in ("one_line_description", "summary_short", "short_summary", "summary_long"):
            val = parsed.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()[:500]
        return None

    def resolve_playback_instruction_speeds(self, instruction: str) -> list[float]:
        if not self.is_available:
            return []
        ins = (instruction or "").strip()
        if not ins:
            return []
        user_content = (
            f"User playback request:\n{ins[:1200]}\n\n"
            "You are an audio playback speed assistant. Speed values are multipliers of normal (1.0). "
            "Slower: 0.5-0.9, faster: 1.1-3.0. Map phrases: \"normal speed\"->1.0, \"double speed\"->2.0, \"half speed\"->0.5.\n"
            "Return ONLY this JSON (no markdown):\n"
            '{"speeds":[1.5]}\n'
            "The \"speeds\" array must list every distinct playback multiplier the user asked for, "
            "each between 0.5 and 3.0. If they only asked for normal (1.0), use an empty array []. "
            "At most 8 values."
        )
        messages = [
            {"role": "system", "content": 'Return ONLY valid JSON with a "speeds" array of numbers.'},
            {"role": "user", "content": user_content},
        ]
        raw = self._generate(messages, max_new_tokens=120)
        parsed = self._extract_json(raw)
        arr = parsed.get("speeds")
        if not isinstance(arr, list):
            arr = parsed.get("multipliers")
        if not isinstance(arr, list):
            return []
        out: list[float] = []
        for x in arr[:8]:
            try:
                v = float(x)
            except (TypeError, ValueError):
                continue
            if 0.5 <= v <= 3.0 and abs(v - 1.0) > 1e-6:
                out.append(round(v, 4))
        return sorted(set(out))


_llm_service = None


def get_llm_service():
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
