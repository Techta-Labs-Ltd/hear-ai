import json
import logging
import re
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import settings as app_settings

try:
    from transformers import BitsAndBytesConfig
    _BNB_AVAILABLE = True
except ImportError:
    BitsAndBytesConfig = None  # type: ignore[assignment,misc]
    _BNB_AVAILABLE = False

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MIN_VRAM_GB = 4.5


class LLMService:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._available = False

    def load(self):
        if not app_settings.QWEN_LLM_ENABLED:
            logger.info("[LLM] Qwen disabled (QWEN_LLM_ENABLED=false) — toxic-bert + NLI path")
            return
        if not self._has_enough_gpu():
            logger.info("[LLM] No GPU with ≥%.1f GB VRAM — LLM disabled", MIN_VRAM_GB)
            return
        try:
            logger.info("[LLM] Loading %s on GPU...", MODEL_ID)
            self._tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

            load_kwargs: dict = {
                "device_map": "auto",
                "torch_dtype": torch.float16,
            }
            if _BNB_AVAILABLE:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                logger.info("[LLM] 4-bit quantisation enabled (~4 GB VRAM)")
            else:
                logger.info("[LLM] bitsandbytes not available — loading fp16 (~14 GB VRAM)")

            self._model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
            self._model.eval()
            self._available = True
            logger.info("[LLM] %s ready on %s", MODEL_ID, next(self._model.parameters()).device)

        except Exception as exc:
            logger.warning("[LLM] Load failed (%s) — falling back to local models", exc)
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def _has_enough_gpu(self) -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info("[LLM] GPU VRAM: %.1f GB (need ≥%.1f GB)", vram_gb, MIN_VRAM_GB)
            return vram_gb >= MIN_VRAM_GB
        except Exception:
            return False

    def _generate(self, messages: list[dict], max_new_tokens: int = 256) -> str:
        with self._lock:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            new_ids = output[0][inputs["input_ids"].shape[1]:]
            return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

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
        if not self._available:
            raise RuntimeError("LLM not loaded")

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
    ) -> dict:
        if not self._available:
            raise RuntimeError("LLM not loaded")

        kw_hint = ""
        if keyword_hits:
            top = sorted(keyword_hits.items(), key=lambda x: x[1], reverse=True)[:8]
            kw_hint = f"\nKeyword analysis pre-detected: {', '.join(t for t, _ in top)}"

        cat_str = ", ".join(categories[:40])
        tag_str = ", ".join(tags[:100])

        user_content = (
            f"Available categories (choose from these only): {cat_str}\n"
            f"Available tags (choose from these only, keep # prefix): {tag_str}\n"
            f"{kw_hint}\n\n"
            f"Transcript:\n{transcript[:2000]}\n\n"
            "Return ONLY this JSON (no markdown, no extra text):\n"
            '{"tags":["#Sports"],"categories":["Sports"],"sentiment":"neutral","new_tags":[],"new_categories":[]}\n\n'
            "Rules:\n"
            "- tags: up to 5, must start with #, must come from the available list\n"
            f"- categories: up to {max_categories}, must come from the available list\n"
            "- new_tags: up to 5 NEW tags you discovered that are NOT in the available list — must start with #\n"
            "- new_categories: up to 2 NEW categories NOT in the available list\n"
            "- sentiment: positive | negative | neutral\n"
            "- Base on MAIN topics only — ignore passing mentions\n"
            "- If multiple distinct topics exist in this text, include a category for each\n"
        )

        messages = [
            {
                "role": "system",
                "content": "You are an expert audio content categorizer. Return ONLY valid JSON.",
            },
            {"role": "user", "content": user_content},
        ]

        raw = self._generate(messages, max_new_tokens=180)
        parsed = self._extract_json(raw)

        valid_tags = set(tags)
        valid_cats = set(categories)

        tags_out = [t for t in parsed.get("tags", []) if t in valid_tags][:5]
        cats_out = [c for c in parsed.get("categories", []) if c in valid_cats][:max_categories]
        new_tags = [
            t for t in parsed.get("new_tags", [])
            if isinstance(t, str) and t.startswith("#") and t not in valid_tags
        ][:5]
        new_cats = [
            c for c in parsed.get("new_categories", [])
            if isinstance(c, str) and c and c not in valid_cats
        ][:2]
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
        if not self._available:
            raise RuntimeError("LLM not loaded")
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
            "Prioritize content type (personal story, interview, news, review) and the emotional centre of the piece.\n"
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
            "speaker = primary narrator name if stated (e.g. Denise Wallace). "
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
        if not self._available:
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


llm_service = LLMService()
