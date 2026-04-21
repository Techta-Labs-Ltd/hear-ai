import json
import logging
import re
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
                    temperature=1.0,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            new_ids = output[0][inputs["input_ids"].shape[1]:]
            return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    @staticmethod
    def _extract_json(text: str) -> dict:
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
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
            '{"tags":["#Sports"],"categories":["Sports"],"sentiment":"neutral"}\n\n'
            "Rules:\n"
            "- tags: up to 5, must start with #, must come from the available list\n"
            f"- categories: up to {max_categories}, must come from the available list\n"
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
        sentiment = parsed.get("sentiment", "neutral")
        if sentiment not in ("positive", "negative", "neutral"):
            sentiment = "neutral"

        logger.info("[LLM/CATEGORIZE] tags=%s categories=%s", tags_out, cats_out)

        return {"tags": tags_out, "categories": cats_out, "sentiment": sentiment}


llm_service = LLMService()
