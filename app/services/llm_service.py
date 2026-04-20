import json
import logging
import os
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

LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MIN_VRAM_GB = 6.0          # 8B model with 4-bit quant needs ~4.5 GB; 6 GB gives headroom


class LlamaService:
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._available = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self):
        if not self._has_enough_gpu():
            logger.info(
                "[LLAMA] No GPU with ≥%.0f GB VRAM detected — Llama disabled, using fallback pipeline",
                MIN_VRAM_GB,
            )
            return
        try:
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            if not hf_token:
                logger.warning(
                    "[LLAMA] HF_TOKEN env var not set — gated model %s may fail to download. "
                    "Set HF_TOKEN in .env to your HuggingFace access token.",
                    LLAMA_MODEL,
                )

            logger.info("[LLAMA] Loading %s on GPU...", LLAMA_MODEL)
            self._tokenizer = AutoTokenizer.from_pretrained(
                LLAMA_MODEL, token=hf_token, use_fast=True
            )

            load_kwargs: dict = {
                "device_map": "auto",
                "torch_dtype": torch.float16,
                "token": hf_token,
            }
            if _BNB_AVAILABLE:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                logger.info("[LLAMA] 4-bit quantisation enabled (~4.5 GB VRAM)")
            else:
                logger.info("[LLAMA] bitsandbytes not available — loading fp16 (~14 GB VRAM)")

            self._model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL, **load_kwargs)
            self._model.eval()
            self._available = True
            device_info = next(self._model.parameters()).device
            logger.info("[LLAMA] Ready on %s", device_info)

        except Exception as exc:
            logger.warning("[LLAMA] Load failed (%s) — falling back to local models", exc)
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _has_enough_gpu(self) -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info("[LLAMA] GPU VRAM: %.1f GB (need ≥%.0f GB)", vram_gb, MIN_VRAM_GB)
            return vram_gb >= MIN_VRAM_GB
        except Exception:
            return False

    def _generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        with self._lock:
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
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
        """Extract first JSON object from model output, gracefully ignoring surrounding text."""
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}

    # ------------------------------------------------------------------
    # Public API — Moderation
    # ------------------------------------------------------------------

    def moderate(
        self,
        transcript: str,
        *,
        detoxify_scores: dict[str, float] | None = None,
        harm_keywords: list[str] | None = None,
        is_borderline: bool = False,
    ) -> dict:
        """Deep moderation analysis.

        Called by the moderator in two situations:
        1. Borderline (toxic-bert score 0.30–0.80): Llama decides whether to flag.
        2. Clearly harmful (score ≥ 0.80): Llama provides structured reason + categories.

        Args:
            detoxify_scores: Raw per-category scores from toxic-bert, e.g.
                             {"toxic": 0.72, "threat": 0.61, ...}
            harm_keywords:   Keywords from the hard-coded list that appeared in text.
            is_borderline:   True when toxic-bert was uncertain; Llama must decide.
        """
        if not self._available:
            raise RuntimeError("Llama not loaded")

        found_kw = [kw for kw in (harm_keywords or []) if kw in transcript.lower()][:10]

        # Build context lines for the prompt
        context_lines = []
        if detoxify_scores:
            top_scores = sorted(detoxify_scores.items(), key=lambda x: x[1], reverse=True)[:6]
            score_str = ", ".join(f"{k}={v:.2f}" for k, v in top_scores if v > 0.05)
            if score_str:
                context_lines.append(f"Toxicity model pre-scores: {score_str}")
        if found_kw:
            context_lines.append(f"Flagged keywords found: {', '.join(found_kw)}")
        if is_borderline:
            context_lines.append(
                "The toxicity model is UNCERTAIN about this content. "
                "Use your judgment — only flag if clearly harmful."
            )

        context = "\n".join(context_lines)

        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "You are a precise content safety classifier for an audio podcast platform. "
            "Return ONLY valid JSON with no explanation or markdown.\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"Transcript:\n{transcript[:2000]}\n\n"
            f"{context}\n\n" if context else
            f"Transcript:\n{transcript[:2000]}\n\n"
        )
        prompt += (
            "Classify this transcript. Return exactly:\n"
            '{"flagged":false,"severity":"none","intent":"safe","reason":"","flagged_categories":[]}\n\n'
            "severity: none | low | medium | high | critical\n"
            "intent: safe | questionable | harmful\n\n"
            "FLAG as harmful ONLY for:\n"
            "  - Direct threats of violence against a named/identifiable person\n"
            "  - Hate speech targeting a group (race, religion, gender, sexuality)\n"
            "  - Sexual content involving minors\n"
            "  - Explicit incitement to terrorism or mass violence\n\n"
            "DO NOT FLAG:\n"
            "  - Sports commentary, match results, player analysis\n"
            "  - Cooking shows, recipes, food content\n"
            "  - News reporting, journalism, current affairs\n"
            "  - Music lyrics about general themes (love, life, community)\n"
            "  - Podcasts, interviews, general conversation\n"
            "  - Fictional stories or drama\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

        raw = self._generate(prompt, max_new_tokens=140)
        parsed = self._extract_json(raw)

        severity = parsed.get("severity", "none")
        if severity not in ("none", "low", "medium", "high", "critical"):
            severity = "none"
        intent = parsed.get("intent", "safe")
        if intent not in ("safe", "questionable", "harmful"):
            intent = "safe"
        flagged = bool(parsed.get("flagged", False))
        # Consistency: always flag if Llama says harmful at high+ severity
        if intent == "harmful" and severity in ("high", "critical"):
            flagged = True
        # Don't flag if Llama says safe regardless of what borderline score said
        if intent == "safe":
            flagged = False
            severity = "none"

        logger.info(
            "[LLAMA/MODERATE] flagged=%s severity=%s intent=%s borderline=%s",
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

    # ------------------------------------------------------------------
    # Public API — Categorization
    # ------------------------------------------------------------------

    def categorize(
        self,
        transcript: str,
        categories: list[str],
        tags: list[str],
        keyword_hits: dict[str, float] | None = None,
    ) -> dict:
        if not self._available:
            raise RuntimeError("Llama not loaded")

        kw_hint = ""
        if keyword_hits:
            top = sorted(keyword_hits.items(), key=lambda x: x[1], reverse=True)[:8]
            kw_hint = f"\nKeyword analysis already found: {', '.join(t for t, _ in top)}"

        cat_str = ", ".join(categories[:40])
        tag_str = ", ".join(tags[:100])

        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            "You are an expert audio content categorizer. Return ONLY valid JSON.\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"Available categories (choose from these only): {cat_str}\n"
            f"Available tags (choose from these only, keep # prefix): {tag_str}\n"
            f"{kw_hint}\n\n"
            f"Transcript:\n{transcript[:2000]}\n\n"
            "Return exactly this JSON shape:\n"
            '{"tags":["#Sports"],"categories":["Sports"],"sentiment":"neutral"}\n\n'
            "Rules:\n"
            "- tags: up to 5, must start with #, must come from the available list above\n"
            "- categories: up to 2, must come from the available list above\n"
            "- sentiment: positive | negative | neutral\n"
            "- Base on MAIN topics only — ignore incidental mentions\n"
            "- If multiple distinct topics exist (e.g. sport + food), include both\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

        raw = self._generate(prompt, max_new_tokens=180)
        parsed = self._extract_json(raw)

        valid_tags = set(tags)
        valid_cats = set(categories)

        tags_out = [t for t in parsed.get("tags", []) if t in valid_tags][:5]
        cats_out = [c for c in parsed.get("categories", []) if c in valid_cats][:2]
        sentiment = parsed.get("sentiment", "neutral")
        if sentiment not in ("positive", "negative", "neutral"):
            sentiment = "neutral"

        logger.info("[LLAMA/CATEGORIZE] tags=%s categories=%s", tags_out, cats_out)

        return {"tags": tags_out, "categories": cats_out, "sentiment": sentiment}


llm_service = LlamaService()
